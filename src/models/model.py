"""
Model definitions for dementia detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig

class AcousticEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Initialize acoustic encoder based on wav2vec2."""
        super().__init__()
        self.cfg = cfg
        
        # Load pretrained wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(cfg.model.acoustic.model_name)
        
        # Freeze feature extractor
        if cfg.model.acoustic.freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Freeze specified transformer layers
        if cfg.model.acoustic.num_frozen_layers > 0:
            for layer in self.wav2vec2.encoder.layers[:cfg.model.acoustic.num_frozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Add projection if needed
        if cfg.model.acoustic.output_dim != self.wav2vec2.config.hidden_size:
            self.projection = nn.Linear(
                self.wav2vec2.config.hidden_size,
                cfg.model.acoustic.output_dim
            )
        else:
            self.projection = nn.Identity()
        
        self.dropout = nn.Dropout(cfg.model.acoustic.dropout)
    
    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through acoustic encoder."""
        # Get wav2vec2 outputs
        outputs = self.wav2vec2(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use mean pooling over time dimension
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Project and apply dropout
        encoded = self.dropout(self.projection(pooled))
        
        return encoded

class TextEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Initialize text encoder based on SBERT."""
        super().__init__()
        self.cfg = cfg
        
        # Load pretrained SBERT
        self.sbert = SentenceTransformer(cfg.model.text.model_name)
        
        # Freeze embeddings if specified
        if cfg.model.text.freeze_embeddings:
            for param in self.sbert.parameters():
                param.requires_grad = False
        
        # Add projection if needed
        if cfg.model.text.output_dim != self.sbert.get_sentence_embedding_dimension():
            self.projection = nn.Linear(
                self.sbert.get_sentence_embedding_dimension(),
                cfg.model.text.output_dim
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through text encoder."""
        return self.projection(text_embeddings)

class EarlyFusionModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Initialize early fusion model."""
        super().__init__()
        self.cfg = cfg
        
        # Define encoders
        self.acoustic_encoder = AcousticEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        
        # Calculate total input dimension
        total_dim = cfg.model.acoustic.output_dim + cfg.model.text.output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_dim
        
        for dim in cfg.model.fusion.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU() if cfg.model.fusion.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(cfg.model.fusion.dropout)
            ])
            prev_dim = dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, cfg.model.num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        waveform: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with early fusion."""
        # Get embeddings from both modalities
        acoustic_emb = self.acoustic_encoder(waveform, attention_mask)
        text_emb = self.text_encoder(text_embeddings)
        
        # Concatenate embeddings
        fused = torch.cat([acoustic_emb, text_emb], dim=1)
        
        # Pass through MLP
        logits = self.mlp(fused)
        
        return logits

class LateFusionModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Initialize late fusion model."""
        super().__init__()
        self.cfg = cfg
        
        # Define encoders
        self.acoustic_encoder = AcousticEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        
        # Define separate classifiers for each modality
        self.acoustic_classifier = nn.Linear(cfg.model.acoustic.output_dim, cfg.model.num_classes)
        self.text_classifier = nn.Linear(cfg.model.text.output_dim, cfg.model.num_classes)
        
        # Fusion weights (can be learned or fixed)
        if cfg.model.fusion.type == 'attention':
            self.fusion = nn.MultiheadAttention(
                cfg.model.acoustic.output_dim,
                cfg.model.fusion.num_heads,
                dropout=cfg.model.fusion.dropout
            )
        else:
            self.register_buffer(
                'modality_weights',
                torch.tensor([
                    cfg.model.fusion.acoustic_weight,
                    cfg.model.fusion.text_weight
                ])
            )
    
    def forward(
        self,
        waveform: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with late fusion."""
        # Get embeddings and individual predictions
        acoustic_emb = self.acoustic_encoder(waveform, attention_mask)
        text_emb = self.text_encoder(text_embeddings)
        
        acoustic_logits = self.acoustic_classifier(acoustic_emb)
        text_logits = self.text_classifier(text_emb)
        
        if self.cfg.model.fusion.type == 'attention':
            # Use attention for dynamic fusion
            fused_emb, _ = self.fusion(
                acoustic_emb.unsqueeze(0),
                text_emb.unsqueeze(0),
                text_emb.unsqueeze(0)
            )
            fused_emb = fused_emb.squeeze(0)
            
            # Final classification
            logits = self.acoustic_classifier(fused_emb)  # Reuse acoustic classifier
        else:
            # Weighted average of logits
            logits = (
                self.modality_weights[0] * acoustic_logits +
                self.modality_weights[1] * text_logits
            )
        
        return logits

def create_model(cfg: DictConfig) -> nn.Module:
    """Create model based on config."""
    if cfg.model.type == 'fusion':
        if cfg.model.fusion.type == 'early':
            return EarlyFusionModel(cfg)
        else:
            return LateFusionModel(cfg)
    elif cfg.model.type == 'acoustic_only':
        return AcousticEncoder(cfg)
    elif cfg.model.type == 'text_only':
        return TextEncoder(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
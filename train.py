"""
Training script for dementia detection models.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from src.models.model import create_model
from src.data.dataset import DementiaDataset
from src.features.feature_extractors import AcousticFeatureExtractor, TextFeatureExtractor

def collate_fn(batch_list):
    # batch_list: list of dicts from Dataset.__getitem__
    # Pad waveforms (wav2vec2 expects shape: batch x samples)
    waveforms = [item['waveform'] for item in batch_list]
    max_len = max([w.size(-1) for w in waveforms])
    padded_waveforms = []
    attention_masks = []
    for w in waveforms:
        # Ensure mono by averaging channels if needed
        if w.dim() > 1 and w.size(0) > 1:
            w = w.mean(dim=0)
        # Remove any extra dimensions and ensure shape is (samples,)
        w = w.squeeze()
        pad = max_len - w.size(0)
        if pad > 0:
            w_p = F.pad(w, (0, pad))
        else:
            w_p = w
        padded_waveforms.append(w_p)
        att = torch.ones(w.size(0))
        if pad > 0:
            att = F.pad(att, (0, pad))
        attention_masks.append(att)

    batch_waveform = torch.stack(padded_waveforms)
    batch_attention = torch.stack(attention_masks)

    # Pad log_melspec and mfcc on the time (last) dimension
    def pad_feature(feats_key):
        feats = [item[feats_key] for item in batch_list]
        # feats shape: (channels, F, T)
        max_t = max([f.size(-1) for f in feats])
        padded = []
        for f in feats:
            pad = max_t - f.size(-1)
            if pad > 0:
                # pad last dim
                f_p = F.pad(f, (0, pad))
            else:
                f_p = f
            padded.append(f_p)
        return torch.stack(padded)

    batch_log_melspec = pad_feature('log_melspec')
    batch_mfcc = pad_feature('mfcc')

    # Stack other fixed-size tensors
    prosodic = torch.stack([item['prosodic_features'] for item in batch_list])
    text_embeddings = torch.stack([item['text_embeddings'] for item in batch_list])
    linguistic = torch.stack([item['linguistic_features'] for item in batch_list])
    labels = torch.stack([item['label'] for item in batch_list])
    subjects = [item['subject'] for item in batch_list]

    return {
        'waveform': batch_waveform,
        'attention_mask': batch_attention,
        'log_melspec': batch_log_melspec,
        'mfcc': batch_mfcc,
        'prosodic_features': prosodic,
        'text_embeddings': text_embeddings,
        'linguistic_features': linguistic,
        'label': labels,
        'subject': subjects
    }

class DementiaDetectionModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        """Initialize training module."""
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Create model
        self.model = create_model(cfg)
        
        # Initialize feature extractors
        self.acoustic_extractor = AcousticFeatureExtractor(cfg)
        self.text_extractor = TextFeatureExtractor(cfg)

        # Loss function with class weights
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(cfg.model.class_weights)
        )

        # Metrics (use torchmetrics directly)
        # Use multiclass task with number of classes from config
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=cfg.model.num_classes)

        # For subject-level aggregation
        self.val_preds = {}
        self.val_labels = {}
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(
            batch['waveform'],
            batch['text_embeddings'],
            batch['attention_mask']
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        print(f"[TRAIN STEP] Batch {batch_idx} - Starting forward pass...")
        logits = self(batch)
        print(f"[TRAIN STEP] Batch {batch_idx} - Computing loss...")
        loss = self.criterion(logits, batch['label'])
        
        # Log metrics
        acc = self.train_acc(logits.softmax(dim=-1), batch['label'])
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        print(f"[TRAIN STEP] Batch {batch_idx} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        try:
            print(f"[VAL] Starting validation step {batch_idx}")
            print(f"[VAL] Batch keys: {batch.keys()}")
            print(f"[VAL] Batch sizes: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in batch.items()]}")
            
            print(f"[VAL] Running forward pass...")
            logits = self(batch)
            print(f"[VAL] Logits shape: {logits.shape}")
            
            print(f"[VAL] Computing loss...")
            loss = self.criterion(logits, batch['label'])
            print(f"[VAL] Loss: {loss.item()}")
            
            # Get predictions and probabilities
            print(f"[VAL] Computing predictions...")
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            
            # Log immediate metrics
            print(f"[VAL] Logging metrics...")
            self.log('val_loss', loss)
            self.log('val_acc', self.val_acc(probs, batch['label']))
            self.log('val_auroc', self.val_auroc(probs, batch['label']))
            
            # Store predictions for subject-level aggregation
            print(f"[VAL] Storing predictions...")
            for subject_id, pred, label in zip(batch['subject'], preds, batch['label']):
                if subject_id not in self.val_preds:
                    self.val_preds[subject_id] = []
                    self.val_labels[subject_id] = label.item()
                self.val_preds[subject_id].append(pred.item())
            
            print(f"[VAL] Validation step {batch_idx} completed successfully")
        except Exception as e:
            print(f"[VAL] ERROR in validation step {batch_idx}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def on_validation_epoch_end(self) -> None:
        """Compute subject-level metrics at end of validation."""
        subject_preds = []
        subject_labels = []
        
        for subject_id in self.val_preds:
            # Majority voting for subject-level prediction
            pred_counts = torch.bincount(torch.tensor(self.val_preds[subject_id]))
            subject_pred = pred_counts.argmax().item()
            
            subject_preds.append(subject_pred)
            subject_labels.append(self.val_labels[subject_id])
        
        # Compute and log subject-level metrics
        subject_acc = (torch.tensor(subject_preds) == torch.tensor(subject_labels)).float().mean()
        self.log('val_subject_acc', subject_acc)
        
        # Clear storage
        self.val_preds = {}
        self.val_labels = {}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer (config lives under cfg.training.optimizer)
        if self.cfg.training.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.training.optimizer.lr,
                weight_decay=self.cfg.training.optimizer.weight_decay,
                betas=(self.cfg.training.optimizer.beta1, self.cfg.training.optimizer.beta2)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.training.optimizer.type}")
        
        # Create scheduler
        if self.cfg.training.scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.scheduler.max_steps,
                eta_min=self.cfg.training.optimizer.lr * 0.01
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.cfg.training.scheduler.type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

def train(cfg: DictConfig) -> None:
    """Main training function."""
    pl.seed_everything(cfg.seed)
    
    print(f"[TRAIN] Starting training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load dataset splits
    print("[TRAIN] Loading dataset splits...")
    with open(cfg.data.paths.splits) as f:
        splits = json.load(f)
    print(f"[TRAIN] Loaded splits: train={len(splits['train'])}, val={len(splits['val'])}")
    
    # Create datasets
    print("[TRAIN] Creating train dataset...")
    train_dataset = DementiaDataset(cfg, splits['train'], mode='train')
    print(f"[TRAIN] Train dataset created: {len(train_dataset)} samples")
    
    print("[TRAIN] Creating val dataset...")
    val_dataset = DementiaDataset(cfg, splits['val'], mode='val')
    print(f"[TRAIN] Val dataset created: {len(val_dataset)} samples")
    
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Check your splits and data paths.")
    
    # Test loading one sample
    print("[TRAIN] Testing validation dataset sample loading...")
    try:
        test_sample = val_dataset[0]
        print(f"[TRAIN] Successfully loaded sample with keys: {test_sample.keys()}")
        print(f"[TRAIN] Sample shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in test_sample.items()]}")
    except Exception as e:
        print(f"[TRAIN] ERROR loading validation sample: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Create data loaders
    # Note: collate_fn is defined at module scope to be pickleable on Windows

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Test validation dataloader
    print("[TRAIN] Testing validation dataloader...")
    try:
        val_iter = iter(val_loader)
        test_batch = next(val_iter)
        print(f"[TRAIN] Successfully loaded validation batch with keys: {test_batch.keys()}")
        print(f"[TRAIN] Batch sizes: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in test_batch.items()]}")
        del test_batch, val_iter
    except Exception as e:
        print(f"[TRAIN] ERROR loading validation batch: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Initialize model and training module
    print("[TRAIN] Creating DementiaDetectionModule (this may download pretrained models)...")
    try:
        model = DementiaDetectionModule(cfg)
        print("[TRAIN] Model created successfully")
    except Exception as e:
        print(f"[TRAIN] ERROR creating model: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Set up logging. Make Weights & Biases optional so the script can run
    # even when 'wandb' isn't installed in the environment.
    if cfg.wandb.enabled:
        try:
            import wandb  # optional dependency
        except Exception:
            print("Warning: 'wandb' is not installed. Proceeding without Weights & Biases. To enable, run: pip install wandb")
            logger = True
        else:
            try:
                logger = WandbLogger(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    tags=cfg.wandb.tags
                )
            except Exception as e:
                print(f"Warning: WandbLogger could not be initialized: {e}. Proceeding without Weights & Biases.")
                logger = True
    else:
        logger = True  # Use default logger
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.training.checkpointing.dirpath,
            monitor=cfg.training.training.checkpointing.monitor,
            mode=cfg.training.training.checkpointing.mode,
            save_top_k=cfg.training.training.checkpointing.save_top_k,
            save_last=cfg.training.training.checkpointing.save_last
        ),
        EarlyStopping(
            monitor=cfg.training.training.early_stopping.monitor,
            mode=cfg.training.training.early_stopping.mode,
            patience=cfg.training.training.early_stopping.patience,
            min_delta=cfg.training.training.early_stopping.min_delta
        )
    ]
    
    # Create trainer
    trainer_kwargs = dict(
        max_epochs=cfg.training.training.max_epochs,
        precision=cfg.training.training.precision,
        gradient_clip_val=cfg.training.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0  # Disable sanity check
    )

    if torch.cuda.is_available():
        trainer_kwargs.update(dict(devices=1, accelerator='gpu'))
        print(f"[TRAIN] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[TRAIN] Using CPU")

    print("[TRAIN] Creating trainer...")
    trainer = pl.Trainer(**trainer_kwargs)
    print("[TRAIN] Trainer created successfully")
    
    # Train model
    print("[TRAIN] Starting trainer.fit()...")
    trainer.fit(model, train_loader)
    print("[TRAIN] Training completed successfully!")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    print("="*60, flush=True)
    print("MAIN FUNCTION CALLED", flush=True)
    print("="*60, flush=True)
    # Print config
    print(OmegaConf.to_yaml(cfg), flush=True)
    
    # Train model
    print("\n" + "="*60, flush=True)
    print("CALLING TRAIN FUNCTION", flush=True)
    print("="*60 + "\n", flush=True)
    train(cfg)

if __name__ == '__main__':
    print("Script starting...", flush=True)
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
"""
Dataset class for dementia detection.
"""

import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from omegaconf import DictConfig
from torch.utils.data import Dataset
from src.features.feature_extractors import AcousticFeatureExtractor, TextFeatureExtractor

class DementiaDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        subject_ids: List[str],
        mode: str = 'train'
    ):
        """Initialize dataset."""
        self.cfg = cfg
        self.mode = mode
        # Load manifest
        self.manifest = pd.read_csv(cfg.data.paths.manifest)

        # Filter by subject IDs
        self.manifest = self.manifest[self.manifest['subject'].isin(subject_ids)].reset_index(drop=True)

        # Initialize feature extractors
        self.acoustic_extractor = AcousticFeatureExtractor(cfg)
        self.text_extractor = TextFeatureExtractor(cfg)

        # Set up augmentations for training
        if mode == 'train' and cfg.features.augmentation.spec_augment.enabled:
            # torchaudio.transforms.SpecAugment expects parameters named
            # n_time_masks, n_freq_masks, time_mask_param, freq_mask_param
            self.spec_augment = torchaudio.transforms.SpecAugment(
                n_time_masks=cfg.features.augmentation.spec_augment.num_masks,
                n_freq_masks=cfg.features.augmentation.spec_augment.num_masks,
                time_mask_param=cfg.features.augmentation.spec_augment.time_mask_param,
                freq_mask_param=cfg.features.augmentation.spec_augment.freq_mask_param,
                iid_masks=True
            )
        else:
            self.spec_augment = None
    
    def _load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file from local path."""
        # Load audio. Prefer torchaudio but fall back to librosa if torchcodec is unavailable
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except (ImportError, ModuleNotFoundError) as e:
            # torchcodec backend may be missing on some installations (ModuleNotFoundError: No module named 'torchcodec')
            msg = str(e)
            if 'torchcodec' in msg or 'TorchCodec' in msg:
                try:
                    import librosa
                except Exception:
                    raise

                # librosa.load returns (y, sr). Use mono=False to preserve channels when possible.
                audio_np, sr = librosa.load(file_path, sr=None, mono=False)
                # librosa returns shape (n,) for mono, or (n_channels, n_samples) for multi-channel
                if audio_np.ndim == 1:
                    waveform = torch.from_numpy(audio_np).unsqueeze(0)
                else:
                    # Ensure numpy array is channels x samples
                    waveform = torch.from_numpy(audio_np)
                sample_rate = sr
            else:
                # If it's a different import error, re-raise
                raise

        # Normalize audio if requested
        if self.cfg.data.preprocessing.normalize_audio:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

        # Trim silence (torchaudio may not implement trim in all versions)
        if self.cfg.data.preprocessing.trim_silence:
            try:
                waveform, _ = torchaudio.functional.trim(waveform, top_db=20)
            except Exception:
                import librosa
                audio_np = waveform.numpy().squeeze()
                # If stereo or multi-channel, convert to mono by averaging channels
                if audio_np.ndim > 1:
                    audio_mono = np.mean(audio_np, axis=0)
                else:
                    audio_mono = audio_np
                trimmed, _ = librosa.effects.trim(audio_mono, top_db=20)
                waveform = torch.from_numpy(trimmed).unsqueeze(0)

        # Resample if needed
        if sample_rate != self.cfg.data.preprocessing.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.cfg.data.preprocessing.sample_rate,
            )
            sample_rate = self.cfg.data.preprocessing.sample_rate

        # Apply training-time augmentations
        if self.mode == 'train':
            if self.cfg.features.augmentation.noise.enabled:
                snr_low, snr_high = self.cfg.features.augmentation.noise.snr_range
                snr = np.random.uniform(snr_low, snr_high)
                noise = torch.randn_like(waveform) * (waveform.std() / (10 ** (snr / 20)))
                waveform = waveform + noise

            if self.cfg.features.augmentation.pitch_shift.enabled:
                shift_low, shift_high = self.cfg.features.augmentation.pitch_shift.shift_range
                shift = np.random.uniform(shift_low, shift_high)
                waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, shift)

        # Enforce max duration (samples)
        max_samples = int(self.cfg.data.preprocessing.max_duration * sample_rate)
        if waveform.size(1) > max_samples:
            if self.mode == 'train':
                start = int(torch.randint(0, waveform.size(1) - max_samples + 1, (1,)).item())
                waveform = waveform[:, start:start + max_samples]
            else:
                start = (waveform.size(1) - max_samples) // 2
                waveform = waveform[:, start:start + max_samples]

        # Let the collate function handle reshaping
        return waveform, sample_rate
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        # Get file info
        row = self.manifest.iloc[idx]
        
        # Load and preprocess audio
        waveform, sample_rate = self._load_audio(row['path'])
        
        # Extract features
        acoustic_features = self.acoustic_extractor(waveform, sample_rate)
        text_features = self.text_extractor(waveform, sample_rate)
        
        # Apply SpecAugment if enabled
        if self.spec_augment is not None:
            acoustic_features['log_melspec'] = self.spec_augment(acoustic_features['log_melspec'])
        
        # Create attention mask (all 1s for now, could be modified for variable lengths)
        attention_mask = torch.ones(waveform.size(1))
        
        return {
            'waveform': waveform,
            'attention_mask': attention_mask,
            'log_melspec': acoustic_features['log_melspec'],
            'mfcc': acoustic_features['mfcc'],
            'prosodic_features': torch.tensor([
                acoustic_features['f0'].mean(),
                acoustic_features['energy'].mean(),
                acoustic_features.get('jitter', 0.0),
                acoustic_features.get('shimmer', 0.0)
            ]),
            'text_embeddings': text_features['text_embedding'],
            'linguistic_features': torch.tensor([
                text_features.get('type_token_ratio', 0.0),
                text_features.get('speech_rate', 0.0),
                *[text_features.get(f'pos_ratio_{pos}', 0.0) for pos in ['noun', 'verb', 'adj']]
            ]),
            'label': torch.tensor(row['label']),
            'subject': row['subject']
        }
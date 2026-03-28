"""
Data preparation script for dementia detection project.
- Scans dataset directories
- Creates train/val/test splits
- Generates manifest file
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import torchaudio

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def scan_local_folder(folder_path: Path) -> List[Dict]:
    """Recursively scan local folder for audio files."""
    results = []
    
    for path in folder_path.rglob('*.wav'):
        # Verify it's a valid audio file
        try:
            # Using soundfile to avoid torchaudio deprecation warnings
            import soundfile as sf
            info = sf.info(str(path))
            results.append({
                'path': str(path),
                'subject': path.parent.name,
                'duration': info.duration
            })
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    return results

def create_splits(
    manifest_df: pd.DataFrame,
    cfg: DictConfig
) -> Dict[str, List[str]]:
    """Create subject-level train/val/test splits."""
    subjects = manifest_df['subject'].unique()
    labels = manifest_df.groupby('subject')['label'].first()
    
    # First split: train vs rest
    train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=cfg.data.splits.train_ratio,
        stratify=labels[subjects] if cfg.data.splits.stratify else None,
        random_state=cfg.data.splits.random_state
    )
    
    # Second split: val vs test from remaining subjects
    val_ratio = cfg.data.splits.val_ratio / (cfg.data.splits.val_ratio + cfg.data.splits.test_ratio)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=val_ratio,
        stratify=labels[temp_subjects] if cfg.data.splits.stratify else None,
        random_state=cfg.data.splits.random_state
    )
    
    return {
        'train': train_subjects.tolist(),
        'val': val_subjects.tolist(),
        'test': test_subjects.tolist()
    }

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main data preparation pipeline."""
    # Create data directory if it doesn't exist
    data_dir = Path(cfg.data_dir)
    os.makedirs(data_dir / 'data', exist_ok=True)
    
    print("Configuration:")
    print(f"Paths: {OmegaConf.to_yaml(cfg.data.paths)}")
    print(f"Splits: {OmegaConf.to_yaml(cfg.data.splits)}")
    
    # Scan datasets
    print("\nScanning dementia dataset...")
    dementia_files = scan_local_folder(Path(cfg.data.paths.dementia))
    print("Scanning control dataset...")
    control_files = scan_local_folder(Path(cfg.data.paths.nodementia))
    
    # Create manifest DataFrame
    manifest_data = []
    
    for files, label in [(dementia_files, 1), (control_files, 0)]:
        for file_info in files:
            manifest_data.append({
                'path': file_info['path'],
                'subject': file_info['subject'],
                'duration': file_info['duration'],
                'label': label
            })
    
    manifest_df = pd.DataFrame(manifest_data)
    
    # Create splits
    splits = create_splits(manifest_df, cfg)
    
    # Save manifest and splits
    os.makedirs('data', exist_ok=True)
    manifest_df.to_csv(cfg.data.paths.manifest, index=False)
    
    with open(cfg.data.paths.splits, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created manifest with {len(manifest_df)} files")
    print(f"Train: {len(splits['train'])} subjects")
    print(f"Val: {len(splits['val'])} subjects")
    print(f"Test: {len(splits['test'])} subjects")

if __name__ == '__main__':
    main()
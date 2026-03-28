"""
Feature extraction script for dementia detection project.
Extracts and caches both acoustic and text features from audio files.
"""

import os
import sys
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import traceback
from src.features.feature_extractors import AcousticFeatureExtractor, TextFeatureExtractor

logger = logging.getLogger(__name__)
# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger.setLevel(logging.INFO)

def save_features(features: Dict[str, torch.Tensor], save_path: Path) -> None:
    """Save features to disk."""
    os.makedirs(save_path.parent, exist_ok=True)
    torch.save(features, save_path)

def extract_and_save_features(
    file_path: str,
    acoustic_extractor: AcousticFeatureExtractor,
    text_extractor: TextFeatureExtractor,
    save_dir: Path,
    cfg: DictConfig
) -> Dict[str, Union[str, float]]:
    """Extract and save features for a single audio file."""
    
    # Create output paths
    file_id = Path(file_path).stem
    acoustic_path = save_dir / 'acoustic' / f"{file_id}.pt"
    text_path = save_dir / 'text' / f"{file_id}.pt"
    
    feature_info = {
        'file_path': file_path,
        'acoustic_path': str(acoustic_path),
        'text_path': str(text_path)
    }
    
    # Skip if features already exist
    if acoustic_path.exists() and text_path.exists() and cfg.features.cache.use_cache:
        msg = f"Skipping {file_path} — cached features exist"
        logger.info(msg)
        print(msg)
        return feature_info
    
    try:
        msg = f"Processing {file_path}"
        logger.info(msg)
        print(msg)
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if needed
        if sample_rate != cfg.data.preprocessing.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, cfg.data.preprocessing.sample_rate)(waveform)
            sample_rate = cfg.data.preprocessing.sample_rate

        # Extract acoustic features
        acoustic_features = acoustic_extractor(waveform, sample_rate)
        save_features(acoustic_features, acoustic_path)

        # Extract text features
        text_features = text_extractor(waveform, sample_rate)
        save_features(text_features, text_path)

        # Add feature stats to info
        feature_info.update({
            'duration': waveform.size(1) / sample_rate,
            'acoustic_dim': acoustic_features['log_melspec'].size(1),
            'text_dim': text_features['text_embedding'].size(0)
        })

        msg = f"Successfully extracted features for {file_path}"
        logger.info(msg)
        print(msg)
        return feature_info

    except Exception as e:
        # Log full traceback for easier debugging
        tb = traceback.format_exc()
        err_msg = f"Error processing {file_path}: {str(e)}"
        logger.error(err_msg)
        print(err_msg)
        print(tb)
        return None

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main feature extraction pipeline."""
    
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))
    
    # Load manifest
    manifest_df = pd.read_csv(cfg.data.paths.manifest)
    
    # Create feature extractors
    acoustic_extractor = AcousticFeatureExtractor(cfg)
    text_extractor = TextFeatureExtractor(cfg)
    
    # Create feature cache directory
    feature_dir = Path(cfg.features.cache.cache_dir)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Process all files
    print("\nExtracting features...")
    print(f"Found {len(manifest_df)} files in manifest")
    print(f"First few file paths:")
    print(manifest_df['path'].head())
    print("\nChecking file existence...")
    
    existing_files = manifest_df['path'].apply(os.path.exists)
    print(f"Number of existing files: {existing_files.sum()} out of {len(manifest_df)}")
    
    if existing_files.sum() == 0:
        print("\nNo files found at the paths in manifest. Example path check:")
        example_path = manifest_df['path'].iloc[0]
        print(f"Path: {example_path}")
        print(f"Exists: {os.path.exists(example_path)}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(example_path))}")
        return

    feature_info_list = []
    
    for file_path in tqdm(manifest_df[existing_files]['path']):
        info = extract_and_save_features(
            file_path,
            acoustic_extractor,
            text_extractor,
            feature_dir,
            cfg
        )
        if info is not None:
            feature_info_list.append(info)
    
    # Save feature info
    if feature_info_list:
        feature_info_df = pd.DataFrame(feature_info_list)
        feature_info_df.to_csv(feature_dir / 'feature_info.csv', index=False)
    
    print(f"\nFeatures extracted for {len(feature_info_list)} files")
    print(f"Features saved to {feature_dir}")

if __name__ == '__main__':
    main()
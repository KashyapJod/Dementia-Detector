"""
Precompute Whisper transcriptions for all audio files to speed up training.
This avoids running Whisper for every training epoch.
"""

import sys
from pathlib import Path
import json

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# Add parent to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
import pandas as pd
from omegaconf import OmegaConf
import hydra
from src.features.feature_extractors import TextFeatureExtractor

def precompute_transcriptions(cfg, manifest_path: str):
    """Precompute transcriptions for all audio files in the manifest."""
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"[PRECOMPUTE] Loaded manifest with {len(manifest)} audio files")
    
    # Initialize text extractor (loads Whisper models)
    print("[PRECOMPUTE] Initializing Whisper ASR (first time may download models)...")
    text_extractor = TextFeatureExtractor(cfg)
    
    # Precompute transcriptions
    print(f"[PRECOMPUTE] Starting transcription of {len(manifest)} files...")
    for idx, row in manifest.iterrows():
        file_path = row['path']
        if not Path(file_path).exists():
            print(f"  [{idx+1}/{len(manifest)}] SKIP {file_path} (file not found)")
            continue
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Get audio hash for cache key
            audio_hash = text_extractor._get_audio_hash(waveform)
            
            # Check if already cached
            if audio_hash in text_extractor.transcription_cache:
                print(f"  [{idx+1}/{len(manifest)}] CACHED {Path(file_path).name}")
            else:
                # Transcribe (this will cache automatically)
                transcript = text_extractor.transcribe_audio(waveform, sample_rate)
                print(f"  [{idx+1}/{len(manifest)}] OK {Path(file_path).name}: {transcript[:50]}...")
        except Exception as e:
            print(f"  [{idx+1}/{len(manifest)}] ERROR {file_path}: {e}")
    
    # Save cache
    print(f"[PRECOMPUTE] Saving cache with {len(text_extractor.transcription_cache)} transcriptions...")
    text_extractor._save_cache()
    print("[PRECOMPUTE] Done!")

if __name__ == '__main__':
    # Use Hydra to load config with proper interpolation
    @hydra.main(config_path="../configs", config_name="config", version_base=None)
    def run(cfg):
        print(f"[PRECOMPUTE] Config loaded via Hydra")
        print(f"[PRECOMPUTE] data_dir: {cfg.data_dir}")
        
        # Resolve manifest path
        manifest_path = Path(cfg.data.paths.manifest)
        print(f"[PRECOMPUTE] Manifest: {manifest_path}")
        
        precompute_transcriptions(cfg, str(manifest_path))
    
    run()

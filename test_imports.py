"""Test imports to find which one is failing."""
import sys
print(f"Python {sys.version}")

imports_to_test = [
    ("os", "import os"),
    ("json", "import json"),
    ("torch", "import torch"),
    ("torch.nn.functional", "import torch.nn.functional as F"),
    ("pytorch_lightning", "import pytorch_lightning as pl"),
    ("pytorch_lightning.callbacks", "from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping"),
    ("torchmetrics", "import torchmetrics"),
    ("pytorch_lightning.loggers", "from pytorch_lightning.loggers import WandbLogger"),
    ("pandas", "import pandas as pd"),
    ("pathlib", "from pathlib import Path"),
    ("typing", "from typing import Dict, List, Optional, Tuple"),
    ("omegaconf", "from omegaconf import DictConfig, OmegaConf"),
    ("hydra", "import hydra"),
    ("hydra.utils", "from hydra.utils import instantiate"),
]

for name, import_stmt in imports_to_test:
    try:
        print(f"Testing: {name}...", end=" ")
        exec(import_stmt)
        print("✓")
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")

print("\nTesting custom modules...")
try:
    print("Testing src.models.model...", end=" ")
    from src.models.model import create_model
    print("✓")
except Exception as e:
    print(f"✗ {type(e).__name__}: {e}")

try:
    print("Testing src.data.dataset...", end=" ")
    from src.data.dataset import DementiaDataset
    print("✓")
except Exception as e:
    print(f"✗ {type(e).__name__}: {e}")

try:
    print("Testing src.features.feature_extractors...", end=" ")
    from src.features.feature_extractors import AcousticFeatureExtractor, TextFeatureExtractor
    print("✓")
except Exception as e:
    print(f"✗ {type(e).__name__}: {e}")

print("\nAll imports tested!")

"""Test script to debug training issues."""
import sys
print(f"Python version: {sys.version}", flush=True)
print(f"Python executable: {sys.executable}", flush=True)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"Error importing torch: {e}", flush=True)

try:
    import pytorch_lightning as pl
    print(f"PyTorch Lightning version: {pl.__version__}", flush=True)
except Exception as e:
    print(f"Error importing pytorch_lightning: {e}", flush=True)

try:
    from omegaconf import DictConfig, OmegaConf
    import hydra
    print(f"Hydra and OmegaConf imported successfully", flush=True)
except Exception as e:
    print(f"Error importing hydra/omegaconf: {e}", flush=True)

print("\nAttempting to load config...", flush=True)
try:
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_path="configs", config_name="config", version_base=None)
    def test_config(cfg: DictConfig):
        print(f"Config loaded successfully!", flush=True)
        print(f"\nConfig content:\n{OmegaConf.to_yaml(cfg)}", flush=True)
        return cfg
    
    test_config()
except Exception as e:
    print(f"Error loading config: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

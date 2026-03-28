import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import json
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from src.data.dataset import DementiaDataset

@hydra.main(config_path="configs", config_name="config", version_base=None)
def test_data(cfg: DictConfig):
    """Test dataset loading."""

    print("Config loaded successfully")
    print(f"Data dir: {cfg.data_dir}")

    # Load splits
    with open(cfg.data.paths.splits) as f:
        splits = json.load(f)

    print(f"\nCreating validation dataset with {len(splits['val'])} subjects...")

    try:
        val_dataset = DementiaDataset(cfg, splits['val'], mode='val')
        print(f"✅ Dataset created successfully: {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ ERROR creating dataset: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading first sample
    print("\nTesting sample loading...")
    try:
        sample = val_dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"Sample keys: {sample.keys()}")
        print(f"\nSample shapes:")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item()}")
            else:
                print(f"  {k}: {type(v).__name__}")
    except Exception as e:
        print(f"❌ ERROR loading sample: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading multiple samples
    print("\nTesting multiple samples...")
    errors = []
    for i in range(min(5, len(val_dataset))):
        try:
            sample = val_dataset[i]
            print(f"  ✅ Sample {i}: subject='{sample['subject']}', waveform shape={sample['waveform'].shape}")
        except Exception as e:
            print(f"  ❌ Sample {i} failed: {type(e).__name__}: {e}")
            errors.append((i, e))

    if errors:
        print(f"\n❌ {len(errors)} samples failed to load")
        print("\nFirst error details:")
        import traceback
        traceback.print_exception(type(errors[0][1]), errors[0][1], errors[0][1].__traceback__)
        return
    else:
        print(f"\n✅ All {min(5, len(val_dataset))} samples loaded successfully!")

    # Test dataloader with collate_fn
    print("\nTesting dataloader with collate function...")
    from train import collate_fn

    try:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        print("✅ DataLoader created successfully")
        
        # Get first batch
        batch = next(iter(val_loader))
        print(f"✅ Batch loaded successfully")
        print(f"Batch keys: {batch.keys()}")
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)} (length {len(v) if hasattr(v, '__len__') else 'N/A'})")
        
    except Exception as e:
        print(f"❌ ERROR with dataloader: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)

if __name__ == '__main__':
    test_data()

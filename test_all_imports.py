"""Test all imports step by step."""
import sys

print("="*60)
print("Testing imports step by step")
print("="*60)

# Test 1
try:
    print("\n1. Testing basic imports...")
    import os, json, torch
    print("   ✓ Basic imports OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2
try:
    print("\n2. Testing transformers...")
    from transformers import Wav2Vec2Model
    print("   ✓ Transformers OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3
try:
    print("\n3. Testing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print("   ✓ Sentence-transformers OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4
try:
    print("\n4. Testing src.models.model...")
    from src.models.model import create_model
    print("   ✓ Model import OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5
try:
    print("\n5. Testing src.data.dataset...")
    from src.data.dataset import DementiaDataset
    print("   ✓ Dataset import OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6
try:
    print("\n6. Testing src.features.feature_extractors...")
    from src.features.feature_extractors import AcousticFeatureExtractor, TextFeatureExtractor
    print("   ✓ Feature extractors OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Import testing complete!")
print("="*60)

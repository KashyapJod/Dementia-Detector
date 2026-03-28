"""Simple test."""
import sys
print("Starting test...", flush=True)
print(f"Python: {sys.version}", flush=True)

try:
    import torch
    print(f"PyTorch: {torch.__version__}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    
print("Test complete", flush=True)

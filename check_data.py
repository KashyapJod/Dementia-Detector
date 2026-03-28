import pandas as pd
import json
from pathlib import Path

# Load manifest
manifest = pd.read_csv('data/manifest.csv')
print(f"Total rows in manifest: {len(manifest)}")
print(f"Unique subjects: {manifest['subject'].nunique()}")
print(f"\nLabel distribution:")
print(manifest['label'].value_counts())

# Load splits
with open('data/splits.json') as f:
    splits = json.load(f)

print(f"\nSplit sizes:")
print(f"  Train: {len(splits['train'])} subjects")
print(f"  Val: {len(splits['val'])} subjects")
print(f"  Test: {len(splits['test'])} subjects")

# Check validation subjects in manifest
val_df = manifest[manifest['subject'].isin(splits['val'])]
print(f"\nValidation data:")
print(f"  Subjects in manifest: {val_df['subject'].nunique()}")
print(f"  Total samples: {len(val_df)}")

# Find missing subjects
missing = set(splits['val']) - set(val_df['subject'].unique())
if missing:
    print(f"\n⚠️ Missing validation subjects in manifest: {missing}")
else:
    print("\n✅ All validation subjects found in manifest")

# Check if files exist
print("\nChecking first 5 validation file paths:")
for i, row in val_df.head(5).iterrows():
    exists = Path(row['path']).exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {row['path']}")

# Check for audio duration issues
print(f"\nAudio duration statistics:")
print(f"  Min: {manifest['duration'].min():.2f}s")
print(f"  Max: {manifest['duration'].max():.2f}s")
print(f"  Mean: {manifest['duration'].mean():.2f}s")
print(f"  Median: {manifest['duration'].median():.2f}s")

# Check validation set specifically
print(f"\nValidation set duration statistics:")
print(f"  Min: {val_df['duration'].min():.2f}s")
print(f"  Max: {val_df['duration'].max():.2f}s")
print(f"  Mean: {val_df['duration'].mean():.2f}s")

# Count samples per validation subject
print(f"\nSamples per validation subject:")
val_counts = val_df['subject'].value_counts()
print(val_counts)

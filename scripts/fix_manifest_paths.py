"""
Fix manifest paths from Linux to Windows paths.
"""
import pandas as pd
from pathlib import Path
import sys

def fix_manifest_paths(manifest_path, old_base, new_base):
    """Update file paths in manifest from old_base to new_base."""
    # Read manifest
    df = pd.read_csv(manifest_path)
    print(f"[FIX] Loaded manifest with {len(df)} rows")
    print(f"[FIX] Old base: {old_base}")
    print(f"[FIX] New base: {new_base}")
    
    # Replace paths
    df['path'] = df['path'].str.replace(old_base, new_base, regex=False)
    
    # Normalize path separators to forward slashes (works on Windows too)
    df['path'] = df['path'].str.replace('\\', '/', regex=False)
    
    # Save updated manifest
    df.to_csv(manifest_path, index=False)
    print(f"[FIX] Updated manifest saved to {manifest_path}")
    
    # Show sample
    print(f"\n[FIX] Sample paths after update:")
    print(df['path'].head(3).to_list())
    
    # Check if files exist
    existing = 0
    for path in df['path']:
        if Path(path).exists():
            existing += 1
    
    print(f"\n[FIX] {existing}/{len(df)} files found on disk")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix manifest file paths.')
    parser.add_argument('--manifest', default='data/manifest.csv', help='Path to manifest CSV')
    parser.add_argument('--old_base', default='/home/karnito/Coding/Demdect', help='Old base path')
    parser.add_argument('--new_base', default='C:/Coding/BioPro', help='New base path')
    
    args = parser.parse_args()
    
    fix_manifest_paths(args.manifest, args.old_base, args.new_base)

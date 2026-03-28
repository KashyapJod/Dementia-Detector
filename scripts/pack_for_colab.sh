#!/usr/bin/env bash
# Create a zip package of the repository ready for upload to Colab/Drive
# Excludes virtual environments and results directories by default

set -euo pipefail
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
OUT=demdect_colab_package.zip

echo "Packing repository from: $ROOT_DIR"

# Files/dirs to exclude
EXCLUDES=(
  "*.pyc"
  "__pycache__"
  ".venv"
  ".git"
  "results"
  "wandb"
  "venv"
)

EXCLUDE_ARGS=()
for e in "${EXCLUDES[@]}"; do
  EXCLUDE_ARGS+=(--exclude="$e")
done

# Create the zip
cd "$ROOT_DIR"
# Use zip with globbing
zip -r "$OUT" . ${EXCLUDE_ARGS[@]}

echo "Created $OUT - upload this to Colab or Drive and extract in /content"
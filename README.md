# Dementia-Detector

Multimodal dementia detection from speech using acoustic and linguistic features.

GitHub repository: https://github.com/KashyapJod/Dementia-Detector

## What Is Included In Git

This repository is configured to upload code and configs only. Large and private artifacts are ignored by default:

- local datasets in `dementia/` and `nodementia/`
- experiment outputs (`checkpoints/`, `lightning_logs/`, `wandb/`, `results/`, `explain_output/`)
- local installers and temporary files

See `.gitignore` for the exact rules.

## Quick Setup (Windows, PowerShell)

1. Create and activate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Keep dataset folders in the project root:

```text
BioPro/
  dementia/
  nodementia/
  data/manifest.csv
  data/splits.json
```

## Fastest Training Settings

Two fast presets are added:

- `configs/training/fastest.yaml`
- `configs/data/fastest.yaml`

Run fastest training:

```powershell
python train.py training=fastest data=fastest wandb.enabled=false
```

This configuration is intended for quick checks and notebook demonstration:

- `max_epochs: 1`
- `batch_size: 1`
- `max_duration: 3.0`
- reduced scheduler and explainability sampling

## Notebook Workflow

Main notebook:

- `dementia_detection_colab.ipynb`

Recommended order:

1. Run environment and import/setup cells.
2. Verify manifest and split files are loaded.
3. Train with fast preset:

```python
!python train.py training=fastest data=fastest wandb.enabled=false
```

4. Run explainability on the latest checkpoint:

```python
!python explain.py model_checkpoint=checkpoints/last.ckpt
```

5. Save outputs to `explain_output/` and review charts/tables.

## Google Colab Instructions

Use these steps to run this project in Google Colab using your Google Drive BioPro folder.

Drive folder link:

- https://drive.google.com/drive/folders/1EdOfb7WdriQ7HYsQDTgr0NIbhKDMMNdj?usp=sharing

1. Open the Drive link and add a shortcut to your My Drive as `BioPro`.

2. Open Colab and install dependencies:

```python
!pip install -U pip
!pip install -r requirements.txt
```

3. Mount Google Drive and switch to your BioPro folder:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/BioPro
```

4. (Optional) If you have GPU runtime enabled in Colab:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

5. Confirm dataset layout in that folder:

```text
BioPro/
  dementia/
  nodementia/
  data/manifest.csv
  data/splits.json
```

6. Run fastest training preset (optional):

```python
!python train.py training=fastest data=fastest wandb.enabled=false
```

7. Run explainability using your uploaded checkpoint (example uses `last-v1.ckpt`):

```python
!python explain.py checkpoint_path=checkpoints/last-v1.ckpt num_samples=10
```

## Upload To GitHub

Target repository:

- https://github.com/KashyapJod/Dementia-Detector

From the project root:

```powershell
git init
git add .
git commit -m "Initial clean upload with fast presets and notebook README"
git branch -M main
git remote add origin https://github.com/KashyapJod/Dementia-Detector.git
git push -u origin main
```

If the remote already exists, use:

```powershell
git remote set-url origin https://github.com/KashyapJod/Dementia-Detector.git
git push -u origin main
```

## Notes

- Use Python 3.11 for best compatibility.
- For full-quality training, switch back to `training=default data=default`.

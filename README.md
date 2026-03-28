# Dementia-Detector

Multimodal dementia detection from speech using acoustic and linguistic features.

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

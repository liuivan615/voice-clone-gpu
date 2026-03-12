# Data Setup

This repository does not include datasets, uploaded media, task outputs, or workspace databases.

## 1. Default data locations

By default the app uses these writable directories:

- `workspace_data/`
- `jobs/`
- `trained_models/`

You can override the main data root in either of these ways:

- Edit `runtime_settings.yaml`
- Set `WAV_DATA_ROOT` in `.env`

## 2. What the app creates automatically

When the application runs, it will create:

- `workspace_data/library/workspace.sqlite3`
- `workspace_data/datasets/files/`
- `workspace_data/datasets/segments/`
- `workspace_data/datasets/versions/`
- `workspace_data/models/`
- `workspace_data/runs/`
- `jobs/` task folders for inference, preprocessing, and training

These are runtime-generated working directories and should stay out of Git.

## 3. Expected user data flow

### Web UI path

The main workflow is driven from the browser UI:

1. Create a dataset.
2. Upload source WAV files.
3. Review auto-segmented clips.
4. Freeze a dataset version.
5. Train a model from that dataset version.
6. Load a model version for inference.

### Smoke test path

For the lightweight training smoke test wrapper, provide a directory that contains `.wav` files, for example:

```text
your_dataset/
  sample_001.wav
  sample_002.wav
  sample_003.wav
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test_train_smoke.ps1 -SourceDir .\sample_wavs
```

## 4. Data that is intentionally excluded from this repo

- Raw datasets
- Segmented datasets
- Uploaded inference media
- Generated results
- SQLite workspace databases
- Temporary job folders
- Trained checkpoints

## 5. Notes

- The current upload endpoints are WAV-first for dataset training input.
- Preprocessing accepts common audio/video files, but inference/training outputs are generated locally and not versioned.
- If you want to keep work products outside the repo tree, set `WAV_DATA_ROOT` to another path in `.env`.

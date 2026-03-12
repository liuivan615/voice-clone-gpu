# Signal Foundry Workstation

A cleaned GitHub-ready export of the original portable `4wav_gpu_frontend_portable` workspace.

This project is a local web workstation for dataset management, media preprocessing, model training,
and inference around a SoftVC-style voice conversion runtime. It preserves the application code and
the trimmed runtime source tree, but intentionally excludes bundled Python environments, model weights,
datasets, caches, logs, and generated outputs.

---

## What This Project Does

- Provides a FastAPI backend plus a browser UI for an end-to-end workflow.
- Manages datasets, file uploads, auto-segmentation, and dataset version snapshots.
- Supports media preprocessing with optional vocal separation.
- Supports training flows for:
  - new training
  - resume training
  - diffusion-only training
- Supports inference from uploaded audio or preprocessed audio variants.
- Tracks model versions and checkpoints in a local workspace library.

---

## What Is Included

- Core application code in `app/`
- Frontend assets in `app/static/`
- Trimmed runtime source code in `runtime/softvc_clean/`
- Generic startup and setup scripts in `tools/`
- Safe example configuration files and setup documentation

## What Is Not Included

The following resource classes were deliberately removed from the GitHub export:

- Portable Python environments (`envs/`)
- Task outputs and logs (`jobs/`)
- Workspace databases and generated model library (`workspace_data/`)
- Legacy trained model bundles (`trained_models/`)
- Runtime pretrained assets and checkpoints
- Separator model files
- ffmpeg binaries
- Uploaded or generated media files

See [docs/model_setup.md](docs/model_setup.md) and [docs/data_setup.md](docs/data_setup.md) for the
missing runtime resources you must provide locally.

---

## Project Structure

```text
app/                        FastAPI app, API routes, services, frontend assets
tools/                      Generic setup, launch, and smoke-test scripts
runtime/softvc_clean/       Trimmed runtime source tree and configs
configs/                    Example configuration files
docs/                       Model and data setup documentation
runtime_settings.yaml       Default local runtime settings
.env.example                Optional environment-variable overrides
requirements.txt            Python dependencies for the cleaned repository
```

---

## Environment Requirements

- **OS:** Windows (validated target of the original portable workspace)
- **Python:** 3.10+
- **GPU:** NVIDIA recommended for training and inference
- **CUDA:** For RTX 50 / Blackwell, PyTorch with CUDA 12.8+ and driver `570.65+`
- **ffmpeg:** Required for preprocessing (provide via PATH or `WAV_FFMPEG_EXE`)

---

## Installation

### 1. Create a virtual environment and install dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1
```

To also install the optional vocal separation CLI into `.venv`:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1 -WithPreprocessTools
```

### 2. Verify or replace the PyTorch build

`requirements.txt` leaves `torch` and `torchaudio` unpinned so you can install the correct
CPU or CUDA wheel for your machine after the initial setup.

### 3. Configure optional environment overrides

Copy `.env.example` to `.env` to override runtime paths. The app loads `.env` automatically.

Common variables:

| Variable | Purpose |
|---|---|
| `WAV_RUNTIME_ROOT` | Root of the runtime source tree |
| `WAV_DATA_ROOT` | Workspace data directory |
| `WAV_SEPARATOR_EXE` | Path to the separator executable |
| `WAV_FFMPEG_EXE` | Path to ffmpeg |

---

## Model Preparation

This repository does not ship the runtime assets needed for training or preprocessing.

Before using training or separator features, prepare the assets described in
[docs/model_setup.md](docs/model_setup.md). At minimum:

- Encoder feature assets under `runtime/softvc_clean/pretrain/`
- Base generator/discriminator checkpoints under `runtime/softvc_clean/pre_trained_model/`
- Separator models under `runtime/separators/models/`
- `ffmpeg` in PATH or via `WAV_FFMPEG_EXE`

---

## Data Preparation

The repo ships no dataset or media content. Normal workflow:

1. Start the app.
2. Open the **Dataset** tab.
3. Create a dataset entry.
4. Upload WAV training files.
5. Review generated segments.
6. Freeze a dataset version.

See [docs/data_setup.md](docs/data_setup.md) for details.

---

## How To Start

```powershell
.\run.bat
```

Or directly via PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\launch_app.ps1
```

Default address: `http://127.0.0.1:8000`

---

## Training Workflow

1. Prepare a dataset and freeze a version.
2. Open the **Training** tab.
3. Choose a dataset version and model name.
4. Choose a preset: `balanced` / `high_quality` / `light`
5. Start one of:
   - full retraining
   - resume training
   - diffusion-only

The backend copies the dataset, resamples audio, writes configs, extracts features, trains the model,
and registers the result into the local model library.

### Optional smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test_train_smoke.ps1 -SourceDir .\sample_wavs
```

Requires runtime assets from [docs/model_setup.md](docs/model_setup.md).

---

## Inference Workflow

1. Open the **Inference** tab.
2. Load a model version from the local model library.
3. Upload an audio or video file.
4. Optionally run preprocessing to extract vocals.
5. Choose the original or vocal-only variant.
6. Start inference and download the result.

---

## Configuration

### `runtime_settings.yaml`

Controls host/port, upload limits, default data root, training defaults, and inference profile fields.

### `.env`

Controls path overrides: runtime root, workspace data root, jobs directory, separator executable,
ffmpeg executable. See `.env.example`.

---

## Notes

- The cleaned repository keeps application logic, but not the original portable environment bundle.
- Training and preprocessing will not work until you supply the required external assets.
- Only the bundled runtime component license file already present in the source was preserved.
- Review the upstream source and license policy for the `softvc_clean` runtime code before redistribution.

---

## Contributors

Built with [Claude Code](https://claude.ai/claude-code) (Anthropic) and [Codex](https://openai.com/blog/openai-codex) (OpenAI).

# Signal Foundry Workstation

A cleaned GitHub-ready export of the original portable `4wav_gpu_frontend_portable` workspace.

This project is a local web workstation for dataset management, media preprocessing, model training, and inference around a SoftVC-style voice conversion runtime. It preserves the application code and the trimmed runtime source tree, but intentionally excludes bundled Python environments, model weights, datasets, caches, logs, and generated outputs.

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

See [docs/model_setup.md](docs/model_setup.md) and [docs/data_setup.md](docs/data_setup.md) for the missing runtime resources you must provide locally.

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

## Environment Requirements

- Windows is the validated target in the original portable workspace.
- Python 3.10+ is recommended.
- NVIDIA GPU is recommended for training and inference.
- For RTX 50 / Blackwell GPUs, the code expects a PyTorch build with CUDA 12.8+ and an NVIDIA driver at least `570.65`.
- `ffmpeg` is required for preprocessing.

## Installation

### 1. Create a virtual environment and install dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1
```

If you also want the optional vocal separation CLI installed into `.venv`, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1 -WithPreprocessTools
```

### 2. Verify or replace the PyTorch build if needed

`requirements.txt` intentionally leaves `torch` and `torchaudio` unpinned so you can replace them with the correct CPU or CUDA build for your machine. If you need a specific CUDA wheel, install it into `.venv` after the initial setup.

### 3. Configure optional environment overrides

Copy `.env.example` to `.env` if you want to override runtime paths. The app loads `.env` automatically.

Common variables:

- `WAV_RUNTIME_ROOT`
- `WAV_DATA_ROOT`
- `WAV_SEPARATOR_EXE`
- `WAV_FFMPEG_EXE`

## Model Preparation

This repository does not ship the runtime assets needed for training or preprocessing.

Before using training or separator features, prepare the assets described in:

- [docs/model_setup.md](docs/model_setup.md)

At minimum, you should expect to provide:

- encoder feature assets under `runtime/softvc_clean/pretrain/`
- base generator/discriminator checkpoints under `runtime/softvc_clean/pre_trained_model/`
- separator models under `runtime/separators/models/`
- `ffmpeg` in `PATH` or via `WAV_FFMPEG_EXE`

## Data Preparation

The repo ships no dataset or media content.

The normal workflow is:

1. Start the app.
2. Open the Dataset tab.
3. Create a dataset entry.
4. Upload WAV training files.
5. Review generated segments.
6. Freeze a dataset version.

More detail is available in:

- [docs/data_setup.md](docs/data_setup.md)

## How To Start

### Fast path

```powershell
.\run.bat
```

### Direct PowerShell path

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\launch_app.ps1
```

The default local address is:

- `http://127.0.0.1:8000`

## Training Workflow

The training flow is driven from the browser UI:

1. Prepare a dataset and freeze a dataset version.
2. Open the Training tab.
3. Choose a dataset version and model name.
4. Choose a preset:
   - `balanced`
   - `high_quality`
   - `light`
5. Start one of these modes:
   - full retraining
   - resume training
   - diffusion-only

The backend then:

- copies the selected dataset version into a local training workspace
- resamples and writes training filelists
- writes runtime configs
- extracts features
- trains the main model and optionally diffusion
- registers the result into the local model library

### Optional smoke test

For a small bounded smoke test with local WAV samples:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test_train_smoke.ps1 -SourceDir .\sample_wavs
```

This smoke path still requires the runtime source plus the model assets documented in [docs/model_setup.md](docs/model_setup.md).

## Inference Workflow

1. Open the Inference tab.
2. Load a model version from the local model library.
3. Upload an audio or video file.
4. Optionally run preprocessing to extract vocals.
5. Choose the original or vocal-only variant.
6. Start inference and download the generated result.

## Configuration

Two configuration layers are available:

### `runtime_settings.yaml`

Controls:

- host and port
- upload limits
- default data root
- training defaults
- default inference profile fields

### `.env`

Controls path overrides, for example:

- runtime root
- workspace data root
- jobs directory
- separator executable
- ffmpeg executable

An example is provided in `.env.example`.

## Resources Excluded From The Repository

The following were intentionally excluded from this GitHub-ready export:

- bundled interpreters and portable environments
- cached runtime bundles
- generated model checkpoints
- datasets and workspace databases
- task logs and outputs
- large pretrained assets
- large separator models
- local absolute-path bootstrap scripts

## Notes And Caveats

- The cleaned repository keeps the application logic, but not the original portable "just run anywhere" environment bundle.
- The web UI and service layer are preserved, but training/preprocessing will not work until you supply the required external assets.
- The root repository license is not inferred automatically here. Only the bundled runtime component license file that already existed in the source runtime was preserved.

## Still Worth Reviewing Manually

- Final dependency versions for your target GPU stack
- The upstream source and license policy for the `softvc_clean` runtime code
- Whether you want to keep the smoke-test helpers in this public repo

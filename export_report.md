# GitHub Export Report

## 1. Source Scan Summary

Original source directory scanned:

- original portable workspace root

Observed top-level layout:

- `app/` - application source
- `tools/` - maintenance and launch scripts
- `runtime/` - bundled runtime source, pretrained assets, ffmpeg, separator support
- `envs/` - portable Python environments
- `jobs/` - task outputs, logs, temp workspaces
- `workspace_data/` - local dataset/model workspace and SQLite database
- `trained_models/` - legacy trained model exports

Large directories identified during scan:

- `jobs/` about 16.87 GB
- `envs/` about 15.72 GB
- `runtime/` about 15.17 GB
- `workspace_data/` about 6.63 GB
- `trained_models/` about 0.58 GB

Conclusion:

The original directory was a portable runtime bundle, not a publish-ready source repository.

## 2. Core Modules Preserved

The following functional areas were identified as the core project and preserved:

- FastAPI app entry and API routes in `app/main.py`
- repository-relative path/config handling in `app/config.py`
- dataset/model library service in `app/services/library_service.py`
- runtime model loading in `app/services/model_service.py`
- inference pipeline in `app/services/pipeline_service.py`
- preprocessing pipeline in `app/services/preprocess_service.py`
- training pipeline in `app/services/training_service.py`
- training presets and GPU checks in `app/services/training_preset_service.py` and `app/services/gpu_compat_service.py`
- browser frontend in `app/static/`
- trimmed runtime source tree in `runtime/softvc_clean/`

## 3. Files And Directories Kept In The Export

Kept as repository content:

- `app/`
- `tools/prepare_smoke_training.py`
- `tools/smoke_train_step.py`
- `tools/setup_env.ps1`
- `tools/launch_app.ps1`
- `tools/test_train_smoke.ps1`
- `runtime/softvc_clean/cluster/`
- `runtime/softvc_clean/configs/`
- `runtime/softvc_clean/configs_template/`
- `runtime/softvc_clean/diffusion/`
- `runtime/softvc_clean/inference/`
- `runtime/softvc_clean/modules/`
- `runtime/softvc_clean/vdecoder/`
- `runtime/softvc_clean/vencoder/`
- `runtime/softvc_clean/*.py`
- `runtime/softvc_clean/requirements*.txt`
- `runtime/softvc_clean/LICENSE`
- `runtime_settings.yaml`
- `configs/runtime_settings.example.yaml`
- `.env.example`
- `.gitignore`
- `README.md`
- `docs/model_setup.md`
- `docs/data_setup.md`

## 4. Key Exclusions

The following were intentionally excluded from the GitHub export:

### Portable environments

- `envs/`

Reason:

- Too large for source control
- Machine-specific runtime bundle
- Not appropriate for a public repository

### Generated task outputs

- `jobs/`

Reason:

- Temporary or generated outputs
- Includes logs, result audio, training workspaces, and test artifacts

### Workspace data and local databases

- `workspace_data/`

Reason:

- Generated local state
- Contains datasets, model library content, and SQLite files

### Legacy trained exports

- `trained_models/`

Reason:

- Contains large model checkpoints and legacy artifacts

### Heavy runtime assets

Large asset contents excluded from `runtime/softvc_clean/`:

- `pretrain/`
- `pre_trained_model/`
- `logs/`
- `ffmpeg/`

Reason:

- Large binaries or pretrained model assets
- Runtime-only artifacts
- Should be fetched or provisioned locally
- Placeholder README files were retained so the expected paths remain visible in the repository

### Separator binaries and model caches

Large separator assets excluded from `runtime/`:

- `runtime/separators/models/`
- `runtime/separators/cache/`
- `runtime/separator_wheels/`

Reason:

- Large local helper assets
- Downloadable or reproducible outside the repo
- A placeholder README was retained under `runtime/separators/models/`

### Environment-specific bootstrap scripts

Not copied from the original `tools/` directory:

- `bootstrap_from_3fanchang.ps1`
- `build_portable_bundle.ps1`
- `check_env50.ps1`
- `patch_runtime50.ps1`
- `setup_env50.ps1`
- `setup_separator_helper.ps1`
- `test_train50.ps1`

Reason:

- Hard-coded local paths
- Depend on private or machine-specific portable environments
- Not suitable for a general GitHub audience without heavy rewriting

## 5. Reorganization Notes

Changes made in the export directory:

- Replaced the original portable-launch logic with generic `.venv` and system-Python detection in `tools/launch_app.ps1`
- Added `tools/setup_env.ps1` for general environment creation
- Added `.env` support in `app/config.py`
- Added environment-variable overrides for:
  - runtime root
  - settings file path
  - data root
  - jobs directory
  - trained model directory
  - separator executable
  - separator model/cache directories
  - ffmpeg executable
- Added example config and setup docs
- Replaced the root `README.md` with a GitHub-oriented public repository README

## 6. Dependency Notes

Dependency sources used:

- original root `requirements.txt`
- original `runtime/softvc_clean/requirements.txt`
- actual imports observed in the kept application/runtime source

Important note:

- `torch` and `torchaudio` remain unpinned so the target user can choose a CPU or CUDA build that matches their machine.

## 7. Risks And Manual Follow-Up

### Needs manual asset provisioning

Training and preprocessing will not work until the user provides:

- encoder assets
- base model checkpoints
- optional diffusion base models
- separator models
- ffmpeg

### Needs manual license confirmation

The root repository license was not inferred automatically.

- The runtime subcomponent license file already present in the trimmed runtime was preserved.
- The final public repository license choice still needs manual confirmation.

### Needs manual source/provenance confirmation

The repository keeps the trimmed `runtime/softvc_clean/` source tree because the app directly imports it.

- If this runtime came from another upstream project, confirm redistribution policy before publishing.

## 8. Pending Manual Confirmation

These items were intentionally **not** copied automatically and should be reviewed manually if needed:

- whether to publish the smoke-test helper scripts publicly
- whether to add a root `LICENSE`
- whether the `runtime/softvc_clean/` source tree should remain vendored or be replaced with a documented external dependency
- the final PyTorch/CUDA installation instructions for the intended GPU targets

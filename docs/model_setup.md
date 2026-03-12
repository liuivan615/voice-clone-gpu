# Model And Runtime Asset Setup

This repository keeps the application code and the trimmed runtime source tree, but it does **not** include pretrained weights, separator models, ffmpeg binaries, or trained checkpoints.

## 1. Runtime source layout already included

The following code and config directories are already part of the repository:

- `runtime/softvc_clean/cluster/`
- `runtime/softvc_clean/configs/`
- `runtime/softvc_clean/configs_template/`
- `runtime/softvc_clean/diffusion/`
- `runtime/softvc_clean/inference/`
- `runtime/softvc_clean/modules/`
- `runtime/softvc_clean/vdecoder/`
- `runtime/softvc_clean/vencoder/`

## 2. Assets you must provide manually

### Encoder and feature assets

Place these under `runtime/softvc_clean/pretrain/`:

- `checkpoint_best_legacy_500.pt`
  Used by the `balanced` preset (`vec768l12`) and related feature extraction.
- `hubert-soft-0d54a1f4.pt`
  Used by the `light` preset (`hubertsoft`).
- `medium.pt`
  Used by the `high_quality` preset (`whisper-ppg`).
- `rmvpe.pt`
  Recommended F0 extractor asset.
- `fcpe.pt`
  Optional alternative F0 extractor asset.

### Main base models

Place these under `runtime/softvc_clean/pre_trained_model/`:

- `768l12/G_0.pth`
- `768l12/D_0.pth`
- `hubertsoft/G_0.pth`
- `hubertsoft/D_0.pth`
- `whisper-ppg/G_0.pth`
- `whisper-ppg/D_0.pth`

Optional tiny preset assets:

- `tiny/vec768l12_vol_emb/G_0.pth`
- `tiny/vec768l12_vol_emb/D_0.pth`

### Diffusion base models

Place these under `runtime/softvc_clean/pre_trained_model/diffusion/`:

- `768l12/model_0.pt`
- `hubertsoft/model_0.pt`
- `whisper-ppg/model_0.pt`

### Vocal separation models

Place these under `runtime/separators/models/`:

- `htdemucs_ft.yaml`
- `f7e0c4bc-ba3fe64a.th`
- `d12395a8-e57c48e6.th`
- `92cfc3b6-ef3bcb9c.th`
- `04573f0d-f3cf25b2.th`
- `UVR_MDXNET_Main.onnx`

These filenames are required because the application checks them directly in `app/services/preprocess_service.py`.

### ffmpeg

Preprocessing needs `ffmpeg`. Use either of these approaches:

- Install `ffmpeg` and make it available in your system `PATH`.
- Set `WAV_FFMPEG_EXE` in `.env` to the full path of `ffmpeg`.

## 3. How the app locates assets

The new GitHub version supports environment-variable overrides:

- `WAV_RUNTIME_ROOT`
- `WAV_SEPARATOR_MODELS_DIR`
- `WAV_SEPARATOR_CACHE_DIR`
- `WAV_SEPARATOR_EXE`
- `WAV_FFMPEG_EXE`

If you do not override them, the app uses repository-relative defaults.

## 4. How trained models are stored

Trained models are **not** committed to the repository. At runtime the application creates and uses:

- `workspace_data/models/`
- `trained_models/`

The web UI and `LibraryService` register trained versions into `workspace_data/models/` and may also migrate legacy metadata from `trained_models/`.

## 5. Recommended verification

After placing assets:

1. Run `tools/setup_env.ps1`.
2. Start the app with `run.bat`.
3. Open `http://127.0.0.1:8000`.
4. Confirm that training presets and separator engines appear as available in the UI.

# Signal Foundry Workstation

A cleaned GitHub-ready export of the original portable `4wav_gpu_frontend_portable` workspace.

This project is a local web workstation for dataset management, media preprocessing, model training,
and inference around a SoftVC-style voice conversion runtime. It preserves the application code and
the trimmed runtime source tree, but intentionally excludes bundled Python environments, model weights,
datasets, caches, logs, and generated outputs.

---

## Quick Start

> Full details for each step are in the sections below.

**1. Install the environment**

```powershell
# Create .venv and install Python dependencies
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1

# Optional: also install the vocal-separation CLI
powershell -ExecutionPolicy Bypass -File .\tools\setup_env.ps1 -WithPreprocessTools
```

Then verify (or replace) the PyTorch build for your GPU — `requirements.txt` leaves
`torch` / `torchaudio` unpinned so you can install the right CUDA wheel:

```powershell
# Example: RTX 50 / Blackwell (CUDA 12.8)
.\.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**2. Prepare model assets**

Place the required files before launching (see [docs/model_setup.md](docs/model_setup.md)):

| Asset | Default location |
|---|---|
| Feature extractor (e.g. `checkpoint_best_legacy_500.pt`) | `runtime/softvc_clean/pretrain/` |
| F0 predictor (`rmvpe.pt` recommended) | `runtime/softvc_clean/pretrain/` |
| Base generator + discriminator (`G_0.pth`, `D_0.pth`) | `runtime/softvc_clean/pre_trained_model/{preset}/` |
| Separator models (Demucs / MDX-Net) | `runtime/separators/models/` |
| `ffmpeg` | System PATH, or set `WAV_FFMPEG_EXE` in `.env` |

**3. Start the app**

```powershell
.\run.bat
```

Open `http://127.0.0.1:8000` in your browser. The top-right chip shows GPU / CUDA status.

**4. Dataset tab — prepare training data**

1. Fill in **Speaker name** and **Dataset name**, click **+ 创建数据集**.
2. Drag-and-drop `.wav` files into the upload zone; tick **上传后立即自动分段** to segment automatically.
3. Review the candidate segments — disable low-quality clips by toggling their checkboxes.
4. *(Optional)* Use **提取人声** to run vocal separation on a recording, then choose the vocal or original variant.
5. Enter a version label (e.g. `v1_clean`) and click **+ 创建数据集版本** to freeze an immutable snapshot.

**5. Training tab — train the model**

1. Select training mode: **新建训练** (from scratch), **继续训练** (resume), or **仅扩散训练**.
2. Pick the frozen dataset version and enter a **model name**.
3. Choose a preset:

   | Preset | Encoder | VRAM |
   |---|---|---|
   | ⚡ 均衡 (Balanced) | ContentVec 768L12 | ~6 GB |
   | 🎯 高质量 (High-Quality) | Whisper-PPG | ~12 GB |
   | 💡 轻量 (Light) | HuBERT-Soft | ~4 GB |

4. Select an F0 predictor (**RMVPE** recommended) and adjust steps / batch size if needed.
5. Click **开始训练**. A live log and progress bar stream training updates over WebSocket.
6. When finished, the model is registered in the local library automatically.

**6. Inference tab — convert audio**

1. Select a model version from the library and click **加载模型**.
2. Upload an audio or video file.
   - *(Optional)* Enable **自动分离人声**, pick a separator engine, and click **开始预处理**; then choose the original or vocal variant.
3. Set pitch shift (**Tran**), noise scale, and other parameters as needed.
4. Click **开始推理**. When done, a player and **下载结果** button appear.

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

---

## 简体中文说明

### 项目简介

这是一个基于 SoftVC 风格语音转换的本地 Web 工作站，用于数据集管理、媒体预处理、模型训练和推理。该仓库是原始便携式工作区 `4wav_gpu_frontend_portable` 的整理版，保留了应用代码和装切后的运行时源码，但不包含 Python 虚拟环境、预训权重、数据集、日志或生成文件。

### 快速开始

1. **安装环境**  
   运行 `tools/setup_env.ps1` 脚本创建 `.venv` 虚拟环境并安装 Python 依赖。根据你的 GPU/CPU 情况，单独安装适合的 `torch`/`torchaudio` 版本。

2. **准备模型资源**  
   在启动前将预训特征提取器、F0 预测器、基础生成器/ 判别器检查点以及分离模型放置到指定目录（详见 `docs/model_setup.md`）。系统不会自带这些权重。

3. **启动应用**  
   执行 `run.bat` 或 `tools/launch_app.ps1`，然后打开浏览器访问 `http://127.0.0.1:8000`。

4. **数据集**  
   在 **Dataset** 页面中创建数据集名称及版本，上传 `.wav` 文件，可勾选“上传后立即自动分段”进行自动切片；检查和筛选片段后，填写版本号并冻结数据集。

5. **训练**  
   在 **Training** 页面选择已冻结的数据集版本和模型名称，选择训练预设（均衡、高质量、轻量）以及 F0 预测器，点击“开始训练”开始新训练/继续训练/仅扬散训练。训练完成后模型会记录到本地模型库。

6. **推理**  
   在 **Inference** 页面加载模型，上传待转换的音频或视频，必要时先进行人声分离，然后设置音高调数、噪声层级等参数并点击“开始推理”，生成的结果可以下载。

### 本仓库包含内容

- FastAPI 后端应用和浏览器前端 UI (`app/`)
- 被装切的 SoftVC 运行时源码 (`runtime/softvc_clean/`)
- 安装、启动和测试脚本 (`tools/`)
- 示例配置文件 (`configs/`, `runtime_settings.yaml` 等)

### 本仓库不包含内容

为了保证仓库轻量和安全，以下内容未包含在内：

- Python 环境 (`envs/`)
- 训练输出、日志、任务记录 (`jobs/`)
- 本地工作区数据库及模型库 (`workspace_data/`)
- 训练模型权重和预训资源
- 分离模型文件
- ffmpeg 可执行文件
- 数据集和生成的媒体文件

请参考 `docs/model_setup.md` 和 `docs/data_setup.md` 准备缺失的资源。

### 环境需求

- **操作系统：** Windows（原便携式环境的目标）
- **Python：** 3.10 以上
- **GPU：** 建议使用 NVIDIA GPU
- **CUDA：** 例如 RTX 50/Blackwell 应使用支持 CUDA 12.8 的 PyTorch 版本
- **ffmpeg：** 用于音频预处理，可设置在 PATH 或通过 `.env` 中的 `WAV_FFMPEG_EXE` 指定

### 工作流程概述

训练流程：准备并冻结数据集 → 在训练页面选择数据集和预设 → 开始新训练/继续训练/扬散训练。  
推理流程：加载模型版本 → 上传音频/视频文件（可选预处理人声）→ 设置参数 → 开始推理 → 下载结果。

更多详情参见 `docs/` 文件夹中的文档。

from __future__ import annotations

import copy
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
STATIC_DIR = APP_DIR / "static"
SERVICES_DIR = APP_DIR / "services"
TOOLS_DIR = ROOT_DIR / "tools"
CONFIGS_DIR = ROOT_DIR / "configs"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name, "").strip()
    candidate = Path(raw_value) if raw_value else default
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


def _resolve_executable(env_name: str, *, candidates: Iterable[Path], command_names: Iterable[str]) -> Path:
    raw_value = os.environ.get(env_name, "").strip()
    if raw_value:
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        return candidate.resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    for command_name in command_names:
        resolved = shutil.which(command_name)
        if resolved:
            return Path(resolved).resolve()
    first_candidate = next(iter(candidates), None)
    if first_candidate is not None:
        return first_candidate.resolve()
    first_name = next(iter(command_names), "")
    return Path(first_name)


_load_dotenv(ROOT_DIR / ".env")

RUNTIME_DIR = _env_path("WAV_RUNTIME_ROOT", ROOT_DIR / "runtime" / "softvc_clean")
JOBS_DIR = _env_path("WAV_JOBS_DIR", ROOT_DIR / "jobs")
TRAINED_MODELS_DIR = _env_path("WAV_TRAINED_MODELS_DIR", ROOT_DIR / "trained_models")
RUNTIME_SETTINGS_PATH = _env_path("WAV_SETTINGS_PATH", ROOT_DIR / "runtime_settings.yaml")
SEPARATORS_DIR = _env_path("WAV_SEPARATORS_DIR", ROOT_DIR / "runtime" / "separators")
SEPARATOR_MODELS_DIR = _env_path("WAV_SEPARATOR_MODELS_DIR", SEPARATORS_DIR / "models")
SEPARATOR_CACHE_DIR = _env_path("WAV_SEPARATOR_CACHE_DIR", SEPARATORS_DIR / "cache")
DEFAULT_VENV_DIR = _env_path("WAV_VENV_DIR", ROOT_DIR / ".venv")
SEPARATOR_EXE = _resolve_executable(
    "WAV_SEPARATOR_EXE",
    candidates=(
        DEFAULT_VENV_DIR / "Scripts" / "audio-separator.exe",
        DEFAULT_VENV_DIR / "Scripts" / "audio-separator",
    ),
    command_names=("audio-separator.exe", "audio-separator"),
)
FFMPEG_EXE = _resolve_executable(
    "WAV_FFMPEG_EXE",
    candidates=(RUNTIME_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",),
    command_names=("ffmpeg.exe", "ffmpeg"),
)

DEFAULT_SETTINGS = {
    "app": {
        "host": "127.0.0.1",
        "port": 8000,
        "upload_limit_mb": 200,
        "media_upload_limit_mb": 0,
        "training_upload_limit_mb": 0,
        "data_root": "workspace_data",
        "allow_cpu_fallback": True,
        "minimum_driver_version": "570.65",
        "task_retention_days": 7,
        "preprocess_retention_days": 7,
    },
    "training_defaults": {
        "f0_predictor": "rmvpe",
        "checkpoint_interval_steps": 500,
        "checkpoint_keep_last": 5,
        "step_count": 2000,
    },
    "default_profile": {
        "checkpoint_path": "",
        "config_path": "configs/config.json",
        "cluster_path": "",
        "diffusion_model_path": "",
        "diffusion_config_path": "configs/diffusion.yaml",
        "use_diffusion": False,
        "speaker": "",
        "device_preference": "auto",
        "tran": 0,
        "slice_db": -40,
        "slice_min_length_ms": 5000,
        "noise_scale": 0.4,
        "pad_seconds": 0.5,
        "f0_predictor": "pm",
        "output_format": "wav",
        "auto_predict_f0": False,
        "cluster_infer_ratio": 0.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_project_dirs() -> None:
    for path in (
        ROOT_DIR,
        APP_DIR,
        STATIC_DIR,
        SERVICES_DIR,
        TOOLS_DIR,
        CONFIGS_DIR,
        JOBS_DIR,
        TRAINED_MODELS_DIR,
        SEPARATORS_DIR,
        SEPARATOR_MODELS_DIR,
        SEPARATOR_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    ensure_project_dirs()
    if not RUNTIME_SETTINGS_PATH.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    with RUNTIME_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULT_SETTINGS, data)


SETTINGS = load_settings()


def write_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = _deep_merge(DEFAULT_SETTINGS, settings)
    with RUNTIME_SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, allow_unicode=True, sort_keys=False)
    return reload_settings()


def reload_settings() -> Dict[str, Any]:
    SETTINGS.clear()
    SETTINGS.update(load_settings())
    return SETTINGS


def resolve_data_root(settings: Optional[Dict[str, Any]] = None) -> Path:
    active_settings = settings or SETTINGS
    raw_value = os.environ.get("WAV_DATA_ROOT", "").strip() or str(active_settings["app"].get("data_root", "workspace_data")).strip() or "workspace_data"
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


def workspace_paths(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    data_root = resolve_data_root(settings)
    return {
        "data_root": data_root,
        "db_path": data_root / "library" / "workspace.sqlite3",
        "datasets_root": data_root / "datasets",
        "dataset_files_root": data_root / "datasets" / "files",
        "dataset_segments_root": data_root / "datasets" / "segments",
        "models_root": data_root / "models",
        "jobs_root": data_root / "runs",
    }


def resolve_runtime_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    data_root_candidate = (resolve_data_root() / candidate).resolve()
    if data_root_candidate.exists():
        return data_root_candidate
    runtime_candidate = (RUNTIME_DIR / candidate).resolve()
    if runtime_candidate.exists():
        return runtime_candidate
    root_candidate = (ROOT_DIR / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    return runtime_candidate


def ensure_runtime_import_path() -> None:
    runtime_str = str(RUNTIME_DIR)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)


def ensure_runtime_environment() -> None:
    ensure_runtime_import_path()
    os.environ.setdefault("FAIRSEQ_SKIP_HYDRA_INIT", "1")
    if FFMPEG_EXE.exists():
        ffmpeg_bin = FFMPEG_EXE.parent
        current_path = os.environ.get("PATH", "")
        ffmpeg_str = str(ffmpeg_bin)
        if ffmpeg_str not in current_path:
            os.environ["PATH"] = ffmpeg_str + os.pathsep + current_path


@contextmanager
def runtime_workdir():
    ensure_runtime_environment()
    previous = Path.cwd()
    try:
        if RUNTIME_DIR.exists():
            os.chdir(RUNTIME_DIR)
        yield
    finally:
        os.chdir(previous)

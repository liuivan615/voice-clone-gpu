from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile

from app.config import (
    FFMPEG_EXE,
    JOBS_DIR,
    RUNTIME_DIR,
    SEPARATOR_EXE,
    SEPARATOR_MODELS_DIR,
    SETTINGS,
    ensure_runtime_environment,
)


ProgressCallback = Callable[[str, float, str, int, int], None]
LogCallback = Callable[[str], None]

ACCEPTED_MEDIA_EXTENSIONS = [".wav", ".mp3", ".mp4", ".mkv", ".mov", ".avi"]
AUDIO_EXTENSIONS = {".wav", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}
PREPROCESS_JOBS_DIR = JOBS_DIR / "preprocess"
SEPARATOR_CACHE_DIR = RUNTIME_DIR.parent / "separators" / "cache"


@dataclass(frozen=True)
class SeparatorEngine:
    key: str
    label: str
    model_filename: str
    required_files: tuple[str, ...]


SEPARATOR_ENGINES = {
    "demucs": SeparatorEngine(
        key="demucs",
        label="Demucs",
        model_filename="htdemucs_ft.yaml",
        required_files=(
            "htdemucs_ft.yaml",
            "f7e0c4bc-ba3fe64a.th",
            "d12395a8-e57c48e6.th",
            "92cfc3b6-ef3bcb9c.th",
            "04573f0d-f3cf25b2.th",
        ),
    ),
    "mdx": SeparatorEngine(
        key="mdx",
        label="MDX-Net",
        model_filename="UVR_MDXNET_Main.onnx",
        required_files=("UVR_MDXNET_Main.onnx",),
    ),
}


class PreprocessError(RuntimeError):
    pass


class PreprocessService:
    def __init__(self) -> None:
        ensure_runtime_environment()
        PREPROCESS_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        SEPARATOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SEPARATOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def accepted_media_types() -> List[str]:
        return list(ACCEPTED_MEDIA_EXTENSIONS)

    @staticmethod
    def preprocess_retention_days() -> int:
        return int(SETTINGS["app"].get("preprocess_retention_days") or SETTINGS["app"].get("task_retention_days") or 7)

    @staticmethod
    def media_upload_limit_mb() -> int:
        return int(SETTINGS["app"].get("media_upload_limit_mb") or 0)

    def separator_engines(self) -> List[Dict[str, Any]]:
        return [self._separator_engine_payload(engine) for engine in SEPARATOR_ENGINES.values()]

    def ensure_engine_available(self, key: str) -> SeparatorEngine:
        engine = self._engine(key)
        if not self._engine_available(engine):
            raise PreprocessError(f"{engine.label} 当前不可用，请检查预处理 helper 或模型文件。")
        return engine

    def task_dir(self, task_id: str) -> Path:
        return PREPROCESS_JOBS_DIR / task_id

    def cleanup_expired_jobs(self) -> None:
        retention_days = self.preprocess_retention_days()
        if retention_days <= 0 or not PREPROCESS_JOBS_DIR.exists():
            return
        cutoff = time.time() - retention_days * 24 * 60 * 60
        for candidate in PREPROCESS_JOBS_DIR.iterdir():
            if not candidate.is_dir():
                continue
            try:
                if candidate.stat().st_mtime < cutoff:
                    shutil.rmtree(candidate, ignore_errors=True)
            except OSError:
                continue

    def source_kind_for_name(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return "audio"

    def validate_media_extension(self, filename: str) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in ACCEPTED_MEDIA_EXTENSIONS:
            joined = ", ".join(ACCEPTED_MEDIA_EXTENSIONS)
            raise PreprocessError(f"当前仅支持以下输入格式：{joined}")

    def resolve_variant_file(self, task_id: str, variant: str) -> Path:
        summary = self.load_summary(task_id)
        if not summary:
            raise FileNotFoundError("预处理结果已过期，请重新上传。")
        normalized_variant = (variant or "").strip().lower()
        if normalized_variant not in {"original", "vocals"}:
            raise ValueError("prepared_variant 只支持 original 或 vocals。")
        available = set(summary.get("available_variants") or [])
        if normalized_variant not in available:
            raise FileNotFoundError("预处理结果已过期，请重新上传。")
        relative_path = summary.get("original_file") if normalized_variant == "original" else summary.get("vocals_file")
        if not relative_path:
            raise FileNotFoundError("预处理结果已过期，请重新上传。")
        target = self.task_dir(task_id) / relative_path
        if not target.exists():
            raise FileNotFoundError("预处理结果已过期，请重新上传。")
        return target

    def load_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        summary_path = self.task_dir(task_id) / "summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def load_snapshot(self, task_id: str) -> Optional[Dict[str, Any]]:
        summary = self.load_summary(task_id)
        if not summary:
            return None
        task_dir = self.task_dir(task_id)
        run_log_path = task_dir / "run.log"
        logs = []
        if run_log_path.exists():
            try:
                logs = [line for line in run_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                logs = []
        return {
            "task_id": task_id,
            "status": summary.get("status", "completed"),
            "stage": summary.get("status", "completed"),
            "progress": 100.0 if summary.get("status") == "completed" else 0.0,
            "message": summary.get("warning") or ("处理完成。" if summary.get("status") == "completed" else summary.get("error") or "预处理失败。"),
            "logs": logs,
            "current_segment": 0,
            "total_segments": 0,
            "result_url": None,
            "error": summary.get("error"),
            "summary": summary,
            "sequence": len(logs) or 1,
            "original_url": self.variant_media_url(task_id, "original") if "original" in (summary.get("available_variants") or []) else None,
            "vocals_url": self.variant_media_url(task_id, "vocals") if "vocals" in (summary.get("available_variants") or []) else None,
            "available_variants": summary.get("available_variants") or [],
            "default_variant": summary.get("default_variant"),
            "warning": summary.get("warning"),
            "engine": summary.get("separator_engine"),
            "meta": {
                "task_kind": "preprocess",
                "task_dir": str(task_dir),
                "separator_engine": summary.get("separator_engine"),
                "source_kind": summary.get("source_kind"),
            },
        }

    def variant_media_url(self, task_id: str, variant: str) -> str:
        return f"/api/preprocess/tasks/{task_id}/media/{variant}"

    def ensure_inference_wav(
        self,
        *,
        task_id: str,
        source_file: Path,
        task_dir: Path,
        progress: ProgressCallback,
        emit_log: Optional[LogCallback] = None,
    ) -> tuple[Path, Dict[str, Any]]:
        if source_file.suffix.lower() == ".wav":
            return source_file, {"source_kind": "audio", "normalized": False}
        prepared_dir = task_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        target = prepared_dir / "original.wav"
        progress("preparing_audio", 0.04, "正在转换输入媒体为推理 WAV...", 0, 0)
        self._normalize_media(
            source_path=source_file,
            output_path=target,
            emit_log=emit_log,
        )
        return target, {"source_kind": self.source_kind_for_name(source_file.name), "normalized": True}

    def run(
        self,
        *,
        task_id: str,
        source_path: Path,
        source_name: str,
        task_dir: Path,
        separator_engine: str,
        progress: ProgressCallback,
        emit_log: Optional[LogCallback] = None,
    ) -> Dict[str, Any]:
        self.validate_media_extension(source_name)
        engine = self.ensure_engine_available(separator_engine)
        normalized_dir = task_dir / "normalized"
        raw_separator_dir = task_dir / "separator_raw"
        vocals_dir = task_dir / "vocals"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        raw_separator_dir.mkdir(parents=True, exist_ok=True)
        vocals_dir.mkdir(parents=True, exist_ok=True)

        original_wav = normalized_dir / "original.wav"
        progress("normalizing_audio", 0.08, "正在提取并转换输入音频...", 0, 0)
        self._normalize_media(source_path=source_path, output_path=original_wav, emit_log=emit_log)

        progress("separating_vocals", 0.35, f"正在使用 {engine.label} 提取人声...", 0, 0)
        separated_vocals = self._run_separator(
            input_wav=original_wav,
            output_dir=raw_separator_dir,
            engine=engine,
            emit_log=emit_log,
        )

        vocals_wav = vocals_dir / "vocals.wav"
        progress("finalizing_audio", 0.82, "正在整理提取后的人声 WAV...", 0, 0)
        self._normalize_media(source_path=separated_vocals, output_path=vocals_wav, emit_log=emit_log)

        warning = None
        available_variants = ["original", "vocals"]
        default_variant = "vocals"
        try:
            if self._looks_like_empty_vocals(vocals_wav):
                available_variants = ["original"]
                default_variant = "original"
                warning = "未检测到清晰人声，建议直接使用原始音频。"
                raise PreprocessError(warning)
        except PreprocessError:
            raise
        except Exception:
            warning = None

        summary = {
            "task_id": task_id,
            "status": "completed",
            "source_file": source_name,
            "source_kind": self.source_kind_for_name(source_name),
            "separator_engine": engine.key,
            "separator_label": engine.label,
            "available_variants": available_variants,
            "default_variant": default_variant,
            "warning": warning,
            "original_file": str(original_wav.relative_to(task_dir)).replace("\\", "/"),
            "vocals_file": str(vocals_wav.relative_to(task_dir)).replace("\\", "/"),
            "duration_seconds": round(self._duration_seconds(original_wav), 4),
        }
        self._write_summary(task_dir / "summary.json", summary)
        return summary

    def failure_summary(
        self,
        *,
        task_id: str,
        source_name: str,
        task_dir: Path,
        separator_engine: str,
        message: str,
    ) -> Dict[str, Any]:
        original_wav = task_dir / "normalized" / "original.wav"
        has_original = original_wav.exists()
        fallback_warning = None
        if has_original:
            fallback_warning = "未检测到清晰人声，建议直接使用原始音频。" if "未检测到清晰人声" in message else "人声分离失败，建议直接使用原始音频。"
        summary = {
            "task_id": task_id,
            "status": "failed",
            "source_file": source_name,
            "source_kind": self.source_kind_for_name(source_name),
            "separator_engine": separator_engine,
            "separator_label": SEPARATOR_ENGINES[separator_engine].label if separator_engine in SEPARATOR_ENGINES else separator_engine,
            "available_variants": ["original"] if has_original else [],
            "default_variant": "original" if has_original else None,
            "warning": fallback_warning,
            "error": message,
            "original_file": str(original_wav.relative_to(task_dir)).replace("\\", "/") if has_original else None,
            "vocals_file": None,
        }
        self._write_summary(task_dir / "summary.json", summary)
        return summary

    def _separator_engine_payload(self, engine: SeparatorEngine) -> Dict[str, Any]:
        available = self._engine_available(engine)
        return {
            "value": engine.key,
            "label": engine.label,
            "available": available,
            "reason_disabled": None if available else "缺少预处理 helper 或本地模型文件。",
            "model_filename": engine.model_filename,
        }

    def _engine(self, key: str) -> SeparatorEngine:
        engine = SEPARATOR_ENGINES.get((key or "").strip().lower())
        if not engine:
            raise PreprocessError("separator_engine 只支持 demucs 或 mdx。")
        return engine

    def _engine_available(self, engine: SeparatorEngine) -> bool:
        if not SEPARATOR_EXE.exists():
            return False
        return all((SEPARATOR_MODELS_DIR / name).exists() for name in engine.required_files)

    def _normalize_media(
        self,
        *,
        source_path: Path,
        output_path: Path,
        emit_log: Optional[LogCallback] = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = FFMPEG_EXE
        if not ffmpeg_path.exists():
            raise PreprocessError("未找到 ffmpeg，可通过 WAV_FFMPEG_EXE 指定路径，或将 ffmpeg 加入系统 PATH。")
        cmd = [
            str(ffmpeg_path),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
        self._run_command(cmd, emit_log=emit_log)
        if not output_path.exists():
            raise PreprocessError("媒体转换失败，未生成标准 WAV。")

    def _run_separator(
        self,
        *,
        input_wav: Path,
        output_dir: Path,
        engine: SeparatorEngine,
        emit_log: Optional[LogCallback] = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["AUDIO_SEPARATOR_MODEL_DIR"] = str(SEPARATOR_MODELS_DIR)
        env["TORCH_HOME"] = str(SEPARATOR_CACHE_DIR / "torch")
        env["TMP"] = str(SEPARATOR_CACHE_DIR / "tmp")
        env["TEMP"] = str(SEPARATOR_CACHE_DIR / "tmp")
        env["HF_HOME"] = str(SEPARATOR_CACHE_DIR / "hf")
        Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["TMP"]).mkdir(parents=True, exist_ok=True)
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        cmd = [
            str(SEPARATOR_EXE),
            "--model_file_dir",
            str(SEPARATOR_MODELS_DIR),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "WAV",
            "--single_stem",
            "Vocals",
            "-m",
            engine.model_filename,
            str(input_wav),
        ]
        self._run_command(cmd, env=env, emit_log=emit_log)
        candidates = sorted(output_dir.glob("*Vocals*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not candidates:
            raise PreprocessError("人声分离失败，未生成可用的人声轨。")
        return candidates[0]

    def _run_command(
        self,
        command: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
        emit_log: Optional[LogCallback] = None,
    ) -> None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        last_message = ""
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            last_message = line
            if emit_log:
                emit_log(line)
        return_code = process.wait()
        if return_code != 0:
            raise PreprocessError(last_message or "外部音频预处理命令执行失败。")

    def _looks_like_empty_vocals(self, vocals_wav: Path) -> bool:
        try:
            data, _ = soundfile.read(vocals_wav)
        except Exception:
            return False
        arr = np.asarray(data, dtype=np.float32)
        if arr.size == 0:
            return True
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        peak = float(np.max(np.abs(arr)))
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
        return peak < 1e-4 or rms < 5e-5

    def _duration_seconds(self, wav_path: Path) -> float:
        try:
            info = soundfile.info(wav_path)
            return float(info.frames / info.samplerate) if info.samplerate else 0.0
        except Exception:
            return 0.0

    def _write_summary(self, summary_path: Path, summary: Dict[str, Any]) -> None:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

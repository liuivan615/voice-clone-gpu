from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import soundfile
import torch

from app.config import runtime_workdir
from app.schemas import RuntimeProfile
from app.services.model_service import ModelService
from app.services.preprocess_service import PreprocessService


ProgressCallback = Callable[[str, float, str, int, int], None]


def _pad_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    pad_width = target_length - current_length
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return np.pad(arr, (pad_left, pad_right), "constant", constant_values=(0, 0))


class PipelineService:
    def __init__(self, model_service: ModelService, preprocess_service: PreprocessService) -> None:
        self.model_service = model_service
        self.preprocess_service = preprocess_service

    def _write_summary(self, summary_path: Path, summary: Dict[str, Any]) -> None:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def run(
        self,
        *,
        task_id: str,
        source_file: Path,
        source_name: str,
        task_dir: Path,
        profile: RuntimeProfile,
        progress: ProgressCallback,
    ) -> Dict[str, Any]:
        segments_dir = task_dir / "segments"
        result_dir = task_dir / "result"
        logs_dir = task_dir / "logs"
        segments_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        prepared_source, source_meta = self.preprocess_service.ensure_inference_wav(
            task_id=task_id,
            source_file=source_file,
            task_dir=task_dir,
            progress=progress,
        )
        prep_offset = 0.08 if source_meta.get("normalized") else 0.0

        progress("loading_model", 0.02 + prep_offset, "正在加载模型...", 0, 0)
        state = self.model_service.get_or_load(profile)
        model = state.model
        active_profile = state.profile
        compat = self.model_service.snapshot()["gpu_compatibility"]
        matched_profile = self.model_service.library_service.find_profile(active_profile.checkpoint_path)

        with runtime_workdir():
            from inference import slicer

            progress("slicing", 0.08 + prep_offset, "正在切分音频...", 0, 0)
            chunks = slicer.cut(str(prepared_source), db_thresh=active_profile.slice_db, min_len=active_profile.slice_min_length_ms)
            audio_data, audio_sr = slicer.chunks2audio(str(prepared_source), chunks)

        if not audio_data:
            raise RuntimeError("切分后没有得到可处理片段，请检查 WAV 内容。")

        slice_manifest = []  # type: List[Dict[str, Any]]
        durations = []  # type: List[float]
        total_segments = len(audio_data)
        progress("slicing", 0.15 + prep_offset, f"切分完成，共 {total_segments} 段。", 0, total_segments)

        for index, (slice_tag, data) in enumerate(audio_data, start=1):
            duration = float(len(data) / audio_sr)
            durations.append(duration)
            chunk_name = f"{index:03d}_{'silence' if slice_tag else 'voice'}.wav"
            chunk_path = segments_dir / chunk_name
            soundfile.write(chunk_path, data, audio_sr, format="wav")
            slice_manifest.append(
                {
                    "index": index,
                    "slice_tag": bool(slice_tag),
                    "duration_seconds": round(duration, 4),
                    "file": chunk_name,
                }
            )

        with (segments_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(slice_manifest, handle, ensure_ascii=False, indent=2)

        audio_output: list[float] = []
        for index, (slice_tag, data) in enumerate(audio_data, start=1):
            progress_ratio = 0.15 + prep_offset + (index / total_segments) * (0.75 - prep_offset)
            if slice_tag:
                length = int(np.ceil(len(data) / audio_sr * model.target_sample))
                audio_output.extend(list(np.zeros(length, dtype=np.float32)))
                progress("inference", progress_ratio, f"跳过静音段 {index}/{total_segments}", index, total_segments)
                continue

            progress("inference", progress_ratio, f"正在处理第 {index}/{total_segments} 段...", index, total_segments)
            target_length = int(np.ceil(len(data) / audio_sr * model.target_sample))
            pad_len = int(audio_sr * active_profile.pad_seconds)
            padded = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])]).astype(np.float32)
            raw_path = io.BytesIO()
            soundfile.write(raw_path, padded, audio_sr, format="wav")
            raw_path.seek(0)

            with runtime_workdir():
                out_audio, _, _ = model.infer(
                    active_profile.speaker,
                    active_profile.tran,
                    raw_path,
                    cluster_infer_ratio=active_profile.cluster_infer_ratio,
                    auto_predict_f0=active_profile.auto_predict_f0,
                    noice_scale=active_profile.noise_scale,
                    f0_predictor=active_profile.f0_predictor,
                )
                model.clear_empty()

            rendered = out_audio.cpu().numpy()
            pad_target = int(model.target_sample * active_profile.pad_seconds)
            if pad_target > 0 and rendered.shape[0] > 2 * pad_target:
                rendered = rendered[pad_target:-pad_target]
            rendered = _pad_array(rendered, target_length)
            audio_output.extend(list(rendered))

        progress("packaging", 0.94, "正在写入输出文件...", total_segments, total_segments)
        output_file = result_dir / f"{source_file.stem}_converted.{active_profile.output_format}"
        soundfile.write(output_file, np.asarray(audio_output, dtype=np.float32), model.target_sample, format=active_profile.output_format)

        summary = {
            "task_id": task_id,
            "source_file": source_name,
            "slice_count": total_segments,
            "min_duration": round(min(durations), 4),
            "max_duration": round(max(durations), 4),
            "total_duration": round(sum(durations), 4),
            "device_used": str(getattr(model, "dev", "cpu")),
            "speaker": active_profile.speaker,
            "result_file": output_file.name,
            "status": "completed",
            "error": None,
            "model_version_id": matched_profile["model_version_id"] if matched_profile else None,
            "dataset_version_id": matched_profile["dataset_version_id"] if matched_profile else None,
            "gpu_name": compat.get("gpu_name"),
            "compute_capability": compat.get("compute_capability"),
            "torch_cuda_version": compat.get("torch_cuda_version"),
            "source_kind": source_meta.get("source_kind", "audio"),
            "normalized_before_inference": bool(source_meta.get("normalized")),
        }
        self._write_summary(task_dir / "summary.json", summary)
        torch.cuda.empty_cache()
        progress("completed", 1.0, "处理完成。", total_segments, total_segments)
        return summary

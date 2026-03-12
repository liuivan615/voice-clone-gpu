from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.config import JOBS_DIR
from app.schemas import RuntimeProfile
from app.services.pipeline_service import PipelineService


class TaskManager:
    def __init__(self, pipeline_service: PipelineService, gpu_lock: Optional[threading.Lock] = None) -> None:
        self.pipeline_service = pipeline_service
        self._tasks = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.RLock()
        self._gpu_lock = gpu_lock or threading.Lock()

    def _touch(self, task_id: str) -> None:
        self._tasks[task_id]["sequence"] += 1
        self._tasks[task_id]["updated_at"] = time.time()

    def _append_log(self, task_id: str, message: str) -> None:
        self._tasks[task_id]["logs"].append(message)
        self._touch(task_id)
        self._flush_logs(task_id)

    def _flush_logs(self, task_id: str) -> None:
        task_dir = Path(self._tasks[task_id]["meta"]["task_dir"])
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "run.log").write_text("\n".join(self._tasks[task_id]["logs"]), encoding="utf-8")

    def create_task(self, *, source_path: Path, source_name: str, profile: RuntimeProfile) -> dict[str, Any]:
        task_id = uuid.uuid4().hex
        task_dir = JOBS_DIR / task_id
        source_dir = task_dir / "source"
        original_name = source_name
        matched_profile = self.pipeline_service.model_service.library_service.find_profile(profile.checkpoint_path)
        task_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        source_suffix = source_path.suffix.lower() or Path(original_name).suffix.lower() or ".wav"
        target_source = source_dir / f"source{source_suffix}"
        shutil.copy2(source_path, target_source)
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            pass

        snapshot = {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "任务已创建。",
            "logs": ["任务已创建。"],
            "current_segment": 0,
            "total_segments": 0,
            "result_url": None,
            "error": None,
            "summary": None,
            "device_used": None,
            "sequence": 1,
            "meta": {
                "source_file": original_name,
                "task_dir": str(task_dir),
                "task_kind": "inference",
            },
            "updated_at": time.time(),
        }

        with self._lock:
            self._tasks[task_id] = snapshot
        self.pipeline_service.model_service.library_service.create_inference_run(
            task_id=task_id,
            model_version_id=matched_profile["id"] if matched_profile else None,
            message="推理任务已创建。",
        )

        worker = threading.Thread(
            target=self._run_task,
            kwargs={"task_id": task_id, "source_path": target_source, "source_name": original_name, "task_dir": task_dir, "profile": profile},
            daemon=True,
        )
        worker.start()
        return snapshot

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            for task in self._tasks.values():
                if task["status"] not in {"completed", "failed"}:
                    return {
                        "task_id": task["task_id"],
                        "status": task["status"],
                        "task_kind": task["meta"].get("task_kind", "inference"),
                    }
        return None

    def _run_task(self, *, task_id: str, source_path: Path, source_name: str, task_dir: Path, profile: RuntimeProfile) -> None:
        def progress(status: str, ratio: float, message: str, current: int, total: int) -> None:
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = status
                task["stage"] = status
                task["progress"] = round(ratio * 100, 2)
                task["message"] = message
                task["current_segment"] = current
                task["total_segments"] = total
                self._append_log(task_id, message)
            self.pipeline_service.model_service.library_service.update_inference_run(
                task_id=task_id,
                status=status,
                stage=status,
                message=message,
                progress=round(ratio * 100, 2),
            )

        try:
            progress("queued", 0.01, "等待 GPU 锁...", 0, 0)
            with self._gpu_lock:
                summary = self.pipeline_service.run(
                    task_id=task_id,
                    source_file=source_path,
                    source_name=source_name,
                    task_dir=task_dir,
                    profile=profile,
                    progress=progress,
                )
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "completed"
                task["progress"] = 100.0
                task["message"] = "处理完成。"
                task["summary"] = summary
                task["result_url"] = f"/api/tasks/{task_id}/result"
                task["device_used"] = summary.get("device_used")
                self._append_log(task_id, "结果文件已生成。")
            self.pipeline_service.model_service.library_service.update_inference_run(
                task_id=task_id,
                status="completed",
                stage="completed",
                message="处理完成。",
                progress=100.0,
                device_used=summary.get("device_used"),
                summary=summary,
            )
        except torch.cuda.OutOfMemoryError as exc:
            self._fail_task(task_id, "CUDA 显存不足，请检查模型规模、输入长度或环境配置。", exc)
        except Exception as exc:
            self._fail_task(task_id, str(exc), exc)

    def _fail_task(self, task_id: str, message: str, exc: Exception) -> None:
        stack = traceback.format_exc()
        task_dir = JOBS_DIR / task_id
        summary = {
            "task_id": task_id,
            "source_file": self._tasks[task_id]["meta"]["source_file"],
            "slice_count": self._tasks[task_id]["total_segments"],
            "min_duration": 0.0,
            "max_duration": 0.0,
            "total_duration": 0.0,
            "device_used": None,
            "speaker": None,
            "result_file": None,
            "status": "failed",
            "error": message,
            "traceback_file": "traceback.log",
            "gpu_name": None,
            "compute_capability": None,
            "torch_cuda_version": torch.version.cuda,
        }
        (task_dir / "traceback.log").write_text(stack, encoding="utf-8")
        (task_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()

        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "failed"
            task["stage"] = "failed"
            task["message"] = message
            task["error"] = message
            task["summary"] = summary
            self._append_log(task_id, f"{message} 详细堆栈见 traceback.log。")
        self.pipeline_service.model_service.library_service.update_inference_run(
            task_id=task_id,
            status="failed",
            stage="failed",
            message=message,
            progress=0.0,
            summary=summary,
        )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return json.loads(json.dumps(task, ensure_ascii=False))

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from app.config import JOBS_DIR
from app.services.training_service import TrainingService


class TrainingTaskManager:
    def __init__(self, training_service: TrainingService, gpu_lock: Optional[threading.Lock] = None) -> None:
        self.training_service = training_service
        self._tasks = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.RLock()
        self._gpu_lock = gpu_lock or threading.Lock()

    def _touch(self, task_id: str) -> None:
        self._tasks[task_id]["sequence"] += 1
        self._tasks[task_id]["updated_at"] = time.time()

    def _append_log(self, task_id: str, message: str) -> None:
        self._tasks[task_id]["logs"].append(message)
        self._touch(task_id)
        task_dir = Path(self._tasks[task_id]["meta"]["task_dir"])
        (task_dir / "run.log").write_text("\n".join(self._tasks[task_id]["logs"]), encoding="utf-8")

    def create_task(
        self,
        *,
        dataset_version_id: str,
        model_name: str,
        device_preference: str,
        f0_predictor: str,
        max_steps: int,
        training_mode: str,
        resume_from_checkpoint_id: Optional[str],
        checkpoint_interval_steps: int,
        checkpoint_keep_last: int,
        main_preset_id: str,
        main_batch_size: int,
        main_precision: str,
        main_all_in_mem: bool,
        use_tiny: bool,
        learning_rate: float,
        log_interval: int,
        diffusion_mode: str,
        diff_batch_size: int,
        diff_amp_dtype: str,
        diff_cache_all_data: bool,
        diff_cache_device: str,
        diff_num_workers: int,
        target_model_version_id: Optional[str],
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        task_dir = JOBS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "训练任务已创建。",
            "logs": ["训练任务已创建。"],
            "current_segment": 0,
            "total_segments": 0,
            "result_url": None,
            "error": None,
            "summary": None,
            "device_used": None,
            "sequence": 1,
            "meta": {
                "dataset_version_id": dataset_version_id,
                "model_name": model_name,
                "training_mode": training_mode,
                "resume_from_checkpoint_id": resume_from_checkpoint_id,
                "checkpoint_interval_steps": checkpoint_interval_steps,
                "checkpoint_keep_last": checkpoint_keep_last,
                "main_preset_id": main_preset_id,
                "main_batch_size": main_batch_size,
                "main_precision": main_precision,
                "main_all_in_mem": main_all_in_mem,
                "use_tiny": use_tiny,
                "learning_rate": learning_rate,
                "log_interval": log_interval,
                "diffusion_mode": diffusion_mode,
                "diff_batch_size": diff_batch_size,
                "diff_amp_dtype": diff_amp_dtype,
                "diff_cache_all_data": diff_cache_all_data,
                "diff_cache_device": diff_cache_device,
                "diff_num_workers": diff_num_workers,
                "target_model_version_id": target_model_version_id,
                "task_dir": str(task_dir),
                "task_kind": "training",
            },
            "updated_at": time.time(),
        }

        with self._lock:
            self._tasks[task_id] = snapshot

        worker = threading.Thread(
            target=self._run_task,
            kwargs={
                "task_id": task_id,
                "dataset_version_id": dataset_version_id,
                "model_name": model_name,
                "task_dir": task_dir,
                "device_preference": device_preference,
                "f0_predictor": f0_predictor,
                "max_steps": max_steps,
                "training_mode": training_mode,
                "resume_from_checkpoint_id": resume_from_checkpoint_id,
                "checkpoint_interval_steps": checkpoint_interval_steps,
                "checkpoint_keep_last": checkpoint_keep_last,
                "main_preset_id": main_preset_id,
                "main_batch_size": main_batch_size,
                "main_precision": main_precision,
                "main_all_in_mem": main_all_in_mem,
                "use_tiny": use_tiny,
                "learning_rate": learning_rate,
                "log_interval": log_interval,
                "diffusion_mode": diffusion_mode,
                "diff_batch_size": diff_batch_size,
                "diff_amp_dtype": diff_amp_dtype,
                "diff_cache_all_data": diff_cache_all_data,
                "diff_cache_device": diff_cache_device,
                "diff_num_workers": diff_num_workers,
                "target_model_version_id": target_model_version_id,
            },
            daemon=True,
        )
        worker.start()
        self.training_service.library_service.create_training_run(
            task_id=task_id,
            dataset_version_id=dataset_version_id,
            model_id=None,
            model_version_id=None,
            status="queued",
            stage="queued",
            message="训练任务已创建。",
            progress=0.0,
        )
        return snapshot

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            for task in self._tasks.values():
                if task["status"] not in {"completed", "failed"}:
                    return {
                        "task_id": task["task_id"],
                        "status": task["status"],
                        "task_kind": task["meta"].get("task_kind", "training"),
                    }
        return None

    def _run_task(
        self,
        *,
        task_id: str,
        dataset_version_id: str,
        model_name: str,
        task_dir: Path,
        device_preference: str,
        f0_predictor: str,
        max_steps: int,
        training_mode: str,
        resume_from_checkpoint_id: Optional[str],
        checkpoint_interval_steps: int,
        checkpoint_keep_last: int,
        main_preset_id: str,
        main_batch_size: int,
        main_precision: str,
        main_all_in_mem: bool,
        use_tiny: bool,
        learning_rate: float,
        log_interval: int,
        diffusion_mode: str,
        diff_batch_size: int,
        diff_amp_dtype: str,
        diff_cache_all_data: bool,
        diff_cache_device: str,
        diff_num_workers: int,
        target_model_version_id: Optional[str],
    ) -> None:
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
            self.training_service.library_service.update_training_run(
                task_id=task_id,
                status=status,
                stage=status,
                message=message,
                progress=round(ratio * 100, 2),
            )

        try:
            progress("queued", 0.01, "等待 GPU 锁...", 0, 0)
            with self._gpu_lock:
                summary = self.training_service.run(
                    task_id=task_id,
                    dataset_version_id=dataset_version_id,
                    model_name=model_name,
                    task_dir=task_dir,
                    device_preference=device_preference,
                    f0_predictor=f0_predictor,
                    max_steps=max_steps,
                    training_mode=training_mode,
                    resume_from_checkpoint_id=resume_from_checkpoint_id,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                    checkpoint_keep_last=checkpoint_keep_last,
                    main_preset_id=main_preset_id,
                    main_batch_size=main_batch_size,
                    main_precision=main_precision,
                    main_all_in_mem=main_all_in_mem,
                    use_tiny=use_tiny,
                    learning_rate=learning_rate,
                    log_interval=log_interval,
                    diffusion_mode=diffusion_mode,
                    diff_batch_size=diff_batch_size,
                    diff_amp_dtype=diff_amp_dtype,
                    diff_cache_all_data=diff_cache_all_data,
                    diff_cache_device=diff_cache_device,
                    diff_num_workers=diff_num_workers,
                    target_model_version_id=target_model_version_id,
                    progress=progress,
                )
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "completed"
                task["stage"] = "completed"
                task["progress"] = 100.0
                task["message"] = "训练完成。"
                task["summary"] = summary
                task["device_used"] = summary.get("device_used")
                self._append_log(task_id, "训练模型已注册到推理模型库。")
            self.training_service.library_service.update_training_run(
                task_id=task_id,
                status="completed",
                stage="completed",
                message="训练完成。",
                progress=100.0,
                device_used=summary.get("device_used"),
                model_id=summary.get("model_id"),
                model_version_id=summary.get("model_version_id"),
                summary=summary,
            )
        except torch.cuda.OutOfMemoryError as exc:
            self._fail_task(task_id, "训练时显存不足，请降低步数或缩小数据规模。", exc)
        except Exception as exc:
            self._fail_task(task_id, str(exc), exc)

    def _fail_task(self, task_id: str, message: str, exc: Exception) -> None:
        stack = traceback.format_exc()
        task_dir = Path(self._tasks[task_id]["meta"]["task_dir"])
        (task_dir / "traceback.log").write_text(stack, encoding="utf-8")
        summary = {
            "task_id": task_id,
            "status": "failed",
            "error": message,
            "traceback_file": "traceback.log",
            "torch_cuda_version": torch.version.cuda,
        }
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
        self.training_service.library_service.update_training_run(
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

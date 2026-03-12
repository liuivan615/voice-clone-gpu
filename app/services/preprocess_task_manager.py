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

from app.services.preprocess_service import PREPROCESS_JOBS_DIR, PreprocessError, PreprocessService


class PreprocessTaskManager:
    def __init__(self, preprocess_service: PreprocessService, gpu_lock: Optional[threading.Lock] = None) -> None:
        self.preprocess_service = preprocess_service
        self._tasks = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.RLock()
        self._gpu_lock = gpu_lock or threading.Lock()

    def _touch(self, task_id: str) -> None:
        self._tasks[task_id]["sequence"] += 1
        self._tasks[task_id]["updated_at"] = time.time()

    def _append_log(self, task_id: str, message: str) -> None:
        if not message:
            return
        self._tasks[task_id]["logs"].append(message)
        self._touch(task_id)
        task_dir = Path(self._tasks[task_id]["meta"]["task_dir"])
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "run.log").write_text("\n".join(self._tasks[task_id]["logs"]), encoding="utf-8")

    def create_task(self, *, source_path: Path, source_name: str, separator_engine: str) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        task_dir = PREPROCESS_JOBS_DIR / task_id
        source_dir = task_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        target_source = source_dir / f"input{source_path.suffix.lower()}"
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
            "message": "预处理任务已创建。",
            "logs": ["预处理任务已创建。"],
            "current_segment": 0,
            "total_segments": 0,
            "result_url": None,
            "error": None,
            "summary": None,
            "sequence": 1,
            "original_url": None,
            "vocals_url": None,
            "available_variants": [],
            "default_variant": None,
            "warning": None,
            "engine": separator_engine,
            "meta": {
                "source_file": source_name,
                "separator_engine": separator_engine,
                "task_dir": str(task_dir),
                "task_kind": "preprocess",
            },
            "updated_at": time.time(),
        }
        with self._lock:
            self._tasks[task_id] = snapshot

        worker = threading.Thread(
            target=self._run_task,
            kwargs={
                "task_id": task_id,
                "source_path": target_source,
                "source_name": source_name,
                "task_dir": task_dir,
                "separator_engine": separator_engine,
            },
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
                        "task_kind": task["meta"].get("task_kind", "preprocess"),
                    }
        return None

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return json.loads(json.dumps(task, ensure_ascii=False))

    def _run_task(
        self,
        *,
        task_id: str,
        source_path: Path,
        source_name: str,
        task_dir: Path,
        separator_engine: str,
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

        try:
            progress("queued", 0.01, "等待 GPU 锁...", 0, 0)
            with self._gpu_lock:
                summary = self.preprocess_service.run(
                    task_id=task_id,
                    source_path=source_path,
                    source_name=source_name,
                    task_dir=task_dir,
                    separator_engine=separator_engine,
                    progress=progress,
                    emit_log=lambda line: self._append_log(task_id, line),
                )
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "completed"
                task["stage"] = "completed"
                task["progress"] = 100.0
                task["message"] = "预处理完成。"
                task["summary"] = summary
                task["available_variants"] = summary.get("available_variants") or []
                task["default_variant"] = summary.get("default_variant")
                task["warning"] = summary.get("warning")
                task["engine"] = separator_engine
                if "original" in task["available_variants"]:
                    task["original_url"] = self.preprocess_service.variant_media_url(task_id, "original")
                if "vocals" in task["available_variants"]:
                    task["vocals_url"] = self.preprocess_service.variant_media_url(task_id, "vocals")
                self._append_log(task_id, "人声分离结果已生成。")
        except torch.cuda.OutOfMemoryError as exc:
            self._fail_task(task_id, source_name, task_dir, separator_engine, "预处理时显存不足，请改用较短音频或更轻的引擎。", exc)
        except PreprocessError as exc:
            self._fail_task(task_id, source_name, task_dir, separator_engine, str(exc), exc)
        except Exception as exc:
            self._fail_task(task_id, source_name, task_dir, separator_engine, str(exc), exc)

    def _fail_task(
        self,
        task_id: str,
        source_name: str,
        task_dir: Path,
        separator_engine: str,
        message: str,
        exc: Exception,
    ) -> None:
        stack = traceback.format_exc()
        (task_dir / "traceback.log").write_text(stack, encoding="utf-8")
        summary = self.preprocess_service.failure_summary(
            task_id=task_id,
            source_name=source_name,
            task_dir=task_dir,
            separator_engine=separator_engine,
            message=message,
        )
        torch.cuda.empty_cache()
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "failed"
            task["stage"] = "failed"
            task["progress"] = 0.0
            task["message"] = message
            task["error"] = message
            task["summary"] = summary
            task["available_variants"] = summary.get("available_variants") or []
            task["default_variant"] = summary.get("default_variant")
            task["warning"] = summary.get("warning")
            task["engine"] = separator_engine
            if "original" in task["available_variants"]:
                task["original_url"] = self.preprocess_service.variant_media_url(task_id, "original")
            task["vocals_url"] = None
            self._append_log(task_id, f"{message} 详细堆栈见 traceback.log。")

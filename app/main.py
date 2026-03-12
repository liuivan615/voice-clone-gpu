from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import JOBS_DIR, SETTINGS, STATIC_DIR, ensure_project_dirs, reload_settings, write_settings
from app.schemas import RuntimeProfile, model_to_dict
from app.services.library_service import LibraryService
from app.services.model_service import ModelService
from app.services.pipeline_service import PipelineService
from app.services.preprocess_service import PreprocessError, PreprocessService
from app.services.preprocess_task_manager import PreprocessTaskManager
from app.services.task_manager import TaskManager
from app.services.training_preset_service import (
    _precision_support_for_device,
    base_model_paths_for_preset,
    diffusion_base_model_for_preset,
    encoder_asset_available,
    get_training_preset,
)
from app.services.training_service import TrainingService
from app.services.training_task_manager import TrainingTaskManager


ensure_project_dirs()

shared_gpu_lock = threading.Lock()
submission_lock = threading.Lock()
library_service: LibraryService | None = None
model_service: ModelService | None = None
preprocess_service: PreprocessService | None = None
pipeline_service: PipelineService | None = None
task_manager: TaskManager | None = None
training_service: TrainingService | None = None
training_task_manager: TrainingTaskManager | None = None
preprocess_task_manager: PreprocessTaskManager | None = None


def rebuild_services() -> None:
    global library_service, model_service, preprocess_service, pipeline_service, task_manager, training_service, training_task_manager, preprocess_task_manager
    library_service = LibraryService()
    model_service = ModelService(library_service)
    preprocess_service = PreprocessService()
    pipeline_service = PipelineService(model_service, preprocess_service)
    task_manager = TaskManager(pipeline_service, gpu_lock=shared_gpu_lock)
    training_service = TrainingService(model_service)
    training_task_manager = TrainingTaskManager(training_service, gpu_lock=shared_gpu_lock)
    preprocess_task_manager = PreprocessTaskManager(preprocess_service, gpu_lock=shared_gpu_lock)


@asynccontextmanager
async def lifespan(_: FastAPI):
    rebuild_services()
    preprocess_service.cleanup_expired_jobs()
    incoming_dir = JOBS_DIR / "incoming"
    if incoming_dir.exists():
        shutil.rmtree(incoming_dir, ignore_errors=True)
    incoming_dir.mkdir(parents=True, exist_ok=True)
    incoming_training_dir = JOBS_DIR / "incoming_training"
    if incoming_training_dir.exists():
        shutil.rmtree(incoming_training_dir, ignore_errors=True)
    incoming_training_dir.mkdir(parents=True, exist_ok=True)
    incoming_preprocess_dir = JOBS_DIR / "incoming_preprocess"
    if incoming_preprocess_dir.exists():
        shutil.rmtree(incoming_preprocess_dir, ignore_errors=True)
    incoming_preprocess_dir.mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(title="WAV Unified GPU Frontend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

rebuild_services()


def _active_gpu_task_detail() -> Dict[str, Any] | None:
    for manager in (training_task_manager, preprocess_task_manager, task_manager):
        active_task = manager.get_active_task()
        if active_task:
            return active_task
    return None


def _ensure_no_active_gpu_task() -> None:
    active_task = _active_gpu_task_detail()
    if not active_task:
        return
    if active_task.get("task_kind") == "training":
        kind_label = "训练"
    elif active_task.get("task_kind") == "preprocess":
        kind_label = "预处理"
    else:
        kind_label = "推理"
    raise HTTPException(
        status_code=409,
        detail=f"已有{kind_label}任务 {active_task['task_id']} 正在运行或排队，请等待结束后再提交新任务。",
    )


async def _stream_upload_to_path(upload: UploadFile, target: Path, *, max_bytes: int = 0) -> int:
    bytes_written = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(4 * 1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if max_bytes and bytes_written > max_bytes:
                raise HTTPException(status_code=400, detail=f"上传文件超过 {max_bytes // (1024 * 1024)}MB 限制。")
            handle.write(chunk)
    return bytes_written


def _settings_payload() -> Dict[str, Any]:
    return {
        "app": SETTINGS["app"],
        "training_defaults": SETTINGS["training_defaults"],
        "runtime": {
            "data_root": str(library_service.data_root),
        },
    }


def _build_runtime_profile(
    *,
    profile_json: str = "",
    profile_id: str = "",
    use_diffusion: bool = False,
    speaker: str = "",
    device_preference: str = "auto",
    tran: int = 0,
    slice_db: int = -40,
    noise_scale: float = 0.4,
    pad_seconds: float = 0.5,
    f0_predictor: str = "pm",
) -> RuntimeProfile:
    defaults = model_to_dict(model_service.get_runtime_options()["default_profile"])
    if profile_json:
        try:
            defaults.update(json.loads(profile_json))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"profile_json 不是合法 JSON：{exc}") from exc
    if profile_id:
        defaults["checkpoint_path"] = profile_id
    defaults["use_diffusion"] = bool(use_diffusion)
    defaults["speaker"] = speaker or defaults.get("speaker", "")
    defaults["device_preference"] = device_preference or defaults.get("device_preference", "auto")
    defaults["tran"] = tran
    defaults["slice_db"] = slice_db
    defaults["noise_scale"] = noise_scale
    defaults["pad_seconds"] = pad_seconds
    defaults["f0_predictor"] = f0_predictor or defaults.get("f0_predictor", "pm")
    return RuntimeProfile(**defaults)


def _media_upload_limit_bytes() -> int:
    limit_mb = int(SETTINGS["app"].get("media_upload_limit_mb") or 0)
    return limit_mb * 1024 * 1024 if limit_mb > 0 else 0


def _resolve_prepared_variant_source(prepared_task_id: str, prepared_variant: str) -> Path:
    active_task = preprocess_task_manager.get_task(prepared_task_id)
    if active_task and active_task["status"] not in {"completed", "failed"}:
        raise HTTPException(status_code=409, detail="预处理尚未完成，请稍后再试。")
    try:
        return preprocess_service.resolve_variant_file(prepared_task_id, prepared_variant)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _dataset_extract_variant_name(source_name: str, variant: str) -> str:
    stem = Path(source_name or "extracted_material").stem or "extracted_material"
    suffix = "vocals" if variant == "vocals" else "original"
    return f"{stem}_{suffix}.wav"


def _start_training_task(
    *,
    dataset_version_id: str,
    model_name: str,
    device_preference: str,
    f0_predictor: str,
    max_steps: int,
    training_mode: str,
    resume_from_checkpoint_id: str,
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
    target_model_version_id: str,
) -> Dict[str, Any]:
    with submission_lock:
        _ensure_no_active_gpu_task()
        task = training_task_manager.create_task(
            dataset_version_id=dataset_version_id,
            model_name=model_name,
            device_preference=device_preference,
            f0_predictor=f0_predictor,
            max_steps=max(1, max_steps),
            training_mode=training_mode,
            resume_from_checkpoint_id=resume_from_checkpoint_id or None,
            checkpoint_interval_steps=max(1, checkpoint_interval_steps),
            checkpoint_keep_last=max(1, checkpoint_keep_last),
            main_preset_id=main_preset_id,
            main_batch_size=max(1, main_batch_size),
            main_precision=main_precision,
            main_all_in_mem=bool(main_all_in_mem),
            use_tiny=bool(use_tiny),
            learning_rate=float(learning_rate),
            log_interval=max(1, log_interval),
            diffusion_mode=diffusion_mode,
            diff_batch_size=max(1, diff_batch_size),
            diff_amp_dtype=diff_amp_dtype,
            diff_cache_all_data=bool(diff_cache_all_data),
            diff_cache_device=diff_cache_device,
            diff_num_workers=max(0, diff_num_workers),
            target_model_version_id=target_model_version_id or None,
        )
    return {"task_id": task["task_id"], "status": task["status"]}


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Static frontend is not ready yet."}, status_code=503)


@app.get("/api/runtime/options")
def get_runtime_options():
    return model_service.get_runtime_options()


@app.get("/api/settings")
def get_settings():
    return _settings_payload()


@app.post("/api/settings")
def update_settings(payload: Dict[str, Any] = Body(...)):
    with submission_lock:
        _ensure_no_active_gpu_task()
        updated_settings = {
            "app": {**SETTINGS["app"], **(payload.get("app") or {})},
            "training_defaults": {**SETTINGS["training_defaults"], **(payload.get("training_defaults") or {})},
            "default_profile": SETTINGS["default_profile"],
        }
        write_settings(updated_settings)
        rebuild_services()
    return {"ok": True, "settings": _settings_payload()}


@app.get("/api/library/overview")
def library_overview():
    datasets = library_service.list_datasets()
    models = library_service.list_models()
    runtime_options = model_service.get_runtime_options()
    return {
        "dataset_count": len(datasets),
        "model_count": len(models),
        "datasets": datasets,
        "models": models,
        "settings": _settings_payload(),
        "runtime": runtime_options,
        "training_presets": runtime_options.get("training_presets", []),
        "training_capabilities": runtime_options.get("training_capabilities", {}),
    }


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": library_service.list_datasets()}


@app.post("/api/datasets")
def create_dataset(payload: Dict[str, Any] = Body(...)):
    speaker = (payload.get("speaker") or "").strip()
    if not speaker:
        raise HTTPException(status_code=400, detail="创建数据集需要 speaker。")
    dataset = library_service.create_dataset(
        name=(payload.get("name") or "").strip() or speaker,
        speaker=speaker,
        description=(payload.get("description") or "").strip(),
    )
    return {"ok": True, "dataset": dataset}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        dataset = library_service.get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"dataset": dataset}


@app.get("/api/datasets/{dataset_id}/files/{file_id}/audio")
def get_dataset_file_audio(dataset_id: str, file_id: str):
    try:
        dataset_file = library_service.get_dataset_file(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if dataset_file["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="音频文件不属于这个数据集。")
    file_path = library_service.absolute_path(dataset_file["storage_path"])
    return FileResponse(file_path, filename=dataset_file["original_name"])


@app.post("/api/datasets/{dataset_id}/files")
async def upload_dataset_files(
    dataset_id: str,
    files: List[UploadFile] = File(...),
    auto_segment: bool = Form(True),
):
    temp_dir = JOBS_DIR / "dataset_uploads" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    uploaded = []
    try:
        try:
            for index, upload in enumerate(files, start=1):
                safe_name = Path(upload.filename or f"sample_{index}.wav").name
                if not safe_name.lower().endswith(".wav"):
                    raise HTTPException(status_code=400, detail="数据集上传当前只支持 .wav 文件。")
                temp_path = temp_dir / f"{index:03d}_{safe_name}"
                await _stream_upload_to_path(upload, temp_path)
                uploaded.append(library_service.add_dataset_file(dataset_id=dataset_id, source_path=temp_path, original_name=safe_name))
            result = {"ok": True, "uploaded": uploaded, "dataset": library_service.get_dataset(dataset_id)}
            if auto_segment:
                result["segmentation"] = library_service.segmentize_dataset(dataset_id)
                result["dataset"] = library_service.get_dataset(dataset_id)
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/datasets/{dataset_id}/segmentize")
def segmentize_dataset(dataset_id: str, payload: Dict[str, Any] = Body(default={})):
    try:
        result = library_service.segmentize_dataset(
            dataset_id,
            min_keep_seconds=float(payload.get("min_keep_seconds", 1.5)),
            max_segment_seconds=float(payload.get("max_segment_seconds", 6.0)),
            merge_gap_ms=int(payload.get("merge_gap_ms", 300)),
            energy_floor_db=float(payload.get("energy_floor_db", -45.0)),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, **result}


@app.get("/api/segments/{segment_id}/audio")
def get_segment_audio(segment_id: str):
    try:
        segment = library_service.get_segment(segment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    segment_path = library_service.absolute_path(segment["storage_path"])
    return FileResponse(segment_path, filename=segment["display_name"])


@app.post("/api/segments/{segment_id}/enabled")
def set_segment_enabled(segment_id: str, payload: Dict[str, Any] = Body(...)):
    try:
        segment = library_service.set_segment_enabled(segment_id, bool(payload.get("enabled", True)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "segment": segment}


@app.post("/api/datasets/{dataset_id}/versions")
def create_dataset_version(dataset_id: str, payload: Dict[str, Any] = Body(default={})):
    try:
        version = library_service.create_dataset_version(dataset_id, label=payload.get("label", ""), notes=payload.get("notes", ""))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400 if isinstance(exc, RuntimeError) else 404, detail=str(exc))
    return {"ok": True, "dataset_version": version}


@app.post("/api/datasets/{dataset_id}/extract")
async def create_dataset_extract_task(
    dataset_id: str,
    file: UploadFile = File(...),
    separator_engine: str = Form("demucs"),
):
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="请先选择待处理素材文件。")

    try:
        library_service.get_dataset(dataset_id)
        preprocess_service.validate_media_extension(safe_name)
        preprocess_service.ensure_engine_available(separator_engine)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PreprocessError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    incoming_dir = JOBS_DIR / "incoming_preprocess"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    source_path = incoming_dir / f"{uuid.uuid4().hex}_{safe_name}"
    max_bytes = _media_upload_limit_bytes()
    try:
        await _stream_upload_to_path(file, source_path, max_bytes=max_bytes)
    except HTTPException:
        source_path.unlink(missing_ok=True)
        raise

    try:
        with submission_lock:
            _ensure_no_active_gpu_task()
            task = preprocess_task_manager.create_task(
                source_path=source_path,
                source_name=safe_name,
                separator_engine=separator_engine,
            )
    except Exception:
        source_path.unlink(missing_ok=True)
        raise
    return {"task_id": task["task_id"], "dataset_id": dataset_id, "status": task["status"]}


@app.post("/api/datasets/{dataset_id}/extract/{task_id}/confirm")
def confirm_dataset_extract(
    dataset_id: str,
    task_id: str,
    payload: Dict[str, Any] = Body(default={}),
):
    variant = (payload.get("variant") or "vocals").strip().lower()
    auto_segment = bool(payload.get("auto_segment", True))
    if variant not in {"original", "vocals"}:
        raise HTTPException(status_code=400, detail="variant 只支持 original 或 vocals。")

    try:
        library_service.get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    source_path = _resolve_prepared_variant_source(task_id, variant)
    summary = preprocess_service.load_summary(task_id)
    if not summary:
        raise HTTPException(status_code=404, detail="提取结果不存在或已过期，请重新提取。")

    try:
        added_file = library_service.add_dataset_file(
            dataset_id=dataset_id,
            source_path=source_path,
            original_name=_dataset_extract_variant_name(summary.get("source_file") or task_id, variant),
        )
        result = {
            "ok": True,
            "task_id": task_id,
            "dataset_id": dataset_id,
            "variant": variant,
            "added_file": added_file,
        }
        if auto_segment:
            result["segmentation"] = library_service.segmentize_dataset(dataset_id)
        result["dataset"] = library_service.get_dataset(dataset_id)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/datasets/{dataset_id}/versions")
def list_dataset_versions(dataset_id: str):
    return {"versions": library_service.list_dataset_versions(dataset_id)}


@app.get("/api/dataset-versions/{version_id}")
def get_dataset_version(version_id: str):
    try:
        version = library_service.get_dataset_version(version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"dataset_version": version}


@app.get("/api/dataset-files/{file_id}/media")
def get_dataset_file_media(file_id: str):
    try:
        file_row = library_service.get_dataset_file(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    file_path = library_service.absolute_path(file_row["storage_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在。")
    return FileResponse(file_path, filename=file_row["original_name"], media_type="audio/wav")


@app.get("/api/datasets/{dataset_id}/files/{file_id}/audio")
def get_dataset_file_audio_alias(dataset_id: str, file_id: str):
    try:
        file_row = library_service.get_dataset_file(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if file_row["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="音频文件不存在。")
    return get_dataset_file_media(file_id)


@app.get("/api/dataset-segments/{segment_id}/media")
def get_dataset_segment_media(segment_id: str):
    try:
        segment = library_service.get_dataset_segment(segment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    file_path = library_service.absolute_path(segment["storage_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="音频片段不存在。")
    return FileResponse(file_path, filename=segment["display_name"], media_type="audio/wav")


@app.get("/api/segments/{segment_id}/audio")
def get_dataset_segment_audio_alias(segment_id: str):
    return get_dataset_segment_media(segment_id)


@app.get("/api/dataset-version-segments/{segment_id}/media")
def get_dataset_version_segment_media(segment_id: str):
    try:
        segment = library_service.get_dataset_version_segment(segment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    file_path = library_service.absolute_path(segment["storage_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="音频片段不存在。")
    return FileResponse(file_path, filename=segment["display_name"], media_type="audio/wav")


@app.get("/api/models")
def list_models():
    return {"models": library_service.list_models()}


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    try:
        model = library_service.get_model(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"model": model}


@app.get("/api/model-versions/{version_id}")
def get_model_version(version_id: str):
    try:
        version = library_service.get_model_version(version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    version["checkpoints"] = library_service.list_checkpoints(version_id)
    return {"model_version": version}


@app.post("/api/models/{model_id}/default-version")
def set_default_model_version(model_id: str, payload: Dict[str, Any] = Body(...)):
    version_id = payload.get("model_version_id")
    if not version_id:
        raise HTTPException(status_code=400, detail="需要 model_version_id。")
    try:
        model = library_service.set_default_model_version(model_id, version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "model": model}


@app.post("/api/model-versions/{version_id}/continue-training")
def continue_training_from_version(version_id: str, payload: Dict[str, Any] = Body(default={})):
    try:
        version = library_service.get_model_version(version_id)
        checkpoints = library_service.list_checkpoints(version_id)
        model = library_service.get_model(version["model_id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    dataset_version_id = payload.get("dataset_version_id") or version.get("dataset_version_id")
    if not dataset_version_id:
        raise HTTPException(status_code=400, detail="当前模型版本没有关联数据集版本，无法继续训练。")
    resume_checkpoint_id = payload.get("checkpoint_id") or (checkpoints[0]["id"] if checkpoints else "")
    if not resume_checkpoint_id:
        raise HTTPException(status_code=400, detail="当前模型版本没有可继续训练的 checkpoint。")
    runtime_capabilities = model_service.get_runtime_options().get("training_capabilities") or {}
    main_defaults = runtime_capabilities.get("main_defaults") or {}
    diffusion_defaults = runtime_capabilities.get("diffusion_defaults") or {}
    return _start_training_task(
        dataset_version_id=dataset_version_id,
        model_name=payload.get("model_name") or model["name"],
        device_preference=payload.get("device_preference") or version.get("device_preference") or "auto",
        f0_predictor=payload.get("f0_predictor") or version.get("f0_predictor") or SETTINGS["training_defaults"]["f0_predictor"],
        max_steps=int(payload.get("max_steps") or SETTINGS["training_defaults"]["step_count"]),
        training_mode="resume",
        resume_from_checkpoint_id=resume_checkpoint_id,
        checkpoint_interval_steps=int(payload.get("checkpoint_interval_steps") or SETTINGS["training_defaults"]["checkpoint_interval_steps"]),
        checkpoint_keep_last=int(payload.get("checkpoint_keep_last") or SETTINGS["training_defaults"]["checkpoint_keep_last"]),
        main_preset_id=payload.get("main_preset_id") or version.get("main_preset_id") or "balanced",
        main_batch_size=int(payload.get("main_batch_size") or main_defaults.get("main_batch_size") or 6),
        main_precision=str(payload.get("main_precision") or main_defaults.get("main_precision") or "fp32"),
        main_all_in_mem=bool(payload.get("main_all_in_mem", main_defaults.get("main_all_in_mem", False))),
        use_tiny=bool(payload.get("use_tiny", version.get("use_tiny", False))),
        learning_rate=float(payload.get("learning_rate") or main_defaults.get("learning_rate") or 0.0001),
        log_interval=int(payload.get("log_interval") or main_defaults.get("log_interval") or 200),
        diffusion_mode=str(payload.get("diffusion_mode") or diffusion_defaults.get("diffusion_mode") or "disabled"),
        diff_batch_size=int(payload.get("diff_batch_size") or diffusion_defaults.get("diff_batch_size") or 48),
        diff_amp_dtype=str(payload.get("diff_amp_dtype") or diffusion_defaults.get("diff_amp_dtype") or "fp32"),
        diff_cache_all_data=bool(payload.get("diff_cache_all_data", diffusion_defaults.get("diff_cache_all_data", True))),
        diff_cache_device=str(payload.get("diff_cache_device") or diffusion_defaults.get("diff_cache_device") or "cpu"),
        diff_num_workers=int(payload.get("diff_num_workers") or diffusion_defaults.get("diff_num_workers") or 4),
        target_model_version_id=payload.get("target_model_version_id") or "",
    )


@app.post("/api/model-versions/{version_id}/retrain")
def retrain_from_version(version_id: str, payload: Dict[str, Any] = Body(default={})):
    try:
        version = library_service.get_model_version(version_id)
        model = library_service.get_model(version["model_id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    dataset_version_id = payload.get("dataset_version_id") or version.get("dataset_version_id")
    if not dataset_version_id:
        raise HTTPException(status_code=400, detail="当前模型版本没有关联数据集版本，无法直接完全重训。")
    runtime_capabilities = model_service.get_runtime_options().get("training_capabilities") or {}
    main_defaults = runtime_capabilities.get("main_defaults") or {}
    diffusion_defaults = runtime_capabilities.get("diffusion_defaults") or {}
    return _start_training_task(
        dataset_version_id=dataset_version_id,
        model_name=payload.get("model_name") or model["name"],
        device_preference=payload.get("device_preference") or version.get("device_preference") or "auto",
        f0_predictor=payload.get("f0_predictor") or version.get("f0_predictor") or SETTINGS["training_defaults"]["f0_predictor"],
        max_steps=int(payload.get("max_steps") or SETTINGS["training_defaults"]["step_count"]),
        training_mode="new",
        resume_from_checkpoint_id="",
        checkpoint_interval_steps=int(payload.get("checkpoint_interval_steps") or SETTINGS["training_defaults"]["checkpoint_interval_steps"]),
        checkpoint_keep_last=int(payload.get("checkpoint_keep_last") or SETTINGS["training_defaults"]["checkpoint_keep_last"]),
        main_preset_id=payload.get("main_preset_id") or version.get("main_preset_id") or "balanced",
        main_batch_size=int(payload.get("main_batch_size") or main_defaults.get("main_batch_size") or 6),
        main_precision=str(payload.get("main_precision") or main_defaults.get("main_precision") or "fp32"),
        main_all_in_mem=bool(payload.get("main_all_in_mem", main_defaults.get("main_all_in_mem", False))),
        use_tiny=bool(payload.get("use_tiny", version.get("use_tiny", False))),
        learning_rate=float(payload.get("learning_rate") or main_defaults.get("learning_rate") or 0.0001),
        log_interval=int(payload.get("log_interval") or main_defaults.get("log_interval") or 200),
        diffusion_mode=str(payload.get("diffusion_mode") or diffusion_defaults.get("diffusion_mode") or "disabled"),
        diff_batch_size=int(payload.get("diff_batch_size") or diffusion_defaults.get("diff_batch_size") or 48),
        diff_amp_dtype=str(payload.get("diff_amp_dtype") or diffusion_defaults.get("diff_amp_dtype") or "fp32"),
        diff_cache_all_data=bool(payload.get("diff_cache_all_data", diffusion_defaults.get("diff_cache_all_data", True))),
        diff_cache_device=str(payload.get("diff_cache_device") or diffusion_defaults.get("diff_cache_device") or "cpu"),
        diff_num_workers=int(payload.get("diff_num_workers") or diffusion_defaults.get("diff_num_workers") or 4),
        target_model_version_id=payload.get("target_model_version_id") or "",
    )


@app.post("/api/runtime/load-model")
def load_model(payload: Dict[str, Any]):
    defaults = model_to_dict(model_service.get_runtime_options()["default_profile"])
    profile_id = payload.get("profile_id")
    if profile_id:
        defaults["checkpoint_path"] = profile_id
    defaults["use_diffusion"] = bool(payload.get("use_diffusion", False))
    if payload.get("speaker") is not None:
        defaults["speaker"] = payload.get("speaker") or ""
    if payload.get("device_preference") is not None:
        defaults["device_preference"] = payload.get("device_preference") or "auto"
    try:
        with submission_lock:
            _ensure_no_active_gpu_task()
            with shared_gpu_lock:
                snapshot = model_service.load_profile(RuntimeProfile(**defaults))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "ok": True,
        "message": "模型加载成功。",
        "snapshot": snapshot,
    }


@app.post("/api/runtime/unload-model")
def unload_model():
    try:
        with submission_lock:
            _ensure_no_active_gpu_task()
            with shared_gpu_lock:
                model_service.unload()
                snapshot = model_service.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "message": "模型已卸载。", "snapshot": snapshot}


@app.post("/api/runtime/import-model")
async def import_model(
    checkpoint: UploadFile = File(...),
    config: UploadFile = File(...),
    label: str = Form(""),
    speaker: str = Form(""),
):
    if not checkpoint.filename.lower().endswith(".pth"):
        raise HTTPException(status_code=400, detail="checkpoint 只支持 .pth 文件。")
    if not config.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="config 只支持 .json 文件。")

    import_dir = JOBS_DIR / "import_model_temp" / uuid.uuid4().hex
    import_dir.mkdir(parents=True, exist_ok=False)

    try:
        checkpoint_path = import_dir / Path(checkpoint.filename).name
        config_path = import_dir / Path(config.filename).name
        checkpoint_path.write_bytes(await checkpoint.read())
        config_path.write_bytes(await config.read())

        if not label:
            label = checkpoint_path.stem
        metadata = model_service.register_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            label=label,
            speaker=speaker,
            source="imported",
        )
        return {"ok": True, "message": "模型已导入。", "model": metadata}
    finally:
        shutil.rmtree(import_dir, ignore_errors=True)


@app.post("/api/preprocess/tasks")
async def create_preprocess_task(
    file: UploadFile = File(...),
    separator_engine: str = Form("demucs"),
):
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="请先选择待处理媒体文件。")
    try:
        preprocess_service.validate_media_extension(safe_name)
        preprocess_service.ensure_engine_available(separator_engine)
    except PreprocessError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    incoming_dir = JOBS_DIR / "incoming_preprocess"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    source_path = incoming_dir / f"{uuid.uuid4().hex}_{safe_name}"
    max_bytes = _media_upload_limit_bytes()
    try:
        await _stream_upload_to_path(file, source_path, max_bytes=max_bytes)
    except HTTPException:
        source_path.unlink(missing_ok=True)
        raise

    try:
        with submission_lock:
            _ensure_no_active_gpu_task()
            task = preprocess_task_manager.create_task(
                source_path=source_path,
                source_name=safe_name,
                separator_engine=separator_engine,
            )
    except Exception:
        source_path.unlink(missing_ok=True)
        raise
    return {"task_id": task["task_id"], "status": task["status"]}


@app.post("/api/tasks")
async def create_task(
    file: UploadFile = File(default=None),
    prepared_task_id: str = Form(""),
    prepared_variant: str = Form(""),
    profile_json: str = Form(""),
    profile_id: str = Form(""),
    use_diffusion: bool = Form(False),
    speaker: str = Form(""),
    device_preference: str = Form("auto"),
    tran: int = Form(0),
    slice_db: int = Form(-40),
    noise_scale: float = Form(0.4),
    pad_seconds: float = Form(0.5),
    f0_predictor: str = Form("pm"),
):
    if file is not None and prepared_task_id:
        raise HTTPException(status_code=400, detail="file 与 prepared_task_id 不能同时提交。")
    if file is None and not prepared_task_id:
        raise HTTPException(status_code=400, detail="请上传媒体文件，或提供 prepared_task_id。")

    profile = _build_runtime_profile(
        profile_json=profile_json,
        profile_id=profile_id,
        use_diffusion=use_diffusion,
        speaker=speaker,
        device_preference=device_preference,
        tran=tran,
        slice_db=slice_db,
        noise_scale=noise_scale,
        pad_seconds=pad_seconds,
        f0_predictor=f0_predictor,
    )
    incoming_dir = JOBS_DIR / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    if prepared_task_id:
        variant = (prepared_variant or "original").strip().lower()
        prepared_source = _resolve_prepared_variant_source(prepared_task_id, variant)
        safe_name = f"{prepared_task_id}_{variant}{prepared_source.suffix.lower() or '.wav'}"
        source_path = incoming_dir / f"{uuid.uuid4().hex}_{safe_name}"
        shutil.copy2(prepared_source, source_path)
    else:
        safe_name = Path(file.filename or "").name
        if not safe_name:
            raise HTTPException(status_code=400, detail="请先选择待处理媒体文件。")
        try:
            preprocess_service.validate_media_extension(safe_name)
        except PreprocessError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        source_path = incoming_dir / f"{uuid.uuid4().hex}_{safe_name}"
        max_bytes = _media_upload_limit_bytes()
        try:
            await _stream_upload_to_path(file, source_path, max_bytes=max_bytes)
        except HTTPException:
            source_path.unlink(missing_ok=True)
            raise

    try:
        with submission_lock:
            _ensure_no_active_gpu_task()
            task = task_manager.create_task(source_path=source_path, source_name=safe_name, profile=profile)
    except Exception:
        if source_path.exists():
            source_path.unlink(missing_ok=True)
        raise
    return {"task_id": task["task_id"], "status": task["status"]}


@app.post("/api/training/tasks")
async def create_training_task(
    files: List[UploadFile] = File(default=None),
    dataset_version_id: str = Form(""),
    model_name: str = Form(""),
    device_preference: str = Form("auto"),
    f0_predictor: str = Form(""),
    max_steps: int = Form(0),
    training_mode: str = Form("new"),
    resume_from_checkpoint_id: str = Form(""),
    checkpoint_interval_steps: int = Form(0),
    checkpoint_keep_last: int = Form(0),
    dataset_name: str = Form(""),
    speaker_name: str = Form(""),
    main_preset_id: str = Form("balanced"),
    main_batch_size: int = Form(0),
    main_precision: str = Form(""),
    main_all_in_mem: bool = Form(False),
    use_tiny: bool = Form(False),
    learning_rate: float = Form(0.0),
    log_interval: int = Form(0),
    diffusion_mode: str = Form("disabled"),
    diff_batch_size: int = Form(0),
    diff_amp_dtype: str = Form(""),
    diff_cache_all_data: bool = Form(True),
    diff_cache_device: str = Form("cpu"),
    diff_num_workers: int = Form(0),
    target_model_version_id: str = Form(""),
):
    training_defaults = SETTINGS["training_defaults"]
    f0_choice = f0_predictor or training_defaults["f0_predictor"]
    step_count = max_steps or training_defaults["step_count"]
    interval_steps = checkpoint_interval_steps or training_defaults["checkpoint_interval_steps"]
    keep_last = checkpoint_keep_last or training_defaults["checkpoint_keep_last"]
    runtime_capabilities = model_service.get_runtime_options().get("training_capabilities") or {}
    main_defaults = runtime_capabilities.get("main_defaults") or {}
    diffusion_defaults = runtime_capabilities.get("diffusion_defaults") or {}
    resolved_main_batch_size = main_batch_size or int(main_defaults.get("main_batch_size") or 6)
    resolved_main_precision = (main_precision or str(main_defaults.get("main_precision") or "fp32")).strip().lower()
    resolved_learning_rate = learning_rate or float(main_defaults.get("learning_rate") or 0.0001)
    resolved_log_interval = log_interval or int(main_defaults.get("log_interval") or 200)
    resolved_diffusion_mode = (diffusion_mode or str(diffusion_defaults.get("diffusion_mode") or "disabled")).strip().lower()
    resolved_diff_batch_size = diff_batch_size or int(diffusion_defaults.get("diff_batch_size") or 48)
    resolved_diff_amp_dtype = (diff_amp_dtype or str(diffusion_defaults.get("diff_amp_dtype") or "fp32")).strip().lower()
    resolved_diff_cache_device = (diff_cache_device or str(diffusion_defaults.get("diff_cache_device") or "cpu")).strip().lower()
    resolved_diff_num_workers = diff_num_workers or int(diffusion_defaults.get("diff_num_workers") or 4)
    resolved_diff_cache_all_data = bool(diff_cache_all_data if diff_cache_all_data is not None else diffusion_defaults.get("diff_cache_all_data", True))
    resolved_main_all_in_mem = bool(main_all_in_mem if main_all_in_mem is not None else main_defaults.get("main_all_in_mem", False))

    if training_mode == "diffusion_only":
        if files:
            raise HTTPException(status_code=400, detail="diffusion_only 不支持重新上传训练音频。")
        if not target_model_version_id:
            raise HTTPException(status_code=400, detail="diffusion_only 需要目标模型版本。")
        try:
            target_version = library_service.get_model_version(target_model_version_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        dataset_version_id = target_version.get("dataset_version_id") or dataset_version_id
        if not dataset_version_id:
            raise HTTPException(status_code=400, detail="关联数据集版本不可用，请重新选择数据集后完全重训。")

    if dataset_version_id:
        try:
            dataset_version = library_service.get_dataset_version(dataset_version_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="关联数据集版本不可用，请重新选择数据集后完全重训。")
        if not dataset_version.get("segments"):
            raise HTTPException(status_code=400, detail="所选数据集版本没有任何音频片段，无法训练。")
        resolved_model_name = (model_name or dataset_version["speaker"]).strip() or dataset_version["speaker"]

        # Pre-flight: validate encoder assets and base models before queuing
        try:
            preset_def = get_training_preset(main_preset_id)
            if not encoder_asset_available(preset_def.encoder):
                raise HTTPException(status_code=400, detail=f"当前便携包缺少 {preset_def.encoder} 对应的编码器资产。")
            base_model_paths_for_preset(main_preset_id, use_tiny=use_tiny)
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if resolved_diffusion_mode != "disabled" or training_mode == "diffusion_only":
            try:
                diffusion_base_model_for_preset(main_preset_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        # Pre-flight: validate precision/dtype against device capabilities
        device_precision = _precision_support_for_device(device_preference)
        if not device_precision.get(resolved_main_precision, False):
            raise HTTPException(status_code=400, detail=f"当前设备不支持训练精度 {resolved_main_precision}。")
        if (resolved_diffusion_mode != "disabled" or training_mode == "diffusion_only") and not device_precision.get(resolved_diff_amp_dtype, False):
            raise HTTPException(status_code=400, detail=f"当前设备不支持扩散 AMP 精度 {resolved_diff_amp_dtype}。")

        return _start_training_task(
            dataset_version_id=dataset_version_id,
            model_name=resolved_model_name,
            device_preference=device_preference,
            f0_predictor=f0_choice,
            max_steps=min(step_count, 50000),
            training_mode=training_mode,
            resume_from_checkpoint_id=resume_from_checkpoint_id,
            checkpoint_interval_steps=interval_steps,
            checkpoint_keep_last=keep_last,
            main_preset_id=main_preset_id,
            main_batch_size=resolved_main_batch_size,
            main_precision=resolved_main_precision,
            main_all_in_mem=resolved_main_all_in_mem,
            use_tiny=use_tiny,
            learning_rate=resolved_learning_rate,
            log_interval=resolved_log_interval,
            diffusion_mode=resolved_diffusion_mode,
            diff_batch_size=resolved_diff_batch_size,
            diff_amp_dtype=resolved_diff_amp_dtype,
            diff_cache_all_data=resolved_diff_cache_all_data,
            diff_cache_device=resolved_diff_cache_device,
            diff_num_workers=resolved_diff_num_workers,
            target_model_version_id=target_model_version_id,
        )

    if not files:
        raise HTTPException(status_code=400, detail="请先上传训练音频，或选择一个已保存的数据集版本。")

    normalized_speaker = (speaker_name or "speaker").strip().replace(" ", "_")
    dataset = library_service.create_dataset(
        name=(dataset_name or normalized_speaker or "dataset").strip(),
        speaker=normalized_speaker,
        description="从训练入口直接创建的数据集。",
    )
    temp_dir = JOBS_DIR / "incoming_training" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        for index, upload in enumerate(files, start=1):
            safe_name = Path(upload.filename or f"sample_{index}.wav").name
            if not safe_name.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="训练入口当前只支持 .wav 文件。")
            target = temp_dir / f"{index:03d}_{safe_name}"
            await _stream_upload_to_path(upload, target)
            library_service.add_dataset_file(dataset_id=dataset["id"], source_path=target, original_name=safe_name)
        library_service.segmentize_dataset(dataset["id"])
        version = library_service.create_dataset_version(dataset["id"], label="v1", notes="由训练入口自动创建")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    resolved_model_name = (model_name or normalized_speaker).strip() or normalized_speaker
    return _start_training_task(
        dataset_version_id=version["id"],
        model_name=resolved_model_name,
        device_preference=device_preference,
        f0_predictor=f0_choice,
        max_steps=min(step_count, 50000),
        training_mode=training_mode,
        resume_from_checkpoint_id=resume_from_checkpoint_id,
        checkpoint_interval_steps=interval_steps,
        checkpoint_keep_last=keep_last,
        main_preset_id=main_preset_id,
        main_batch_size=resolved_main_batch_size,
        main_precision=resolved_main_precision,
        main_all_in_mem=resolved_main_all_in_mem,
        use_tiny=use_tiny,
        learning_rate=resolved_learning_rate,
        log_interval=resolved_log_interval,
        diffusion_mode=resolved_diffusion_mode,
        diff_batch_size=resolved_diff_batch_size,
        diff_amp_dtype=resolved_diff_amp_dtype,
        diff_cache_all_data=resolved_diff_cache_all_data,
        diff_cache_device=resolved_diff_cache_device,
        diff_num_workers=resolved_diff_num_workers,
        target_model_version_id=target_model_version_id,
    )


@app.get("/api/preprocess/tasks/{task_id}")
def get_preprocess_task(task_id: str):
    task = preprocess_task_manager.get_task(task_id)
    if task:
        return task
    snapshot = preprocess_service.load_snapshot(task_id)
    if snapshot:
        return snapshot
    raise HTTPException(status_code=404, detail="预处理任务不存在。")


@app.get("/api/preprocess/tasks/{task_id}/media/{variant}")
def get_preprocess_task_media(task_id: str, variant: str):
    try:
        media_path = preprocess_service.resolve_variant_file(task_id, variant)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="预处理媒体文件不存在。")
    return FileResponse(media_path, filename=media_path.name, media_type="audio/wav")


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return task


@app.get("/api/training/tasks/{task_id}")
def get_training_task(task_id: str):
    task = training_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="训练任务不存在。")
    return task


@app.get("/api/tasks/{task_id}/result")
def download_result(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在。")
    if task["status"] != "completed" or not task.get("summary") or not task["summary"].get("result_file"):
        raise HTTPException(status_code=409, detail="任务尚未生成结果文件。")

    result_file = JOBS_DIR / task_id / "result" / task["summary"]["result_file"]
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在。")
    return FileResponse(result_file, filename=result_file.name)


@app.websocket("/ws/tasks/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            snapshot = task_manager.get_task(task_id)
            if not snapshot:
                await websocket.send_json({"error": "task_not_found", "task_id": task_id})
                await websocket.close(code=1008)
                return
            if snapshot["sequence"] != last_sequence:
                last_sequence = snapshot["sequence"]
                await websocket.send_json(snapshot)
            if snapshot["status"] in {"completed", "failed"}:
                await websocket.close()
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/preprocess/tasks/{task_id}")
async def preprocess_task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            snapshot = preprocess_task_manager.get_task(task_id) or preprocess_service.load_snapshot(task_id)
            if not snapshot:
                await websocket.send_json({"error": "task_not_found", "task_id": task_id})
                await websocket.close(code=1008)
                return
            if snapshot["sequence"] != last_sequence:
                last_sequence = snapshot["sequence"]
                await websocket.send_json(snapshot)
            if snapshot["status"] in {"completed", "failed"}:
                await websocket.close()
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/training/tasks/{task_id}")
async def training_task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            snapshot = training_task_manager.get_task(task_id)
            if not snapshot:
                await websocket.send_json({"error": "task_not_found", "task_id": task_id})
                await websocket.close(code=1008)
                return
            if snapshot["sequence"] != last_sequence:
                last_sequence = snapshot["sequence"]
                await websocket.send_json(snapshot)
            if snapshot["status"] in {"completed", "failed"}:
                await websocket.close()
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/training/{task_id}")
async def training_task_ws_alias(websocket: WebSocket, task_id: str):
    await training_task_ws(websocket, task_id)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=SETTINGS["app"]["host"],
        port=int(SETTINGS["app"]["port"]),
        reload=False,
    )

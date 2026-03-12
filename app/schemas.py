from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuntimeProfile(BaseModel):
    checkpoint_path: str = ""
    config_path: str = "configs/config.json"
    cluster_path: str = ""
    diffusion_model_path: str = ""
    diffusion_config_path: str = "configs/diffusion.yaml"
    use_diffusion: bool = False
    speaker: str = ""
    device_preference: str = "auto"
    tran: int = 0
    slice_db: int = -40
    slice_min_length_ms: int = 5000
    noise_scale: float = 0.4
    pad_seconds: float = 0.5
    f0_predictor: str = "pm"
    output_format: str = "wav"
    auto_predict_f0: bool = False
    cluster_infer_ratio: float = 0.0


class GpuCompatibility(BaseModel):
    cuda_available: bool
    torch_cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_blackwell: bool = False
    meets_cuda_requirement: bool = True
    meets_driver_requirement: bool = True
    can_use_gpu: bool = False
    block_gpu_inference: bool = False
    messages: List[str] = Field(default_factory=list)


class RuntimeOptions(BaseModel):
    models: List[str]
    configs: List[str]
    cluster_models: List[str]
    diffusion_models: List[str]
    devices: List[Dict[str, str]]
    speakers: List[str]
    default_profile: RuntimeProfile
    loaded_profile: Optional[RuntimeProfile] = None
    loaded_device: Optional[str] = None
    gpu_compatibility: GpuCompatibility
    runtime_root: str


class LoadModelRequest(BaseModel):
    profile: RuntimeProfile


class TaskSummary(BaseModel):
    task_id: str
    source_file: str
    slice_count: int = 0
    min_duration: float = 0.0
    max_duration: float = 0.0
    total_duration: float = 0.0
    device_used: Optional[str] = None
    speaker: Optional[str] = None
    result_file: Optional[str] = None
    status: str
    error: Optional[str] = None
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    torch_cuda_version: Optional[str] = None


class TaskSnapshot(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    logs: List[str] = Field(default_factory=list)
    current_segment: int = 0
    total_segments: int = 0
    result_url: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[TaskSummary] = None
    sequence: int = 0
    meta: Dict[str, Any] = Field(default_factory=dict)


def model_to_dict(model, **kwargs):
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch

from app.config import SETTINGS


def _parse_version(value: Optional[str]) -> Tuple[int, ...]:
    if not value:
        return tuple()
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _driver_version() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def inspect_gpu_compatibility() -> Dict[str, Any]:
    minimum_driver = SETTINGS["app"]["minimum_driver_version"]
    info: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
        "driver_version": _driver_version(),
        "is_blackwell": False,
        "meets_cuda_requirement": True,
        "meets_driver_requirement": True,
        "can_use_gpu": False,
        "block_gpu_inference": False,
        "messages": [],
    }

    if not info["cuda_available"]:
        info["messages"].append("未检测到 CUDA。应用将只允许 CPU 回退或提示修复环境。")
        return info

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    capability_text = f"{capability[0]}.{capability[1]}"
    is_blackwell = capability >= (12, 0) or "RTX 50" in gpu_name.upper() or "BLACKWELL" in gpu_name.upper()
    torch_cuda_version = info["torch_cuda_version"] or ""

    info["gpu_name"] = gpu_name
    info["compute_capability"] = capability_text
    info["is_blackwell"] = is_blackwell
    info["can_use_gpu"] = True

    cuda_tuple = _parse_version(torch_cuda_version)
    if is_blackwell and cuda_tuple < (12, 8):
        info["meets_cuda_requirement"] = False
        info["block_gpu_inference"] = True
        info["can_use_gpu"] = False
        info["messages"].append(
            "检测到 RTX 50 / Blackwell，但当前 Torch CUDA 版本低于 12.8；此项目要求至少使用 CUDA 12.8 路线。"
        )
    elif is_blackwell and cuda_tuple >= (12, 8) and not torch_cuda_version.startswith("12.8"):
        info["messages"].append(
            f"当前为 Torch CUDA {torch_cuda_version}，高于项目首选的 cu128 基线，允许继续运行。"
        )

    driver_version = info["driver_version"]
    if driver_version and _parse_version(driver_version) < _parse_version(minimum_driver):
        info["meets_driver_requirement"] = False
        info["block_gpu_inference"] = True
        info["can_use_gpu"] = False
        info["messages"].append(f"当前驱动版本 {driver_version} 低于项目要求的 {minimum_driver}。")
    elif is_blackwell and not driver_version:
        info["messages"].append("无法读取 NVIDIA 驱动版本，请手动确认驱动满足项目要求。")

    if info["can_use_gpu"]:
        info["messages"].append(f"已识别 GPU：{gpu_name}，Compute Capability {capability_text}。")

    info["compatible"] = bool(info["can_use_gpu"])
    info["message"] = "；".join(info["messages"]) if info["messages"] else "CUDA 状态正常。"
    info["blocked_reason"] = info["message"] if info["block_gpu_inference"] else None

    return info


def available_device_options() -> List[Dict[str, str]]:
    options = [{"label": "Auto", "value": "auto"}]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(index)
            options.append({"label": f"CUDA:{index} {gpu_name}", "value": f"cuda:{index}"})
    options.append({"label": "CPU", "value": "cpu"})
    return options

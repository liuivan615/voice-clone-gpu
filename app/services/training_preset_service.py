from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from app.config import RUNTIME_DIR


@dataclass(frozen=True)
class TrainingPresetDefinition:
    preset_id: str
    label: str
    encoder: str
    encoder_dim: int
    ssl_dim: int
    gin_channels: int
    filter_channels: int
    recommended_vram_gb: int
    description: str
    is_default: bool = False


PRESET_DEFINITIONS: List[TrainingPresetDefinition] = [
    TrainingPresetDefinition(
        preset_id="balanced",
        label="均衡",
        encoder="vec768l12",
        encoder_dim=768,
        ssl_dim=768,
        gin_channels=768,
        filter_channels=768,
        recommended_vram_gb=6,
        description="推荐大多数场景，兼顾效果、速度和显存占用。",
        is_default=True,
    ),
    TrainingPresetDefinition(
        preset_id="high_quality",
        label="高质量",
        encoder="whisper-ppg",
        encoder_dim=1024,
        ssl_dim=1024,
        gin_channels=1024,
        filter_channels=1024,
        recommended_vram_gb=12,
        description="内容还原更精确，但训练更慢、资源占用更高。",
    ),
    TrainingPresetDefinition(
        preset_id="light",
        label="轻量",
        encoder="hubertsoft",
        encoder_dim=256,
        ssl_dim=256,
        gin_channels=256,
        filter_channels=768,
        recommended_vram_gb=4,
        description="显存不足时优先尝试的轻量方案。",
    ),
]

PRESET_INDEX = {item.preset_id: item for item in PRESET_DEFINITIONS}
ENCODER_TO_PRESET = {item.encoder: item.preset_id for item in PRESET_DEFINITIONS}


def get_training_preset(preset_id: str) -> TrainingPresetDefinition:
    preset = PRESET_INDEX.get((preset_id or "").strip())
    if not preset:
        raise ValueError(f"未知训练套餐：{preset_id}")
    return preset


def infer_preset_id_from_encoder(encoder: str) -> Optional[str]:
    return ENCODER_TO_PRESET.get((encoder or "").strip())


def infer_use_tiny_from_config(config: Dict[str, Any]) -> bool:
    model = config.get("model") or {}
    return bool(
        model.get("use_depthwise_conv")
        and model.get("flow_share_parameter")
        and int(model.get("filter_channels") or 0) == 512
    )


def resolve_architecture_fields(encoder: str, *, use_tiny: bool = False) -> Dict[str, int]:
    normalized = (encoder or "").strip()
    if normalized in {"vec768l12", "dphubert", "wavlmbase+"}:
        fields = {"ssl_dim": 768, "gin_channels": 768, "filter_channels": 768}
    elif normalized in {"vec256l9", "hubertsoft"}:
        fields = {"ssl_dim": 256, "gin_channels": 256, "filter_channels": 768}
    elif normalized in {"whisper-ppg", "cnhubertlarge"}:
        fields = {"ssl_dim": 1024, "gin_channels": 1024, "filter_channels": 1024}
    elif normalized == "whisper-ppg-large":
        fields = {"ssl_dim": 1280, "gin_channels": 1280, "filter_channels": 1280}
    else:
        fields = {"ssl_dim": 768, "gin_channels": 768, "filter_channels": 768}
    if use_tiny:
        fields["filter_channels"] = 512
    return fields


def inspect_config_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
    model = config.get("model") or {}
    encoder = str(model.get("speech_encoder") or "vec768l12")
    use_tiny = infer_use_tiny_from_config(config)
    derived = resolve_architecture_fields(encoder, use_tiny=use_tiny)
    return {
        "main_preset_id": infer_preset_id_from_encoder(encoder),
        "speech_encoder": encoder,
        "use_tiny": use_tiny,
        "ssl_dim": int(model.get("ssl_dim") or derived["ssl_dim"]),
        "gin_channels": int(model.get("gin_channels") or derived["gin_channels"]),
        "filter_channels": int(model.get("filter_channels") or derived["filter_channels"]),
    }


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def base_model_paths_for_preset(preset_id: str, *, use_tiny: bool = False) -> Dict[str, Path]:
    if use_tiny:
        if preset_id != "balanced":
            raise FileNotFoundError("当前便携包仅为均衡套餐提供 tiny 主模型底模。")
        root = RUNTIME_DIR / "pre_trained_model" / "tiny" / "vec768l12_vol_emb"
    elif preset_id == "balanced":
        root = RUNTIME_DIR / "pre_trained_model" / "768l12"
    elif preset_id == "high_quality":
        root = RUNTIME_DIR / "pre_trained_model" / "whisper-ppg"
    elif preset_id == "light":
        root = RUNTIME_DIR / "pre_trained_model" / "hubertsoft"
    else:
        raise FileNotFoundError(f"未知主模型套餐：{preset_id}")
    generator_path = root / "G_0.pth"
    discriminator_path = root / "D_0.pth"
    if not _exists(generator_path) or not _exists(discriminator_path):
        raise FileNotFoundError(f"当前便携包缺少 {preset_id} 套餐的主模型底模。")
    return {"generator_path": generator_path, "discriminator_path": discriminator_path}


def diffusion_base_model_for_preset(preset_id: str) -> Path:
    if preset_id == "balanced":
        candidate = RUNTIME_DIR / "pre_trained_model" / "diffusion" / "768l12" / "model_0.pt"
    elif preset_id == "high_quality":
        candidate = RUNTIME_DIR / "pre_trained_model" / "diffusion" / "whisper-ppg" / "model_0.pt"
    elif preset_id == "light":
        candidate = RUNTIME_DIR / "pre_trained_model" / "diffusion" / "hubertsoft" / "model_0.pt"
    else:
        raise FileNotFoundError(f"未知扩散套餐：{preset_id}")
    if not _exists(candidate):
        raise FileNotFoundError(f"当前便携包缺少 {preset_id} 套餐的扩散底模。")
    return candidate


def encoder_asset_available(encoder: str) -> bool:
    normalized = (encoder or "").strip()
    if normalized in {"vec768l12", "vec256l9"}:
        return _exists(RUNTIME_DIR / "pretrain" / "checkpoint_best_legacy_500.pt")
    if normalized == "hubertsoft":
        return _exists(RUNTIME_DIR / "pretrain" / "hubert-soft-0d54a1f4.pt")
    if normalized == "whisper-ppg":
        return _exists(RUNTIME_DIR / "pretrain" / "medium.pt")
    return False


def _tiny_support(preset: TrainingPresetDefinition) -> Dict[str, Any]:
    if preset.preset_id != "balanced":
        return {"available": False, "reason_disabled": "当前便携包仅为均衡套餐提供 tiny 主模型底模。"}
    candidate = RUNTIME_DIR / "pre_trained_model" / "tiny" / "vec768l12_vol_emb" / "G_0.pth"
    if _exists(candidate):
        return {"available": True, "reason_disabled": None}
    return {"available": False, "reason_disabled": "当前便携包缺少均衡套餐的 tiny 底模。"}


def build_training_presets() -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for preset in PRESET_DEFINITIONS:
        encoder_ok = encoder_asset_available(preset.encoder)
        main_ok = False
        try:
            base_model_paths_for_preset(preset.preset_id, use_tiny=False)
            main_ok = True
        except FileNotFoundError:
            main_ok = False
        available = encoder_ok and main_ok
        reason = None
        if not encoder_ok:
            reason = f"缺少 {preset.encoder} 对应的编码器资产。"
        elif not main_ok:
            reason = "缺少该套餐对应的主模型底模。"
        payload.append(
            {
                "id": preset.preset_id,
                "label": preset.label,
                "encoder": preset.encoder,
                "encoder_dim": preset.encoder_dim,
                "ssl_dim": preset.ssl_dim,
                "gin_channels": preset.gin_channels,
                "filter_channels": preset.filter_channels,
                "recommended_vram_gb": preset.recommended_vram_gb,
                "description": preset.description,
                "is_default": preset.is_default,
                "available": available,
                "reason_disabled": reason,
            }
        )
    return payload


def _precision_support_for_device(device_value: str) -> Dict[str, bool]:
    normalized = (device_value or "auto").strip().lower()
    if normalized == "cpu":
        return {"fp32": True, "fp16": False, "bf16": False}
    if not torch.cuda.is_available():
        return {"fp32": True, "fp16": False, "bf16": False}
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {"fp32": True, "fp16": True, "bf16": bf16_supported}


def build_training_capabilities(devices: List[Dict[str, str]]) -> Dict[str, Any]:
    precision_support_by_device = {
        item["value"]: _precision_support_for_device(item["value"])
        for item in devices
        if item.get("value")
    }
    tiny_support_by_preset = {preset.preset_id: _tiny_support(preset) for preset in PRESET_DEFINITIONS}
    encoder_support = {preset.encoder: encoder_asset_available(preset.encoder) for preset in PRESET_DEFINITIONS}
    diffusion_support: Dict[str, Dict[str, Any]] = {}
    for preset in PRESET_DEFINITIONS:
        try:
            diffusion_base_model_for_preset(preset.preset_id)
            diffusion_support[preset.preset_id] = {"available": True, "reason_disabled": None}
        except FileNotFoundError as exc:
            diffusion_support[preset.preset_id] = {"available": False, "reason_disabled": str(exc)}
    config_template_path = RUNTIME_DIR / "configs_template" / "config_template.json"
    diffusion_template_path = RUNTIME_DIR / "configs_template" / "diffusion_template.yaml"
    main_defaults: Dict[str, Any] = {
        "main_batch_size": 6,
        "main_precision": "fp32",
        "main_all_in_mem": False,
        "use_tiny": False,
        "learning_rate": 0.0001,
        "log_interval": 200,
    }
    diffusion_defaults: Dict[str, Any] = {
        "diffusion_mode": "disabled",
        "diff_batch_size": 48,
        "diff_amp_dtype": "fp32",
        "diff_cache_all_data": True,
        "diff_cache_device": "cpu",
        "diff_num_workers": 4,
    }
    if config_template_path.exists():
        try:
            with config_template_path.open("r", encoding="utf-8") as handle:
                config_template = json.load(handle)
            train_defaults = config_template.get("train") or {}
            main_defaults.update(
                {
                    "main_batch_size": int(train_defaults.get("batch_size") or 6),
                    "main_precision": "fp32"
                    if not train_defaults.get("fp16_run")
                    else ("bf16" if str(train_defaults.get("half_type") or "fp16").lower() == "bf16" else "fp16"),
                    "main_all_in_mem": bool(train_defaults.get("all_in_mem", False)),
                    "learning_rate": float(train_defaults.get("learning_rate") or 0.0001),
                    "log_interval": int(train_defaults.get("log_interval") or 200),
                }
            )
        except Exception:
            pass
    if diffusion_template_path.exists():
        try:
            with diffusion_template_path.open("r", encoding="utf-8") as handle:
                diffusion_template = yaml.safe_load(handle) or {}
            train_defaults = diffusion_template.get("train") or {}
            diffusion_defaults.update(
                {
                    "diff_batch_size": int(train_defaults.get("batch_size") or 48),
                    "diff_amp_dtype": str(train_defaults.get("amp_dtype") or "fp32"),
                    "diff_cache_all_data": bool(train_defaults.get("cache_all_data", True)),
                    "diff_cache_device": str(train_defaults.get("cache_device") or "cpu"),
                    "diff_num_workers": int(train_defaults.get("num_workers") or 4),
                }
            )
        except Exception:
            pass
    return {
        "precision_support_by_device": precision_support_by_device,
        "tiny_support_by_preset": tiny_support_by_preset,
        "encoder_asset_support": encoder_support,
        "diffusion_asset_support": diffusion_support,
        "main_defaults": main_defaults,
        "diffusion_defaults": diffusion_defaults,
    }

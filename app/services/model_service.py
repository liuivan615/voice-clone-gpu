from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from app.config import RUNTIME_DIR, SETTINGS, ensure_runtime_environment, resolve_runtime_path, runtime_workdir
from app.schemas import RuntimeProfile, model_to_dict
from app.services.gpu_compat_service import available_device_options, inspect_gpu_compatibility
from app.services.library_service import LibraryService
from app.services.preprocess_service import PreprocessService
from app.services.training_preset_service import build_training_capabilities, build_training_presets

F0_PREDICTORS = [
    {"value": "rmvpe", "label": "RMVPE", "tier": "recommended", "description": "推荐默认。质量最好，对歌声和带伴奏更稳，但资源占用更高。"},
    {"value": "fcpe", "label": "FCPE", "tier": "fast", "description": "现代快速备选，速度更快，质量通常接近 RMVPE。"},
    {"value": "crepe", "label": "CREPE", "tier": "legacy", "description": "旧兼容选项，效果可用但一般不作为首选。"},
    {"value": "harvest", "label": "Harvest", "tier": "light", "description": "轻量传统算法，适合兼容和快速实验。"},
    {"value": "dio", "label": "DIO", "tier": "light", "description": "轻量传统算法，速度快，但复杂音频稳定性不如 RMVPE。"},
    {"value": "pm", "label": "Parselmouth", "tier": "light", "description": "轻量兼容选项，适合基础场景。"},
]


@dataclass
class LoadedModelState:
    profile_key: str
    profile: RuntimeProfile
    model: Any
    speakers: List[str]
    device_used: str


class ModelService:
    def __init__(self, library_service: LibraryService) -> None:
        self.library_service = library_service
        self._lock = threading.RLock()
        self._state = None  # type: Optional[LoadedModelState]

    def _scan_files(self, root: Path, patterns: List[str], **kwargs) -> List[str]:
        exclude_prefixes = kwargs.get("exclude_prefixes", tuple())
        results = set()  # type: Set[str]
        if not root.exists():
            return []
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and not path.name.startswith(exclude_prefixes):
                    results.add(path.relative_to(root).as_posix())
        return sorted(results)

    def _config_speakers(self, config_path: Optional[Path]) -> List[str]:
        if not config_path or not config_path.exists() or config_path.suffix.lower() != ".json":
            return []
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []
        spk = config_data.get("spk", {})
        if isinstance(spk, dict):
            return list(spk.keys())
        return []

    def _runtime_profiles(self) -> List[Dict[str, Any]]:
        profiles = []
        runtime_models = self._scan_files(RUNTIME_DIR, ["logs/**/*.pth", "models/**/*.pth"], exclude_prefixes=("D_",))
        for value in runtime_models:
            config_path = "configs/config.json"
            config_candidate = resolve_runtime_path(config_path)
            speakers = self._config_speakers(config_candidate)
            profiles.append(
                {
                    "id": value,
                    "label": Path(value).name,
                    "name": Path(value).stem,
                    "checkpoint_path": value,
                    "config_path": config_path,
                    "speakers": speakers,
                    "speaker": speakers[0] if speakers else "",
                    "source": "runtime",
                }
            )
        return profiles

    def _discover_profiles(self) -> List[Dict[str, Any]]:
        library_profiles = self.library_service.get_inference_profiles()
        runtime_profiles = self._runtime_profiles()
        trained_profiles = sorted(
            [item for item in library_profiles if item.get("source") != "runtime"],
            key=lambda item: item.get("created_at") or "",
            reverse=True,
        )
        return trained_profiles + sorted(runtime_profiles, key=lambda item: item.get("label") or "")

    def _find_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        profile = self.library_service.find_profile(profile_id)
        if profile:
            return profile
        for runtime_profile in self._runtime_profiles():
            if runtime_profile["id"] == profile_id or runtime_profile["checkpoint_path"] == profile_id:
                return runtime_profile
        return None

    def _normalize_profile(self, profile: Optional[RuntimeProfile]) -> RuntimeProfile:
        default_profile = RuntimeProfile(**SETTINGS["default_profile"])
        if profile is None:
            normalized = default_profile
        else:
            merged = model_to_dict(default_profile)
            merged.update(model_to_dict(profile, exclude_none=True))
            normalized = RuntimeProfile(**merged)

        profiles = self._discover_profiles()
        models = [item["checkpoint_path"] for item in profiles]
        configs = self._scan_files(RUNTIME_DIR, ["configs/*.json", "logs/**/*.json"])

        if not normalized.checkpoint_path and profiles:
            normalized.checkpoint_path = profiles[0]["checkpoint_path"]
            normalized.config_path = profiles[0]["config_path"]
            normalized.speaker = profiles[0].get("speaker", "") or normalized.speaker
        elif normalized.checkpoint_path:
            matched = self._find_profile(normalized.checkpoint_path)
            if matched:
                normalized.checkpoint_path = matched["checkpoint_path"]
                normalized.config_path = matched["config_path"]
                normalized.speaker = matched.get("speaker", "") or normalized.speaker
                if normalized.use_diffusion and matched.get("diffusion_model_path") and matched.get("diffusion_config_path"):
                    normalized.diffusion_model_path = matched["diffusion_model_path"]
                    normalized.diffusion_config_path = matched["diffusion_config_path"]
                elif not normalized.use_diffusion:
                    normalized.diffusion_model_path = ""
            elif not normalized.config_path and configs:
                normalized.config_path = configs[0]
        elif not normalized.config_path and configs:
            normalized.config_path = configs[0]

        speakers = self._config_speakers(resolve_runtime_path(normalized.config_path))
        if not normalized.speaker and speakers:
            normalized.speaker = speakers[0]
        return normalized

    def _profile_key(self, profile: RuntimeProfile) -> str:
        return json.dumps(model_to_dict(profile), sort_keys=True, ensure_ascii=False)

    def get_runtime_options(self) -> Dict[str, Any]:
        normalized = self._normalize_profile(None)
        compat = inspect_gpu_compatibility()
        profiles = self._discover_profiles()
        models = [item["checkpoint_path"] for item in profiles]
        model_library = self.library_service.list_models()
        devices = available_device_options()
        cluster_models = self._scan_files(
            RUNTIME_DIR,
            ["logs/**/*.pt", "logs/**/*.pkl", "models/**/*.pt", "models/**/*.pkl"],
        )
        diffusion_models = sorted(
            value
            for value in self._scan_files(RUNTIME_DIR, ["logs/**/*.pt", "models/**/*.pt"])
            if "diffusion" in value.lower()
        )
        speakers = self._config_speakers(resolve_runtime_path(normalized.config_path))
        loaded_profile = self._state.profile if self._state else None
        loaded_device = self._state.device_used if self._state else None

        current_profile = None
        if self._state:
            current_loaded = self._find_profile(self._state.profile.checkpoint_path)
            current_profile = {
                "id": current_loaded.get("id", self._state.profile.checkpoint_path) if current_loaded else self._state.profile.checkpoint_path,
                "label": current_loaded.get("label") if current_loaded else (Path(self._state.profile.checkpoint_path).name or "Loaded profile"),
                "speakers": self._state.speakers,
                **model_to_dict(self._state.profile),
            }
        default_match = self._find_profile(normalized.checkpoint_path) if normalized.checkpoint_path else None

        return {
            "models": models,
            "profiles": profiles,
            "model_library": model_library,
            "configs": self._scan_files(RUNTIME_DIR, ["configs/*.json", "logs/**/*.json"]),
            "cluster_models": cluster_models,
            "diffusion_models": diffusion_models,
            "devices": devices,
            "f0_predictors": F0_PREDICTORS,
            "speakers": speakers if not self._state else self._state.speakers,
            "default_profile": normalized,
            "defaults": {"profile_id": default_match["id"] if default_match else normalized.checkpoint_path, **model_to_dict(normalized)},
            "loaded_profile": loaded_profile,
            "current_profile": current_profile,
            "loaded_device": loaded_device,
            "gpu_compatibility": compat,
            "gpu_compat": compat,
            "runtime_root": str(RUNTIME_DIR),
            "upload_limit_mb": SETTINGS["app"]["upload_limit_mb"],
            "media_upload_limit_mb": SETTINGS["app"].get("media_upload_limit_mb", 0),
            "training_upload_limit_mb": SETTINGS["app"].get("training_upload_limit_mb", 0),
            "data_root": str(self.library_service.data_root),
            "training_defaults": SETTINGS["training_defaults"],
            "training_presets": build_training_presets(),
            "training_capabilities": build_training_capabilities(devices),
            "preprocess_supported": True,
            "accepted_media_types": PreprocessService.accepted_media_types(),
            "separator_engines": PreprocessService().separator_engines(),
        }

    def register_model(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path,
        label: str,
        speaker: str,
        source: str,
    ) -> Dict[str, Any]:
        return self.library_service.import_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            label=label,
            speaker=speaker,
            source=source,
        )

    def unload(self) -> None:
        with self._lock:
            if not self._state:
                return
            with runtime_workdir():
                try:
                    self._state.model.unload_model()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            self._state = None

    def load_profile(self, profile: Optional[RuntimeProfile]) -> Dict[str, Any]:
        normalized = self._normalize_profile(profile)
        profile_key = self._profile_key(normalized)
        compat = inspect_gpu_compatibility()

        if normalized.device_preference != "cpu" and compat["block_gpu_inference"]:
            raise RuntimeError("当前 GPU 环境不满足项目要求，已阻断 GPU 推理。请修复 CUDA 12.8+ 或驱动后再试。")

        with self._lock:
            if self._state and self._state.profile_key == profile_key:
                return self.snapshot()

            self.unload()

            checkpoint_path = resolve_runtime_path(normalized.checkpoint_path)
            config_path = resolve_runtime_path(normalized.config_path)
            cluster_path = resolve_runtime_path(normalized.cluster_path)
            diffusion_model_path = resolve_runtime_path(normalized.diffusion_model_path)
            diffusion_config_path = resolve_runtime_path(normalized.diffusion_config_path)

            if not checkpoint_path or not checkpoint_path.exists():
                raise FileNotFoundError("未找到可用模型 checkpoint，请先运行 bootstrap 并提供推理模型。")
            if not config_path or not config_path.exists():
                raise FileNotFoundError("未找到可用配置文件 config.json。")

            ensure_runtime_environment()
            model = None
            try:
                with runtime_workdir():
                    from inference.infer_tool import Svc

                    device_preference = normalized.device_preference
                    svc_device = None if device_preference == "auto" else device_preference
                    model = Svc(
                        str(checkpoint_path),
                        str(config_path),
                        device=svc_device,
                        cluster_model_path=str(cluster_path) if cluster_path and cluster_path.exists() else "",
                        diffusion_model_path=str(diffusion_model_path) if diffusion_model_path and diffusion_model_path.exists() else "logs/44k/diffusion/model_0.pt",
                        diffusion_config_path=str(diffusion_config_path) if diffusion_config_path and diffusion_config_path.exists() else "configs/diffusion.yaml",
                        shallow_diffusion=bool(diffusion_model_path and diffusion_model_path.exists()),
                    )

                speakers = list(model.spk2id.keys()) if getattr(model, "spk2id", None) else []
                if not normalized.speaker and speakers:
                    normalized.speaker = speakers[0]
                if normalized.speaker and speakers and normalized.speaker not in speakers:
                    raise RuntimeError(f"说话人 {normalized.speaker} 不在可用 speaker 列表中。")
            except Exception:
                if model is not None:
                    with runtime_workdir():
                        try:
                            model.unload_model()
                        except Exception:
                            pass
                    torch.cuda.empty_cache()
                raise

            self._state = LoadedModelState(
                profile_key=profile_key,
                profile=normalized,
                model=model,
                speakers=speakers,
                device_used=str(getattr(model, "dev", "cpu")),
            )
            return self.snapshot()

    def get_or_load(self, profile: Optional[RuntimeProfile]) -> LoadedModelState:
        self.load_profile(profile)
        assert self._state is not None
        return self._state

    def snapshot(self) -> Dict[str, Any]:
        options = self.get_runtime_options()
        return {
            "profile": options["loaded_profile"] or options["default_profile"],
            "speakers": options["speakers"],
            "device_used": options["loaded_device"],
            "gpu_compatibility": options["gpu_compatibility"],
        }

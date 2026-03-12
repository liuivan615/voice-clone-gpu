from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from app.config import RUNTIME_DIR, SETTINGS, runtime_workdir
from app.services.gpu_compat_service import inspect_gpu_compatibility
from app.services.model_service import ModelService
from app.services.training_preset_service import (
    base_model_paths_for_preset,
    diffusion_base_model_for_preset,
    encoder_asset_available,
    get_training_preset,
    resolve_architecture_fields,
)


ProgressCallback = Callable[[str, float, str, int, int], None]


class TrainingService:
    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service
        self.library_service = model_service.library_service

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _write_yaml(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)

    def _copy_if_exists(self, source_path: Path, target_path: Path) -> None:
        if not source_path.exists():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    def _copy_tree(self, source_dir: Path, target_dir: Path) -> None:
        if not source_dir.exists():
            return
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)

    def _encoder_pretrain_assets(self, encoder: str) -> List[Path]:
        normalized = (encoder or "").strip()
        if normalized in {"vec768l12", "vec256l9"}:
            return [RUNTIME_DIR / "pretrain" / "checkpoint_best_legacy_500.pt"]
        if normalized == "hubertsoft":
            return [RUNTIME_DIR / "pretrain" / "hubert-soft-0d54a1f4.pt"]
        if normalized == "whisper-ppg":
            return [RUNTIME_DIR / "pretrain" / "medium.pt"]
        return []

    def _validate_precision(self, precision: str, device_name: str, *, context: str = "主训练") -> None:
        normalized = (precision or "fp32").strip().lower()
        if normalized not in {"fp32", "fp16", "bf16"}:
            raise RuntimeError(f"{context}不支持的训练精度：{precision}")
        if device_name == "cpu" and normalized != "fp32":
            raise RuntimeError(f"{context}在 CPU 下当前仅支持 fp32。")
        if device_name.startswith("cuda") and normalized == "bf16":
            if not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                raise RuntimeError(f"{context}当前 GPU 不支持 bf16。")

    def _copy_uploaded_files(self, source_paths: List[Path], target_dir: Path) -> List[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for source_path in source_paths:
            destination = target_dir / source_path.name
            shutil.copy2(source_path, destination)
            copied.append(destination)
        return copied

    def _write_segment(self, output_path: Path, samples: np.ndarray, sample_rate: int) -> Path | None:
        normalized = np.asarray(samples, dtype=np.float32)
        if normalized.ndim > 1:
            normalized = librosa.to_mono(normalized.T)
        duration = float(normalized.shape[0] / sample_rate) if sample_rate else 0.0
        if duration < 1.5:
            return None
        peak = float(np.abs(normalized).max()) if normalized.size else 0.0
        if peak < 1e-4:
            return None
        sf.write(str(output_path), normalized, sample_rate)
        return output_path

    def _chunk_long_segment(self, samples: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        max_window_samples = int(sample_rate * 8.0)
        minimum_tail_samples = int(sample_rate * 2.0)
        if samples.shape[0] <= max_window_samples:
            return [samples]

        chunks = []
        cursor = 0
        while cursor < samples.shape[0]:
            chunk = samples[cursor:cursor + max_window_samples]
            if chunk.shape[0] < minimum_tail_samples and chunks:
                break
            chunks.append(chunk)
            cursor += max_window_samples
        return chunks

    def _auto_slice_training_sources(self, raw_dir: Path) -> List[Path]:
        wav_files = sorted(raw_dir.glob("*.wav"))
        if len(wav_files) >= 3:
            return wav_files

        runtime_str = str(RUNTIME_DIR)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)

        generated_segments = []
        with runtime_workdir():
            from inference import slicer

            for wav_path in wav_files:
                chunks = slicer.cut(str(wav_path), db_thresh=-40, min_len=1800)
                audio_data, audio_sr = slicer.chunks2audio(str(wav_path), chunks)
                segment_index = 0
                for slice_tag, data in audio_data:
                    if slice_tag:
                        continue
                    for chunk in self._chunk_long_segment(np.asarray(data, dtype=np.float32), audio_sr):
                        segment_index += 1
                        output_path = raw_dir / f"{wav_path.stem}_seg{segment_index:03d}.wav"
                        written = self._write_segment(output_path, chunk, audio_sr)
                        if written:
                            generated_segments.append(written)

        if len(generated_segments) >= 3:
            originals_dir = raw_dir / "_originals"
            originals_dir.mkdir(parents=True, exist_ok=True)
            for wav_path in wav_files:
                shutil.move(str(wav_path), str(originals_dir / wav_path.name))
            return sorted(generated_segments)

        for segment_path in generated_segments:
            segment_path.unlink(missing_ok=True)
        return wav_files

    def _resample_dataset(self, raw_dir: Path, dataset_dir: Path, sample_rate: int) -> List[Path]:
        generated = []
        for speaker_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
            output_dir = dataset_dir / speaker_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            for wav_path in sorted(speaker_dir.glob("*.wav")):
                wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
                wav, _ = librosa.effects.trim(wav, top_db=40)
                peak = np.abs(wav).max() or 1.0
                wav = 0.98 * wav / peak
                resampled = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
                output_path = output_dir / wav_path.name
                sf.write(str(output_path), resampled, sample_rate)
                generated.append(output_path)
        return generated

    def _write_filelists(self, dataset_dir: Path, filelists_dir: Path) -> tuple[List[Path], List[Path]]:
        speaker_dir = next(path for path in dataset_dir.iterdir() if path.is_dir())
        wavs = sorted(speaker_dir.glob("*.wav"))
        if len(wavs) < 3:
            raise RuntimeError("自动切分后仍不足 3 条有效训练片段。请补充更长或更多干净语音。")
        val = wavs[:2]
        train = wavs[2:]
        filelists_dir.mkdir(parents=True, exist_ok=True)
        (filelists_dir / "train.txt").write_text(
            "\n".join(f"./dataset/44k/{speaker_dir.name}/{path.name}" for path in train) + "\n",
            encoding="utf-8",
        )
        (filelists_dir / "val.txt").write_text(
            "\n".join(f"./dataset/44k/{speaker_dir.name}/{path.name}" for path in val) + "\n",
            encoding="utf-8",
        )
        return train, val

    def _write_config(self, workspace: Path, speaker: str, batch_size: int) -> None:
        config_path = RUNTIME_DIR / "configs" / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config["spk"] = {speaker: 0}
        config["model"]["n_speakers"] = 1
        config["model"]["speech_encoder"] = "vec768l12"
        config["model"]["ssl_dim"] = 768
        config["model"]["gin_channels"] = 768
        config["model"]["filter_channels"] = 768
        config["train"]["log_interval"] = 1
        config["train"]["eval_interval"] = 1
        config["train"]["epochs"] = 1
        config["train"]["batch_size"] = batch_size
        config["train"]["keep_ckpts"] = 1
        config["train"]["all_in_mem"] = False
        config["train"]["port"] = "8011"
        target = workspace / "configs" / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        shutil.copy2(RUNTIME_DIR / "configs" / "diffusion.yaml", workspace / "configs" / "diffusion.yaml")

    def _copy_training_assets(self, workspace: Path) -> None:
        logs_dir = workspace / "logs" / "44k"
        logs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNTIME_DIR / "pre_trained_model" / "768l12" / "G_0.pth", logs_dir / "G_0.pth")
        shutil.copy2(RUNTIME_DIR / "pre_trained_model" / "768l12" / "D_0.pth", logs_dir / "D_0.pth")
        pretrain_dir = workspace / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNTIME_DIR / "pretrain" / "checkpoint_best_legacy_500.pt", pretrain_dir / "checkpoint_best_legacy_500.pt")
        if (RUNTIME_DIR / "pretrain" / "rmvpe.pt").exists():
            shutil.copy2(RUNTIME_DIR / "pretrain" / "rmvpe.pt", pretrain_dir / "rmvpe.pt")
        if (RUNTIME_DIR / "pretrain" / "fcpe.pt").exists():
            shutil.copy2(RUNTIME_DIR / "pretrain" / "fcpe.pt", pretrain_dir / "fcpe.pt")

    def _compute_features(self, workspace: Path, f0_predictor_name: str, progress: ProgressCallback) -> None:
        os.environ.setdefault("FAIRSEQ_SKIP_HYDRA_INIT", "1")
        runtime_str = str(RUNTIME_DIR)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)

        import utils
        from modules.mel_processing import spectrogram_torch

        previous_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            sample_rate = 44100
            hop_length = 512
            device = torch.device("cpu")
            encoder = utils.get_speech_encoder("vec768l12", device=device)
            f0_predictor = utils.get_f0_predictor(
                f0_predictor_name,
                sampling_rate=sample_rate,
                hop_length=hop_length,
                device=None,
                threshold=0.05,
            )
            wavs = sorted((workspace / "dataset" / "44k").glob("*/*.wav"))
            total = len(wavs)
            for index, wav_path in enumerate(wavs, start=1):
                progress("featurizing", 0.4 + (index / max(total, 1)) * 0.25, f"正在生成特征 {index}/{total}...", index, total)
                wav, _ = librosa.load(str(wav_path), sr=sample_rate)
                audio_norm = torch.FloatTensor(wav).unsqueeze(0)
                wav16k = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
                wav16k = torch.from_numpy(wav16k).to(device)
                content = encoder.encoder(wav16k)
                torch.save(content.cpu(), str(wav_path) + ".soft.pt")
                f0, uv = f0_predictor.compute_f0_uv(wav)
                np.save(str(wav_path) + ".f0.npy", np.asanyarray((f0, uv), dtype=object))
                spec = spectrogram_torch(audio_norm, 2048, sample_rate, hop_length, 2048, center=False)
                torch.save(torch.squeeze(spec, 0), str(wav_path).replace(".wav", ".spec.pt"))
        finally:
            os.chdir(previous_cwd)

    def _run_training_steps(
        self,
        workspace: Path,
        device_name: str,
        max_steps: int,
        progress: ProgressCallback,
    ) -> Dict[str, Any]:
        runtime_str = str(RUNTIME_DIR)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)

        import modules.commons as commons
        import utils
        from data_utils import TextAudioCollate, TextAudioSpeakerLoader
        from models import MultiPeriodDiscriminator, SynthesizerTrn
        from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
        from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

        previous_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            device = torch.device(device_name)
            hps = utils.get_hparams_from_file(str(workspace / "configs" / "config.json"))
            hps.model_dir = str(workspace / "logs" / "44k")
            torch.manual_seed(hps.train.seed)

            train_dataset = TextAudioSpeakerLoader(str(workspace / "filelists" / "train.txt"), hps, all_in_mem=bool(hps.train.all_in_mem))
            train_loader = DataLoader(
                train_dataset,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
                batch_size=hps.train.batch_size,
                collate_fn=TextAudioCollate(),
            )
            train_iterator = iter(train_loader)

            net_g = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
            ).to(device)
            net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
            optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
            optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

            utils.load_checkpoint(str(workspace / "logs" / "44k" / "G_0.pth"), net_g, optim_g, skip_optimizer=True)
            utils.load_checkpoint(str(workspace / "logs" / "44k" / "D_0.pth"), net_d, optim_d, skip_optimizer=True)

            final_metrics = {}
            for step in range(1, max_steps + 1):
                try:
                    items = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    items = next(train_iterator)

                progress("training", 0.68 + (step / max(max_steps, 1)) * 0.22, f"正在训练第 {step}/{max_steps} 步...", step, max_steps)
                c, f0, spec, y, spk, lengths, uv, volume = items
                g = spk.to(device)
                spec = spec.to(device)
                y = y.to(device)
                c = c.to(device)
                f0 = f0.to(device)
                uv = uv.to(device)
                lengths = lengths.to(device)
                if volume is not None:
                    volume = volume.to(device)

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(
                    c,
                    f0,
                    uv,
                    spec,
                    g=g,
                    c_lengths=lengths,
                    spec_lengths=lengths,
                    vol=volume,
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                optim_d.zero_grad()
                loss_disc.backward()
                optim_d.step()

                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.use_automatic_f0_prediction else 0
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                optim_g.zero_grad()
                loss_gen_all.backward()
                optim_g.step()

                final_metrics = {
                    "loss_disc": float(loss_disc.detach().cpu()),
                    "loss_gen": float(loss_gen_all.detach().cpu()),
                }

            checkpoint_step = max_steps
            g_path = workspace / "logs" / "44k" / f"G_{checkpoint_step}.pth"
            d_path = workspace / "logs" / "44k" / f"D_{checkpoint_step}.pth"
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, checkpoint_step, str(g_path))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, checkpoint_step, str(d_path))
            torch.cuda.empty_cache()
            final_metrics["checkpoint_step"] = checkpoint_step
            final_metrics["g_path"] = str(g_path)
            final_metrics["d_path"] = str(d_path)
            return final_metrics
        finally:
            os.chdir(previous_cwd)

    def _resolve_training_device(self, device_preference: str) -> str:
        normalized = (device_preference or "auto").strip().lower()
        compat = inspect_gpu_compatibility()

        if normalized == "cpu":
            return "cpu"

        if normalized != "auto" and not normalized.startswith("cuda"):
            raise RuntimeError(f"不支持的训练设备选项：{device_preference}")

        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前未检测到 CUDA，无法按所选设备进行训练。")

        if compat["block_gpu_inference"]:
            raise RuntimeError("当前 GPU 环境不满足项目要求，已阻断 GPU 训练。请修复 CUDA 12.8+ 或驱动后再试。")

        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return normalized

    def _prepare_dataset_version_workspace(self, workspace: Path, dataset_version_id: str, speaker_name: str) -> Dict[str, Any]:
        dataset_version = self.library_service.get_dataset_version(dataset_version_id)
        source_root = workspace / "dataset_source" / speaker_name
        source_root.mkdir(parents=True, exist_ok=True)
        copied_files = []
        for index, segment in enumerate(dataset_version["segments"], start=1):
            source_path = self.library_service.absolute_path(segment["storage_path"])
            destination = source_root / f"{index:04d}_{segment['display_name']}"
            shutil.copy2(source_path, destination)
            copied_files.append(destination)
        return {"dataset_version": dataset_version, "source_root": source_root.parent, "copied_files": copied_files}

    def _build_resume_context(self, resume_from_checkpoint_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not resume_from_checkpoint_id:
            return None
        checkpoint = self.library_service.get_checkpoint(resume_from_checkpoint_id)
        if not checkpoint.get("discriminator_path"):
            raise RuntimeError("当前选择的 checkpoint 不包含判别器状态，无法继续训练。请改用本地训练生成的 checkpoint。")
        version = self.library_service.get_model_version(checkpoint["model_version_id"])
        model = self.library_service.get_model(version["model_id"])
        return {
            "checkpoint": checkpoint,
            "version": version,
            "model": model,
            "generator_path": self.library_service.absolute_path(checkpoint["generator_path"]),
            "discriminator_path": self.library_service.absolute_path(checkpoint["discriminator_path"]),
            "config_path": self.library_service.absolute_path(version["config_path"]),
            "diffusion_config_path": self.library_service.absolute_path(version["diffusion_config_path"]) if version.get("diffusion_config_path") else None,
        }

    def _resolve_diffusion_only_target(self, version_id: str) -> Dict[str, Any]:
        if not version_id:
            raise RuntimeError("diffusion_only 需要目标模型版本。")
        version = self.library_service.get_model_version(version_id)
        dataset_version_id = version.get("dataset_version_id")
        if not dataset_version_id:
            raise RuntimeError("关联数据集版本不可用，请重新选择数据集后完全重训。")
        dataset_version = self.library_service.get_dataset_version(dataset_version_id)
        if not dataset_version.get("segments"):
            raise RuntimeError("关联数据集版本不可用，请重新选择数据集后完全重训。")
        model = self.library_service.get_model(version["model_id"])
        return {
            "version": version,
            "dataset_version": dataset_version,
            "model": model,
        }

    def _main_runtime_overrides(
        self,
        *,
        config: Dict[str, Any],
        speaker: str,
        encoder: str,
        use_tiny: bool,
        batch_size: int,
        precision: str,
        keep_ckpts: int,
        all_in_mem: bool,
        learning_rate: float,
        log_interval: int,
    ) -> Dict[str, Any]:
        arch = resolve_architecture_fields(encoder, use_tiny=use_tiny)
        config["spk"] = {speaker: 0}
        config.setdefault("model", {})
        config["model"]["n_speakers"] = 1
        config["model"]["speech_encoder"] = encoder
        config["model"]["ssl_dim"] = arch["ssl_dim"]
        config["model"]["gin_channels"] = arch["gin_channels"]
        config["model"]["filter_channels"] = arch["filter_channels"]
        config.setdefault("train", {})
        config["train"]["batch_size"] = max(1, int(batch_size))
        config["train"]["fp16_run"] = precision != "fp32"
        config["train"]["half_type"] = "bf16" if precision == "bf16" else "fp16"
        config["train"]["learning_rate"] = float(learning_rate)
        config["train"]["log_interval"] = max(1, int(log_interval))
        config["train"]["keep_ckpts"] = max(1, int(keep_ckpts))
        config["train"]["all_in_mem"] = bool(all_in_mem)
        config["train"]["port"] = "8011"
        config.setdefault("data", {})
        config["data"]["training_files"] = "filelists/train.txt"
        config["data"]["validation_files"] = "filelists/val.txt"
        return arch

    def _write_runtime_training_configs(
        self,
        *,
        workspace: Path,
        speaker: str,
        preset_id: str,
        encoder: str,
        use_tiny: bool,
        batch_size: int,
        precision: str,
        keep_ckpts: int,
        all_in_mem: bool,
        learning_rate: float,
        log_interval: int,
        diff_batch_size: int,
        diff_amp_dtype: str,
        diff_cache_all_data: bool,
        diff_cache_device: str,
        diff_num_workers: int,
        training_device: str,
        base_config_path: Optional[Path] = None,
        base_diffusion_config_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if base_config_path:
            config = self._load_json(base_config_path)
        else:
            template_name = "config_tiny_template.json" if use_tiny else "config_template.json"
            config = self._load_json(RUNTIME_DIR / "configs_template" / template_name)
        arch = self._main_runtime_overrides(
            config=config,
            speaker=speaker,
            encoder=encoder,
            use_tiny=use_tiny,
            batch_size=batch_size,
            precision=precision,
            keep_ckpts=keep_ckpts,
            all_in_mem=all_in_mem,
            learning_rate=learning_rate,
            log_interval=log_interval,
        )
        config_path = workspace / "configs" / "config.json"
        self._write_json(config_path, config)

        diffusion_template_path = base_diffusion_config_path if base_diffusion_config_path and base_diffusion_config_path.exists() else RUNTIME_DIR / "configs_template" / "diffusion_template.yaml"
        diffusion_config = self._load_yaml(diffusion_template_path)
        diffusion_config.setdefault("data", {})
        diffusion_config.setdefault("model", {})
        diffusion_config.setdefault("train", {})
        diffusion_config.setdefault("env", {})
        diffusion_config["data"]["training_files"] = "filelists/train.txt"
        diffusion_config["data"]["validation_files"] = "filelists/val.txt"
        diffusion_config["data"]["encoder"] = encoder
        diffusion_config["data"]["encoder_out_channels"] = get_training_preset(preset_id).encoder_dim
        diffusion_config["data"]["unit_interpolate_mode"] = config["data"].get("unit_interpolate_mode", "nearest")
        diffusion_config["model"]["n_spk"] = 1
        diffusion_config["spk"] = {speaker: 0}
        diffusion_config["device"] = training_device
        diffusion_config["env"]["expdir"] = "logs/44k/diffusion"
        diffusion_config["train"]["batch_size"] = max(1, int(diff_batch_size))
        diffusion_config["train"]["amp_dtype"] = diff_amp_dtype
        diffusion_config["train"]["cache_all_data"] = bool(diff_cache_all_data)
        diffusion_config["train"]["cache_device"] = diff_cache_device
        diffusion_config["train"]["cache_fp16"] = diff_amp_dtype != "fp32"
        diffusion_config["train"]["num_workers"] = max(0, int(diff_num_workers))
        diffusion_config.setdefault("vocoder", {})
        diffusion_config["vocoder"]["ckpt"] = "pretrain/nsf_hifigan/model"
        diffusion_config_path = workspace / "configs" / "diffusion.yaml"
        self._write_yaml(diffusion_config_path, diffusion_config)
        return {
            "config_path": config_path,
            "diffusion_config_path": diffusion_config_path,
            "architecture": arch,
        }

    def _copy_runtime_training_assets(self, workspace: Path, *, preset_id: str, encoder: str, use_tiny: bool) -> None:
        logs_dir = workspace / "logs" / "44k"
        logs_dir.mkdir(parents=True, exist_ok=True)
        base_paths = base_model_paths_for_preset(preset_id, use_tiny=use_tiny)
        shutil.copy2(base_paths["generator_path"], logs_dir / "G_0.pth")
        shutil.copy2(base_paths["discriminator_path"], logs_dir / "D_0.pth")
        pretrain_dir = workspace / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        for asset_path in self._encoder_pretrain_assets(encoder):
            self._copy_if_exists(asset_path, pretrain_dir / asset_path.name)
        self._copy_if_exists(RUNTIME_DIR / "pretrain" / "rmvpe.pt", pretrain_dir / "rmvpe.pt")
        self._copy_if_exists(RUNTIME_DIR / "pretrain" / "fcpe.pt", pretrain_dir / "fcpe.pt")
        self._copy_tree(RUNTIME_DIR / "pretrain" / "nsf_hifigan", pretrain_dir / "nsf_hifigan")

    def _compute_features_gpu_preferred(
        self,
        workspace: Path,
        f0_predictor_name: str,
        feature_device_name: str,
        encoder: str,
        use_diffusion: bool,
        progress: ProgressCallback,
    ) -> None:
        command = [
            sys.executable,
            str(RUNTIME_DIR / "preprocess_hubert_f0.py"),
            "--in_dir",
            "dataset/44k",
            "--device",
            feature_device_name,
            "--f0_predictor",
            f0_predictor_name,
            "--num_processes",
            "1",
        ]
        if use_diffusion:
            command.append("--use_diff")
        progress("featurizing", 0.4, f"正在为 {encoder} 生成训练特征{' 与扩散 mel' if use_diffusion else ''}...", 0, 0)
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONPATH": str(RUNTIME_DIR), "FAIRSEQ_SKIP_HYDRA_INIT": "1"},
        )
        last_line = ""
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            last_line = line
            if any(token in line.lower() for token in ("using", "loaded", "diff", "encoder", "extractor")):
                progress("featurizing", 0.52, line, 0, 0)
        if process.wait() != 0:
            raise RuntimeError(last_line or "训练特征生成失败。")
        progress("featurizing", 0.66, "训练特征生成完成。", 0, 0)

    def _write_training_config(self, workspace: Path, speaker: str, batch_size: int, keep_ckpts: int, base_config_path: Optional[Path] = None) -> None:
        config_path = base_config_path or (RUNTIME_DIR / "configs" / "config.json")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config["spk"] = {speaker: 0}
        config["model"]["n_speakers"] = 1
        config["model"]["speech_encoder"] = "vec768l12"
        config["model"]["ssl_dim"] = 768
        config["model"]["gin_channels"] = 768
        config["model"]["filter_channels"] = 768
        config["train"]["log_interval"] = 1
        config["train"]["eval_interval"] = 1
        config["train"]["epochs"] = 1
        config["train"]["batch_size"] = batch_size
        config["train"]["keep_ckpts"] = keep_ckpts
        config["train"]["all_in_mem"] = False
        config["train"]["port"] = "8011"
        target = workspace / "configs" / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        shutil.copy2(RUNTIME_DIR / "configs" / "diffusion.yaml", workspace / "configs" / "diffusion.yaml")

    def _prune_checkpoints(self, checkpoints: List[Dict[str, Any]], keep_last: int) -> List[Dict[str, Any]]:
        if keep_last <= 0:
            keep_last = 1
        auto_checkpoints = [item for item in checkpoints if item["kind"] == "auto"]
        auto_checkpoints.sort(key=lambda item: item["step"])
        while len(auto_checkpoints) > keep_last:
            stale = auto_checkpoints.pop(0)
            Path(stale["generator_path"]).unlink(missing_ok=True)
            if stale.get("discriminator_path"):
                Path(stale["discriminator_path"]).unlink(missing_ok=True)
            checkpoints = [item for item in checkpoints if item is not stale]
        return checkpoints

    def _run_diffusion_training(
        self,
        *,
        workspace: Path,
        preset_id: str,
        device_name: str,
        max_steps: int,
        amp_dtype: str,
        progress: ProgressCallback,
        resume_model_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        runtime_str = str(RUNTIME_DIR)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)

        from diffusion.data_loaders import get_data_loaders
        from diffusion.logger.saver import Saver
        from diffusion.solver import test as diffusion_test
        from diffusion.unit2mel import DotDict, Unit2Mel
        from diffusion.vocoder import Vocoder

        previous_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            args = DotDict(self._load_yaml(workspace / "configs" / "diffusion.yaml"))
            args.device = device_name
            loader_train, loader_test = get_data_loaders(args)
            vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device_name)
            model = Unit2Mel(
                args.data.encoder_out_channels,
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max,
            ).to(device_name)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.train.lr),
                weight_decay=float(args.train.weight_decay),
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(args.train.decay_step)),
                gamma=float(args.train.gamma),
            )

            init_ckpt_path = resume_model_path if resume_model_path and resume_model_path.exists() else diffusion_base_model_for_preset(preset_id)
            checkpoint = torch.load(init_ckpt_path, map_location=torch.device(device_name))
            model.load_state_dict(checkpoint["model"])
            initial_global_step = int(checkpoint.get("global_step") or 0)
            if resume_model_path and checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])

            saver = Saver(args, initial_global_step=initial_global_step)
            if amp_dtype == "fp32":
                use_autocast = False
                cast_dtype = torch.float32
            elif amp_dtype == "bf16":
                use_autocast = device_name.startswith("cuda")
                cast_dtype = torch.bfloat16
            else:
                use_autocast = device_name.startswith("cuda")
                cast_dtype = torch.float16
            scaler = GradScaler(enabled=use_autocast and cast_dtype == torch.float16)
            train_iter = iter(loader_train)
            total_target_step = initial_global_step + max_steps
            last_loss = 0.0

            for step_offset in range(1, max_steps + 1):
                try:
                    data = next(train_iter)
                except StopIteration:
                    train_iter = iter(loader_train)
                    data = next(train_iter)
                saver.global_step_increment()
                optimizer.zero_grad()
                for key in list(data.keys()):
                    if not key.startswith("name"):
                        data[key] = data[key].to(args.device)

                current_step = initial_global_step + step_offset
                progress("training_diffusion", 0.96 + (step_offset / max(max_steps, 1)) * 0.03, f"正在训练扩散模型第 {current_step}/{total_target_step} 步（{device_name}）...", current_step, total_target_step)
                if cast_dtype == torch.float32:
                    loss = model(
                        data["units"].float(),
                        data["f0"],
                        data["volume"],
                        data["spk_id"],
                        aug_shift=data["aug_shift"],
                        gt_spec=data["mel"].float(),
                        infer=False,
                        k_step=model.k_step_max,
                    )
                else:
                    with autocast(device_type="cuda", enabled=use_autocast, dtype=cast_dtype):
                        loss = model(
                            data["units"],
                            data["f0"],
                            data["volume"],
                            data["spk_id"],
                            aug_shift=data["aug_shift"],
                            gt_spec=data["mel"],
                            infer=False,
                            k_step=model.k_step_max,
                        )
                if torch.isnan(loss):
                    raise RuntimeError("扩散训练出现 NaN loss。")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                last_loss = float(loss.detach().cpu())

                # Periodic checkpoint save + validation (mirrors solver.py interval_val pattern)
                interval_val = getattr(args.train, "interval_val", 0) or 0
                if interval_val > 0 and saver.global_step % interval_val == 0:
                    saver.save_model(model, optimizer if args.train.save_opt else None, postfix=f"{saver.global_step}")
                    interval_force = getattr(args.train, "interval_force_save", 0) or 0
                    if interval_force > 0:
                        stale_step = saver.global_step - interval_val
                        if stale_step > 0 and stale_step % interval_force != 0:
                            saver.delete_model(postfix=f"{stale_step}")
                    # Run validation test set
                    if loader_test is not None:
                        test_loss = diffusion_test(args, model, vocoder, loader_test, saver)
                        saver.log_info(f" --- <validation> --- \nloss: {test_loss:.3f}. ")
                        saver.log_value({"validation/loss": test_loss})
                        model.train()

            saver.save_model(model, optimizer if args.train.save_opt else None, postfix=f"{saver.global_step}")
            final_model_path = workspace / str(args.env.expdir) / f"model_{saver.global_step}.pt"
            return {
                "diffusion_model_path": final_model_path,
                "diffusion_config_path": workspace / "configs" / "diffusion.yaml",
                "diffusion_step_count": saver.global_step,
                "loss_diffusion": last_loss,
            }
        finally:
            os.chdir(previous_cwd)

    def _run_training_steps_v2(
        self,
        workspace: Path,
        device_name: str,
        max_steps: int,
        checkpoint_interval_steps: int,
        checkpoint_keep_last: int,
        resume_context: Optional[Dict[str, Any]],
        progress: ProgressCallback,
    ) -> Dict[str, Any]:
        runtime_str = str(RUNTIME_DIR)
        if runtime_str not in sys.path:
            sys.path.insert(0, runtime_str)

        import modules.commons as commons
        import utils
        from data_utils import TextAudioCollate, TextAudioSpeakerLoader
        from models import MultiPeriodDiscriminator, SynthesizerTrn
        from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
        from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

        previous_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            device = torch.device(device_name)
            hps = utils.get_hparams_from_file(str(workspace / "configs" / "config.json"))
            hps.model_dir = str(workspace / "logs" / "44k")
            torch.manual_seed(hps.train.seed)

            train_dataset = TextAudioSpeakerLoader(str(workspace / "filelists" / "train.txt"), hps, all_in_mem=bool(hps.train.all_in_mem))
            train_loader = DataLoader(
                train_dataset,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
                batch_size=hps.train.batch_size,
                collate_fn=TextAudioCollate(),
            )
            train_iterator = iter(train_loader)

            net_g = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
            ).to(device)
            net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
            optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
            optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

            start_step = 0
            if resume_context:
                _, optim_g, _, start_step = utils.load_checkpoint(str(resume_context["generator_path"]), net_g, optim_g, skip_optimizer=False)
                _, optim_d, _, disc_step = utils.load_checkpoint(str(resume_context["discriminator_path"]), net_d, optim_d, skip_optimizer=False)
                start_step = max(start_step, disc_step)
            else:
                utils.load_checkpoint(str(workspace / "logs" / "44k" / "G_0.pth"), net_g, optim_g, skip_optimizer=True)
                utils.load_checkpoint(str(workspace / "logs" / "44k" / "D_0.pth"), net_d, optim_d, skip_optimizer=True)

            use_autocast = bool(device.type == "cuda" and hps.train.fp16_run)
            half_dtype = torch.bfloat16 if str(hps.train.half_type).lower() == "bf16" else torch.float16
            scaler = GradScaler(enabled=use_autocast and half_dtype == torch.float16)
            final_metrics = {}
            checkpoints: List[Dict[str, Any]] = []
            total_target_step = start_step + max_steps
            for step_offset in range(1, max_steps + 1):
                try:
                    items = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    items = next(train_iterator)

                current_step = start_step + step_offset
                progress("training_main", 0.68 + (step_offset / max(max_steps, 1)) * 0.2, f"正在训练主模型第 {current_step}/{total_target_step} 步（{device_name}）...", current_step, total_target_step)
                c, f0, spec, y, spk, lengths, uv, volume = items
                g = spk.to(device)
                spec = spec.to(device)
                y = y.to(device)
                c = c.to(device)
                f0 = f0.to(device)
                uv = uv.to(device)
                lengths = lengths.to(device)
                if volume is not None:
                    volume = volume.to(device)

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                with autocast(device_type="cuda", enabled=use_autocast, dtype=half_dtype):
                    y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(
                        c,
                        f0,
                        uv,
                        spec,
                        g=g,
                        c_lengths=lengths,
                        spec_lengths=lengths,
                        vol=volume,
                    )
                    y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                    y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                optim_d.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(loss_disc).backward()
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    optim_d.step()

                with autocast(device_type="cuda", enabled=use_autocast, dtype=half_dtype):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.use_automatic_f0_prediction else 0
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
                optim_g.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(loss_gen_all).backward()
                    scaler.step(optim_g)
                    scaler.update()
                else:
                    loss_gen_all.backward()
                    optim_g.step()

                final_metrics = {
                    "loss_disc": float(loss_disc.detach().cpu()),
                    "loss_gen": float(loss_gen_all.detach().cpu()),
                }
                if checkpoint_interval_steps > 0 and current_step < total_target_step and current_step % checkpoint_interval_steps == 0:
                    progress("saving_main", 0.9, f"正在保存主模型 checkpoint {current_step}...", current_step, total_target_step)
                    g_path = workspace / "logs" / "44k" / f"G_{current_step}.pth"
                    d_path = workspace / "logs" / "44k" / f"D_{current_step}.pth"
                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, current_step, str(g_path))
                    utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, current_step, str(d_path))
                    checkpoints.append({"step": current_step, "generator_path": str(g_path), "discriminator_path": str(d_path), "kind": "auto", "is_final": False})
                    checkpoints = self._prune_checkpoints(checkpoints, checkpoint_keep_last)

            final_step = total_target_step
            progress("saving_main", 0.93, f"正在保存主模型最终 checkpoint {final_step}...", final_step, total_target_step)
            g_path = workspace / "logs" / "44k" / f"G_{final_step}.pth"
            d_path = workspace / "logs" / "44k" / f"D_{final_step}.pth"
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, final_step, str(g_path))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, final_step, str(d_path))
            checkpoints.append({"step": final_step, "generator_path": str(g_path), "discriminator_path": str(d_path), "kind": "final", "is_final": True})
            torch.cuda.empty_cache()
            final_metrics["checkpoint_step"] = final_step
            final_metrics["g_path"] = str(g_path)
            final_metrics["d_path"] = str(d_path)
            final_metrics["checkpoints"] = checkpoints
            return final_metrics
        finally:
            os.chdir(previous_cwd)

    def run(
        self,
        *,
        task_id: str,
        dataset_version_id: str,
        task_dir: Path,
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
        progress: ProgressCallback,
    ) -> Dict[str, Any]:
        workspace = task_dir / "training_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        dataset_dir = workspace / "dataset" / "44k"
        filelists_dir = workspace / "filelists"

        training_mode = (training_mode or "new").strip().lower()
        diffusion_mode = (diffusion_mode or "disabled").strip().lower()
        if training_mode not in {"new", "resume", "diffusion_only"}:
            raise RuntimeError(f"不支持的训练模式：{training_mode}")
        if diffusion_mode not in {"disabled", "chain", "deferred"}:
            raise RuntimeError(f"不支持的扩散训练方式：{diffusion_mode}")

        progress("preparing", 0.04, "正在准备训练工作区...", 0, 0)
        training_device = self._resolve_training_device(device_preference)
        feature_device = training_device if training_device.startswith("cuda") else "cpu"
        self._validate_precision(main_precision, training_device, context="主训练")
        if training_mode == "diffusion_only" or diffusion_mode != "disabled":
            self._validate_precision(diff_amp_dtype, training_device, context="扩散训练")

        resume_context = self._build_resume_context(resume_from_checkpoint_id if training_mode == "resume" else None)
        diffusion_target = self._resolve_diffusion_only_target(target_model_version_id or "") if training_mode == "diffusion_only" else None
        active_dataset_version_id = diffusion_target["version"]["dataset_version_id"] if diffusion_target else dataset_version_id
        dataset_version = self.library_service.get_dataset_version(active_dataset_version_id)
        speaker_name = dataset_version["speaker"]
        if training_mode == "resume":
            if not resume_context:
                raise RuntimeError("继续训练需要 resume checkpoint。")
            main_preset_id = resume_context["version"].get("main_preset_id") or main_preset_id or "balanced"
            use_tiny = bool(resume_context["version"].get("use_tiny"))
            model_name = resume_context["model"]["name"]
        elif training_mode == "diffusion_only":
            if not diffusion_target:
                raise RuntimeError("diffusion_only 需要目标模型版本。")
            main_preset_id = diffusion_target["version"].get("main_preset_id") or main_preset_id or "balanced"
            use_tiny = bool(diffusion_target["version"].get("use_tiny"))
            model_name = diffusion_target["model"]["name"]

        preset = get_training_preset(main_preset_id)
        encoder = preset.encoder
        if training_mode == "resume" and resume_context:
            encoder = resume_context["version"].get("speech_encoder") or encoder
        elif training_mode == "diffusion_only" and diffusion_target:
            encoder = diffusion_target["version"].get("speech_encoder") or encoder
        if not encoder_asset_available(encoder):
            raise RuntimeError(f"当前便携包缺少 {encoder} 对应的编码器资产。")
        base_model_paths_for_preset(main_preset_id, use_tiny=use_tiny)
        if diffusion_mode != "disabled" or training_mode == "diffusion_only":
            diffusion_base_model_for_preset(main_preset_id)
        normalized_diff_cache_device = (diff_cache_device or "cpu").strip().lower()
        if normalized_diff_cache_device == "cuda":
            normalized_diff_cache_device = training_device
        if normalized_diff_cache_device.startswith("cuda") and not training_device.startswith("cuda"):
            raise RuntimeError("CPU 训练不支持将扩散缓存放入 CUDA。")

        prepared = self._prepare_dataset_version_workspace(workspace, active_dataset_version_id, speaker_name)
        progress("preparing", 0.12, f"已装载数据集版本 {dataset_version['label']}，共 {len(prepared['copied_files'])} 条片段。", len(prepared["copied_files"]), len(prepared["copied_files"]))
        generated = self._resample_dataset(prepared["source_root"], dataset_dir, 44100)
        progress("preparing", 0.24, f"重采样完成，共 {len(generated)} 条。", len(generated), len(generated))
        train_files, val_files = self._write_filelists(dataset_dir, filelists_dir)
        progress("preparing", 0.32, f"训练列表已生成，train={len(train_files)} val={len(val_files)}。", len(train_files), len(train_files) + len(val_files))
        config_paths = self._write_runtime_training_configs(
            workspace=workspace,
            speaker=speaker_name,
            preset_id=main_preset_id,
            encoder=encoder,
            use_tiny=use_tiny,
            batch_size=main_batch_size,
            precision=main_precision,
            keep_ckpts=checkpoint_keep_last,
            all_in_mem=main_all_in_mem,
            learning_rate=learning_rate,
            log_interval=log_interval,
            diff_batch_size=diff_batch_size,
            diff_amp_dtype=diff_amp_dtype,
            diff_cache_all_data=diff_cache_all_data,
            diff_cache_device=normalized_diff_cache_device,
            diff_num_workers=diff_num_workers,
            training_device=training_device,
            base_config_path=Path(resume_context["config_path"]) if resume_context else None,
            base_diffusion_config_path=Path(resume_context["diffusion_config_path"]) if resume_context and resume_context.get("diffusion_config_path") else None,
        )
        self._copy_runtime_training_assets(workspace, preset_id=main_preset_id, encoder=encoder, use_tiny=use_tiny)
        progress("preparing", 0.38, "训练配置与底模已写入工作区。", len(train_files), len(train_files) + len(val_files))

        use_diffusion_features = training_mode == "diffusion_only" or diffusion_mode in {"chain", "deferred"}
        self._compute_features_gpu_preferred(
            workspace,
            f0_predictor,
            feature_device,
            encoder,
            use_diffusion_features,
            progress,
        )

        metrics: Dict[str, Any] = {
            "loss_disc": None,
            "loss_gen": None,
            "checkpoint_step": 0,
            "checkpoints": [],
            "g_path": "",
        }
        version = diffusion_target["version"] if diffusion_target else None
        if training_mode != "diffusion_only":
            metrics = self._run_training_steps_v2(
                workspace,
                training_device,
                max_steps,
                checkpoint_interval_steps,
                checkpoint_keep_last,
                resume_context,
                progress,
            )
            progress("saving_main", 0.95, "正在注册主模型到模型库...", metrics["checkpoint_step"], metrics["checkpoint_step"])
            version = self.library_service.register_trained_version(
                model_name=model_name,
                speaker=speaker_name,
                dataset_version_id=active_dataset_version_id,
                training_mode=training_mode,
                f0_predictor=f0_predictor,
                device_preference=device_preference,
                device_used=training_device,
                config_path=config_paths["config_path"],
                checkpoints=metrics["checkpoints"],
                final_generator_path=Path(metrics["g_path"]),
                step_count=int(metrics["checkpoint_step"]),
                loss_disc=metrics["loss_disc"],
                loss_gen=metrics["loss_gen"],
                parent_version_id=resume_context["version"]["id"] if resume_context else None,
                parent_checkpoint_id=resume_context["checkpoint"]["id"] if resume_context else None,
                main_preset_id=main_preset_id,
                speech_encoder=encoder,
                use_tiny=use_tiny,
                ssl_dim=config_paths["architecture"]["ssl_dim"],
                gin_channels=config_paths["architecture"]["gin_channels"],
                filter_channels=config_paths["architecture"]["filter_channels"],
                diffusion_status="training" if diffusion_mode == "chain" else "not_trained",
                diffusion_params={
                    "diffusion_mode": diffusion_mode,
                    "diff_batch_size": diff_batch_size,
                    "diff_amp_dtype": diff_amp_dtype,
                    "diff_cache_all_data": diff_cache_all_data,
                    "diff_cache_device": normalized_diff_cache_device,
                    "diff_num_workers": diff_num_workers,
                },
            )
        elif version is None:
            raise RuntimeError("未找到扩散训练目标版本。")
        if "checkpoints" not in version:
            version["checkpoints"] = self.library_service.list_checkpoints(version["id"])

        diffusion_summary = {
            "diffusion_status": version.get("diffusion_status"),
            "diffusion_model_path": version.get("diffusion_model_path"),
            "diffusion_config_path": version.get("diffusion_config_path"),
            "diffusion_step_count": version.get("diffusion_step_count"),
        }
        run_diffusion_now = training_mode == "diffusion_only" or diffusion_mode == "chain"
        if run_diffusion_now:
            previous_state = {
                "diffusion_status": version.get("diffusion_status") or "not_trained",
                "diffusion_model_path": version.get("diffusion_model_path"),
                "diffusion_config_path": version.get("diffusion_config_path"),
                "diffusion_step_count": version.get("diffusion_step_count"),
                "diffusion_params": version.get("diffusion_params"),
            }
            self.library_service.update_model_version_diffusion(
                version["id"],
                diffusion_status="training",
                diffusion_params={
                    "diffusion_mode": "chain" if training_mode == "diffusion_only" else diffusion_mode,
                    "diff_batch_size": diff_batch_size,
                    "diff_amp_dtype": diff_amp_dtype,
                    "diff_cache_all_data": diff_cache_all_data,
                    "diff_cache_device": normalized_diff_cache_device,
                    "diff_num_workers": diff_num_workers,
                },
            )
            progress("preparing_diffusion", 0.96, "正在准备扩散训练...", 0, 0)
            try:
                diff_metrics = self._run_diffusion_training(
                    workspace=workspace,
                    preset_id=main_preset_id,
                    device_name=training_device,
                    max_steps=max_steps,
                    amp_dtype=diff_amp_dtype,
                    progress=progress,
                    resume_model_path=self.library_service.absolute_path(version["diffusion_model_path"]) if training_mode == "diffusion_only" and version.get("diffusion_model_path") else None,
                )
                version = self.library_service.update_model_version_diffusion(
                    version["id"],
                    diffusion_status="trained",
                    diffusion_model_path=diff_metrics["diffusion_model_path"],
                    diffusion_config_path=diff_metrics["diffusion_config_path"],
                    diffusion_step_count=int(diff_metrics["diffusion_step_count"]),
                    diffusion_params={
                        "diffusion_mode": "chain" if training_mode == "diffusion_only" else diffusion_mode,
                        "diff_batch_size": diff_batch_size,
                        "diff_amp_dtype": diff_amp_dtype,
                        "diff_cache_all_data": diff_cache_all_data,
                        "diff_cache_device": normalized_diff_cache_device,
                        "diff_num_workers": diff_num_workers,
                    },
                )
                diffusion_summary = {
                    "diffusion_status": version.get("diffusion_status"),
                    "diffusion_model_path": version.get("diffusion_model_path"),
                    "diffusion_config_path": version.get("diffusion_config_path"),
                    "diffusion_step_count": version.get("diffusion_step_count"),
                }
            except Exception:
                self.library_service.update_model_version_diffusion(
                    version["id"],
                    diffusion_status=previous_state["diffusion_status"],
                    diffusion_model_path=self.library_service.absolute_path(previous_state["diffusion_model_path"]) if previous_state.get("diffusion_model_path") else None,
                    diffusion_config_path=self.library_service.absolute_path(previous_state["diffusion_config_path"]) if previous_state.get("diffusion_config_path") else None,
                    diffusion_step_count=previous_state.get("diffusion_step_count"),
                    diffusion_params=previous_state.get("diffusion_params"),
                )
                raise
        elif training_mode != "diffusion_only" and diffusion_mode == "deferred":
            version = self.library_service.update_model_version_diffusion(
                version["id"],
                diffusion_status="not_trained",
                diffusion_params={
                    "diffusion_mode": diffusion_mode,
                    "diff_batch_size": diff_batch_size,
                    "diff_amp_dtype": diff_amp_dtype,
                    "diff_cache_all_data": diff_cache_all_data,
                    "diff_cache_device": normalized_diff_cache_device,
                    "diff_num_workers": diff_num_workers,
                },
            )
            diffusion_summary = {
                "diffusion_status": version.get("diffusion_status"),
                "diffusion_model_path": version.get("diffusion_model_path"),
                "diffusion_config_path": version.get("diffusion_config_path"),
                "diffusion_step_count": version.get("diffusion_step_count"),
            }
        if "checkpoints" not in version:
            version["checkpoints"] = self.library_service.list_checkpoints(version["id"])

        device_index = None
        if training_device.startswith("cuda") and torch.cuda.is_available():
            device_index = torch.device(training_device).index
            if device_index is None:
                device_index = 0

        summary = {
            "task_id": task_id,
            "source_file": f"dataset version {dataset_version['label']}",
            "slice_count": len(train_files),
            "sample_count": len(dataset_version["segments"]),
            "prepared_sample_count": len(dataset_version["segments"]),
            "min_duration": 0.0,
            "max_duration": 0.0,
            "total_duration": 0.0,
            "device_used": training_device,
            "speaker": speaker_name,
            "result_file": Path(metrics["g_path"]).name if metrics.get("g_path") else None,
            "latest_checkpoint": Path(metrics["g_path"]).name if metrics.get("g_path") else None,
            "status": "completed",
            "error": None,
            "gpu_name": torch.cuda.get_device_name(device_index) if device_index is not None else None,
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(device_index))) if device_index is not None else None,
            "torch_cuda_version": torch.version.cuda,
            "train_count": len(train_files),
            "val_count": len(val_files),
            "workspace": str(workspace),
            "dataset_version_id": active_dataset_version_id,
            "training_mode": training_mode,
            "resume_from_checkpoint_id": resume_context["checkpoint"]["id"] if resume_context else None,
            "feature_device_used": feature_device,
            "registered_profile_id": version["id"],
            "registered_label": version["label"],
            "registered_model": model_name,
            "model_id": version["model_id"],
            "model_version_id": version["id"],
            "config_file": version["config_path"],
            "latest_checkpoint_id": version["checkpoints"][0]["id"] if version["checkpoints"] else None,
            "loss_disc": metrics["loss_disc"],
            "loss_gen": metrics["loss_gen"],
            "main_preset_id": version.get("main_preset_id"),
            "speech_encoder": version.get("speech_encoder"),
            "use_tiny": version.get("use_tiny"),
            "ssl_dim": version.get("ssl_dim"),
            "gin_channels": version.get("gin_channels"),
            "filter_channels": version.get("filter_channels"),
            **diffusion_summary,
        }
        with (task_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return summary

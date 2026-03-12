from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def copy_sample_wavs(source_dir: Path, target_dir: Path, limit: int) -> list[Path]:
    wavs = sorted(source_dir.glob("*.wav"))[:limit]
    if not wavs:
        raise RuntimeError(f"No wav files found in {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for wav in wavs:
        destination = target_dir / wav.name
        shutil.copy2(wav, destination)
        copied.append(destination)
    return copied


def resample_dataset(raw_dir: Path, dataset_dir: Path, sample_rate: int) -> list[Path]:
    speaker_dirs = [path for path in raw_dir.iterdir() if path.is_dir()]
    generated: list[Path] = []
    for speaker_dir in speaker_dirs:
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


def write_filelists(dataset_dir: Path, filelists_dir: Path) -> tuple[list[Path], list[Path]]:
    speaker_dir = next(path for path in dataset_dir.iterdir() if path.is_dir())
    wavs = sorted(speaker_dir.glob("*.wav"))
    if len(wavs) < 3:
        raise RuntimeError("Need at least 3 wavs for smoke training.")
    val = wavs[:2]
    train = wavs[2:]
    filelists_dir.mkdir(parents=True, exist_ok=True)
    (filelists_dir / "train.txt").write_text("\n".join(f"./dataset/44k/{speaker_dir.name}/{path.name}" for path in train) + "\n", encoding="utf-8")
    (filelists_dir / "val.txt").write_text("\n".join(f"./dataset/44k/{speaker_dir.name}/{path.name}" for path in val) + "\n", encoding="utf-8")
    return train, val


def write_config(runtime_root: Path, workspace: Path, speaker: str) -> None:
    config_path = runtime_root / "configs" / "config.json"
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
    config["train"]["batch_size"] = 2
    config["train"]["keep_ckpts"] = 1
    config["train"]["all_in_mem"] = False
    config["train"]["port"] = "8011"
    target = workspace / "configs" / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    diffusion_source = runtime_root / "configs" / "diffusion.yaml"
    diffusion_target = workspace / "configs" / "diffusion.yaml"
    shutil.copy2(diffusion_source, diffusion_target)


def copy_base_checkpoints(runtime_root: Path, workspace: Path) -> None:
    logs_dir = workspace / "logs" / "44k"
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime_root / "pre_trained_model" / "768l12" / "G_0.pth", logs_dir / "G_0.pth")
    shutil.copy2(runtime_root / "pre_trained_model" / "768l12" / "D_0.pth", logs_dir / "D_0.pth")


def compute_features(runtime_root: Path, workspace: Path, f0_predictor_name: str) -> None:
    import os
    import sys

    os.environ.setdefault("FAIRSEQ_SKIP_HYDRA_INIT", "1")
    runtime_str = str(runtime_root)
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

        for wav_path in sorted((workspace / "dataset" / "44k").glob("*/*.wav")):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--speaker", default="ivan")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--f0-predictor", default="dio")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    runtime_root = Path(args.runtime_root)
    workspace = Path(args.workspace)

    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "dataset_raw").mkdir(parents=True, exist_ok=True)
    (workspace / "dataset").mkdir(parents=True, exist_ok=True)
    (workspace / "filelists").mkdir(parents=True, exist_ok=True)

    copied = copy_sample_wavs(source_dir, workspace / "dataset_raw" / args.speaker, args.limit)
    generated = resample_dataset(workspace / "dataset_raw", workspace / "dataset" / "44k", args.sample_rate)
    train, val = write_filelists(workspace / "dataset" / "44k", workspace / "filelists")
    write_config(runtime_root, workspace, args.speaker)
    copy_base_checkpoints(runtime_root, workspace)
    shutil.copytree(runtime_root / "pretrain", workspace / "pretrain")
    compute_features(runtime_root, workspace, args.f0_predictor)

    summary = {
        "copied": [path.name for path in copied],
        "resampled_count": len(generated),
        "train_count": len(train),
        "val_count": len(val),
        "workspace": str(workspace),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

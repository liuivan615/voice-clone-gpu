from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    workspace = Path(args.workspace)
    os.environ.setdefault("FAIRSEQ_SKIP_HYDRA_INIT", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.chdir(workspace)

    import sys

    runtime_str = str(runtime_root)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)

    import modules.commons as commons
    import utils
    from data_utils import TextAudioCollate, TextAudioSpeakerLoader
    from models import MultiPeriodDiscriminator, SynthesizerTrn
    from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

    device = torch.device(args.device)
    hps = utils.get_hparams_from_file(str(workspace / "configs" / "config.json"))
    hps.model_dir = str(workspace / "logs" / "44k")
    torch.manual_seed(hps.train.seed)

    train_dataset = TextAudioSpeakerLoader(str(workspace / "filelists" / "train.txt"), hps, all_in_mem=False)
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        batch_size=hps.train.batch_size,
        collate_fn=TextAudioCollate(),
    )

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

    items = next(iter(train_loader))
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

    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, 1, str(workspace / "logs" / "44k" / "G_1.pth"))
    utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, 1, str(workspace / "logs" / "44k" / "D_1.pth"))
    torch.cuda.empty_cache()
    print(
        {
            "loss_disc": float(loss_disc.detach().cpu()),
            "loss_gen": float(loss_gen_all.detach().cpu()),
            "g1_exists": (workspace / "logs" / "44k" / "G_1.pth").exists(),
            "d1_exists": (workspace / "logs" / "44k" / "D_1.pth").exists(),
        }
    )


if __name__ == "__main__":
    main()

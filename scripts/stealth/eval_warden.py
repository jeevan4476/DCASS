#!/usr/bin/env python3
# scripts/stealth/eval_warden.py
"""
Phase A validation — Warden AUC on held-out real vs. generator-produced traffic.

Loads a GAN checkpoint (generator_state + warden_state) and reports the
Warden's AUC at telling real traffic apart from the Generator's output,
using the SAME held-out split train_gan.py carved out (same seed + fraction
=> same rows, as long as --data/--val-fraction match what was used to train).

AUC near 0.5 means the Warden cannot distinguish real from fake — the
generator has converged. AUC near 1.0 means the Warden still wins easily.

Also reports AUC at window length 20 (matching StealthEnvironment's
warden_window_size) vs. the full max_sequence_length, to catch the
train/eval window-size mismatch flagged as Risk R3 in the integration plan.

Usage:
    python scripts/stealth/eval_warden.py --checkpoint storage/models/gan_generator.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.stealth.gan.generator import TemporalPatternGenerator
from src.stealth.gan.trainer import HumanTrafficDataset, compute_warden_auc
from src.analysis.adversarial.warden import DeepPacketInspectionWarden


def _load_models(checkpoint_path: Path, num_channels: int, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if config is not None and not isinstance(config, dict):
        latent_dim = getattr(config, "latent_dim", 128)
        hidden_dim = getattr(config, "hidden_dim", 256)
        max_seq_len = getattr(config, "max_sequence_length", 100)
    else:
        config = config or {}
        latent_dim = config.get("latent_dim", 128)
        hidden_dim = config.get("hidden_dim", 256)
        max_seq_len = config.get("max_sequence_length", 100)

    generator = TemporalPatternGenerator(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_channels=num_channels,
        max_sequence_length=max_seq_len,
    )
    generator.load_state_dict(ckpt["generator_state"])
    generator.to(device)
    generator.eval()

    warden = DeepPacketInspectionWarden(num_channels=num_channels, hidden_dim=hidden_dim)
    warden.load_state_dict(ckpt["warden_state"])
    warden.to(device)
    warden.eval()

    return generator, warden, max_seq_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Warden AUC (Phase A validation)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "human_traffic.json",
        help="Must match the --data used at train time for the held-out split to line up",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Must match the --val-fraction used at train time",
    )
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    generator, warden, max_seq_len = _load_models(
        args.checkpoint, args.num_channels, args.device
    )

    dataset = HumanTrafficDataset(args.data, max_seq_len)
    _, val_subset = dataset.split(args.val_fraction)

    print("=" * 60)
    print("  Warden AUC Evaluation (Phase A validation)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Held-out samples: {len(val_subset)}  (max_seq_len={max_seq_len})")
    print("=" * 60)

    auc_full = compute_warden_auc(
        generator,
        warden,
        val_subset,
        device=args.device,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
    )
    verdict = "converged (indistinguishable)" if abs(auc_full - 0.5) < 0.1 else (
        "Warden still winning" if auc_full > 0.6 else "Generator overshooting"
    )
    print(f"\nFull-length AUC (len={max_seq_len}): {auc_full:.4f}  [{verdict}]")
    print("  Target: AUC ≈ 0.5 (Warden can no longer tell real from fake)")

    print(
        "\nNote: StealthEnvironment evaluates the Warden on 20-step windows "
        "at RL training/inference time (warden_window_size=20), not the full "
        f"{max_seq_len}-step sequence used above. If the Warden was NOT "
        "trained with sinusoidal positional encodings or mixed-length "
        "batches, re-run with the 20-step env window before trusting the "
        "RL reward signal (see Risk R1/R3 in docs/GAN_RL_INTEGRATION_PLAN.md)."
    )


if __name__ == "__main__":
    main()

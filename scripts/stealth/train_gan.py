#!/usr/bin/env python3
# scripts/stealth/train_gan.py
"""
Phase A — GAN pretraining CLI.

Thin wrapper around `src.stealth.gan.trainer.GANTrainer` / `train_gan()`.
Trains the Generator + Warden (WGAN-GP) adversarial pair and writes a
checkpoint directly to the path `StealthScheduler` expects by default
(`storage/models/gan_generator.pt`), so no manual copying is needed.

See docs/GAN_RL_INTEGRATION_PLAN.md (Phase A) for the full rationale.

Usage:
    python scripts/stealth/train_gan.py \\
        --data data/human_traffic.json \\
        --epochs 20 --batch-size 32 --val-fraction 0.1 \\
        --out storage/models/gan_generator.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.stealth.gan.trainer import GANTrainer, HumanTrafficDataset, TrainingConfig
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the DCASS WGAN-GP traffic-mimicry GAN (Phase A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "human_traffic.json",
        help="Path to real human-traffic JSON (synthetic fallback if missing)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the dataset held out from training for Phase D's "
        "Warden-AUC evaluation (see scripts/stealth/eval_warden.py)",
    )
    parser.add_argument(
        "--no-gradient-penalty",
        action="store_true",
        help="Disable WGAN-GP's gradient penalty (not recommended — training "
        "is unstable without it; kept as an escape hatch)",
    )
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--warden-lr", type=float, default=2e-4)
    parser.add_argument("--warden-steps", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "gan",
        help="Directory for per-epoch snapshots",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
        help="Final checkpoint path (this is what StealthScheduler loads by default)",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_channels=args.num_channels,
        max_sequence_length=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        generator_lr=args.generator_lr,
        warden_lr=args.warden_lr,
        warden_steps=args.warden_steps,
        use_gradient_penalty=not args.no_gradient_penalty,
        lambda_gp=args.lambda_gp,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("=" * 60)
    print("  DCASS GAN Pretraining (Phase A)")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  Device: {args.device}")
    print(f"  Gradient penalty: {config.use_gradient_penalty} (lambda={config.lambda_gp})")
    print("=" * 60)

    dataset = HumanTrafficDataset(args.data, config.max_sequence_length)

    if args.val_fraction > 0:
        train_subset, val_subset = dataset.split(args.val_fraction)
        print(
            f"Dataset split: {len(train_subset)} train / {len(val_subset)} held-out "
            f"({args.val_fraction:.0%}) — held-out samples are NEVER used for training."
        )
        train_loader = DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )

    trainer = GANTrainer(config)
    trainer.train(train_loader)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "generator_state": trainer.generator.state_dict(),
        "warden_state": trainer.warden.state_dict(),
        "generator_optimizer": trainer.generator_optimizer.state_dict(),
        "warden_optimizer": trainer.warden_optimizer.state_dict(),
        "config": trainer.config,
        "metrics_history": trainer.metrics_history,
    }
    torch.save(checkpoint, args.out)
    print(f"\n✓ Final checkpoint written to {args.out}")
    print("  Next: python scripts/stealth/eval_warden.py --checkpoint " f"{args.out}")


if __name__ == "__main__":
    main()

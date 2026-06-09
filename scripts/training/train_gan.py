#!/usr/bin/env python3
"""
Train the GAN Stealth Scheduler.

Trains the TemporalPatternGenerator (Generator) against the
DeepPacketInspectionWarden (Discriminator) using real/synthetic
human traffic data.

Usage:
    python scripts/train_gan.py
    python scripts/train_gan.py --epochs 50 --device cuda --batch-size 32
    python scripts/train_gan.py --resume models/gan/epoch_010.pt
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.stealth.gan.generator import TemporalPatternGenerator
from src.stealth.gan.trainer import (
    GANTrainer,
    TrainingConfig,
    HumanTrafficDataset,
)
from src.analysis.adversarial.warden import DeepPacketInspectionWarden


def main():
    parser = argparse.ArgumentParser(description="Train GAN stealth scheduler")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "behavioral" / "human_traffic.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-gen", type=float, default=1e-4)
    parser.add_argument("--lr-warden", type=float, default=2e-4)
    parser.add_argument("--warden-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models" / "gan")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wgan-gp", action="store_true", help="Use WGAN-GP training")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- config ----------------------------------------------------------
    config = TrainingConfig(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3,
        max_sequence_length=50,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        generator_lr=args.lr_gen,
        warden_lr=args.lr_warden,
        warden_steps=args.warden_steps,
        use_gradient_penalty=args.wgan_gp,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=5,
    )

    # --- dataset ---------------------------------------------------------
    print(f"Loading traffic data from {args.data} ...")
    dataset = HumanTrafficDataset(args.data, max_sequence_length=config.max_sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    print(f"  {len(dataset)} sessions, {len(loader)} batches/epoch")

    # --- trainer ---------------------------------------------------------
    trainer = GANTrainer(config)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # --- train -----------------------------------------------------------
    trainer.train(loader, num_epochs=config.num_epochs)

    # --- save final ------------------------------------------------------
    trainer.save_checkpoint("final")
    print(f"\nTraining complete. Checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()

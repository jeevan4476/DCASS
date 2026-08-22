#!/usr/bin/env python3
"""
Train WGAN-GP Temporal Pattern Generator and Adversarial Warden for DCASS.

Trains the Generator to mimic human social media posting timing distributions
and saves the model checkpoint to `storage/models/gan_generator.pt`.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.gan.trainer import GANTrainer, TrainingConfig, HumanTrafficDataset

TRAFFIC_DATA = PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
MODELS_DIR = PROJECT_ROOT / "storage" / "models"
CHECKPOINT_PATH = MODELS_DIR / "gan_generator.pt"

def main():
    print("=" * 70)
    print("DCASS WGAN-GP Temporal Pattern Generator Training")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"• Hardware Acceleration Device: {device.upper()}")
    if device == "cuda":
        print(f"• Active GPU:                   {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.enabled = False  # Enable double backwards for WGAN-GP RNNs

    if not TRAFFIC_DATA.exists():
        print(f"Error: Dataset {TRAFFIC_DATA} not found. Running generator script first...")
        from scripts.stealth.generate_traffic_dataset import main as gen_main
        gen_main(2000)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3,
        max_sequence_length=100,
        batch_size=32,
        num_epochs=15,
        generator_lr=1e-4,
        warden_lr=2e-4,
        warden_steps=5,
        use_gradient_penalty=True,
        lambda_gp=10.0,
        device=device,
        checkpoint_dir=MODELS_DIR,
        log_interval=10
    )

    print(f"\nLoading human traffic dataset from {TRAFFIC_DATA}...")
    dataset = HumanTrafficDataset(TRAFFIC_DATA, max_sequence_length=config.max_sequence_length)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset Loaded: {len(dataset):,} sessions across {len(train_loader)} batches per epoch.")

    trainer = GANTrainer(config)
    print("\nStarting WGAN-GP Adversarial Training Loop...")
    trainer.train(train_loader, num_epochs=config.num_epochs)

    # Save generator checkpoint directly formatted for StealthScheduler
    checkpoint_data = {
        "epoch": config.num_epochs,
        "generator_state": trainer.generator.state_dict(),
        "warden_state": trainer.warden.state_dict(),
        "config": config
    }
    torch.save(checkpoint_data, CHECKPOINT_PATH)
    print("\n✅ WGAN-GP Training Complete!")
    print(f"   Saved Generator Checkpoint -> {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()

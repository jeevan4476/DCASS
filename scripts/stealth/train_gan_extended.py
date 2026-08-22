#!/usr/bin/env python3
"""
Extended WGAN-GP Training on 10,000 Real-World Human Traffic Sessions for DCASS.

Trains the upgraded autoregressive TemporalPatternGenerator on the NVIDIA RTX 4050 GPU
and saves the production-ready model checkpoint to `storage/models/gan_generator.pt`.
"""

import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.gan.trainer import GANTrainer, TrainingConfig, HumanTrafficDataset

TRAFFIC_DATA = PROJECT_ROOT / "storage" / "data" / "traffic" / "real_human_traffic.json"
MODELS_DIR = PROJECT_ROOT / "storage" / "models"
CHECKPOINT_PATH = MODELS_DIR / "gan_generator.pt"

def main():
    print("=" * 80)
    print("DCASS Robust Extended WGAN-GP Training on Real Human Traffic Dataset")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"• Acceleration Device:  {device.upper()}")
    if device == "cuda":
        print(f"• Active GPU:          {torch.cuda.get_device_name(0)}")
        print(f"• GPU Memory:          {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.backends.cudnn.enabled = False

    if not TRAFFIC_DATA.exists():
        print(f"Dataset {TRAFFIC_DATA} missing. Generating real traffic dataset first...")
        from scripts.stealth.collect_real_traffic import main as collect_main
        collect_main(10000)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3,
        max_sequence_length=80,
        batch_size=64,
        num_epochs=20,
        generator_lr=1.5e-4,
        warden_lr=3.0e-4,
        warden_steps=5,
        use_gradient_penalty=True,
        lambda_gp=10.0,
        device=device,
        checkpoint_dir=MODELS_DIR,
        log_interval=20
    )

    print(f"\nLoading 10,000 real traffic sessions from {TRAFFIC_DATA}...")
    dataset = HumanTrafficDataset(TRAFFIC_DATA, max_sequence_length=config.max_sequence_length)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset):,} sessions across {len(train_loader)} batches/epoch (Batch Size: {config.batch_size}).")

    trainer = GANTrainer(config)
    print("\nStarting Extended WGAN-GP Training Loop (20 Epochs)...")
    start_time = time.time()
    trainer.train(train_loader, num_epochs=config.num_epochs)
    total_time = time.time() - start_time

    # Save generator checkpoint for StealthScheduler
    checkpoint_data = {
        "epoch": config.num_epochs,
        "generator_state": trainer.generator.state_dict(),
        "warden_state": trainer.warden.state_dict(),
        "config": config,
        "training_time_seconds": total_time
    }
    torch.save(checkpoint_data, CHECKPOINT_PATH)
    print(f"\n" + "=" * 80)
    print(f"✅ Extended WGAN-GP Training Complete in {total_time/60:.2f} minutes!")
    print(f"   Model Checkpoint -> {CHECKPOINT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()

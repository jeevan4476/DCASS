# src/stealth/gan/trainer.py
"""
GAN Trainer for DCASS Stealth System.

This module implements the adversarial training loop for the Generator and Warden.
The Generator learns to produce human-like transmission patterns while the Warden
learns to detect steganographic behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import json
from datetime import datetime

from .generator import TemporalPatternGenerator, sample_latent, compute_generator_loss
from ...analysis.adversarial.warden import (
    DeepPacketInspectionWarden,
    compute_warden_loss,
    compute_gradient_penalty
)


@dataclass
class TrainingConfig:
    """
    Configuration for GAN training.

    Attributes:
        latent_dim: Dimension of generator latent space
        hidden_dim: Hidden dimension for both models
        num_channels: Number of distribution channels
        max_sequence_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        generator_lr: Generator learning rate
        warden_lr: Warden learning rate
        warden_steps: Number of Warden updates per generator update
        use_gradient_penalty: Whether to use WGAN-GP
        lambda_gp: Gradient penalty coefficient
        device: Device to train on
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Steps between logging
    """
    latent_dim: int = 128
    hidden_dim: int = 256
    num_channels: int = 3
    max_sequence_length: int = 100
    batch_size: int = 32
    num_epochs: int = 100
    generator_lr: float = 1e-4
    warden_lr: float = 2e-4
    warden_steps: int = 5
    use_gradient_penalty: bool = False
    lambda_gp: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("checkpoints/gan")
    log_interval: int = 10


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    epoch: int
    step: int
    generator_loss: float
    warden_loss: float
    real_bot_prob: float  # Warden's prediction on real data
    fake_bot_prob: float  # Warden's prediction on fake data
    generator_confidence: float
    gradient_penalty: Optional[float] = None


class HumanTrafficDataset(Dataset):
    """
    Dataset of real human traffic patterns.

    Expected data format (JSON):
    [
        {
            "delays": [5.2, 3.1, 12.4, ...],
            "channels": [0, 1, 0, 2, ...],
            "time_of_day": 14  # Hour 0-23
        },
        ...
    ]
    """

    def __init__(self, data_path: Path, max_sequence_length: int = 100):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON file with traffic data
            max_sequence_length: Maximum sequence length (truncate if longer)
        """
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length

        # Load data
        if data_path.exists():
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            # Generate synthetic data for testing
            print(f"Warning: {data_path} not found, generating synthetic data")
            self.data = self._generate_synthetic_data(1000)

    def _generate_synthetic_data(self, num_samples: int) -> list[dict]:
        """Generate synthetic human-like traffic for testing."""
        import numpy as np

        data = []
        for _ in range(num_samples):
            # Variable length sequences
            seq_len = np.random.randint(10, self.max_sequence_length)

            # Poisson-like delays with burstiness
            base_rate = np.random.uniform(5, 20)
            delays = np.random.exponential(base_rate, seq_len).tolist()

            # Random channel switching
            channels = np.random.randint(0, 3, seq_len).tolist()

            # Random time of day
            time_of_day = np.random.randint(0, 24)

            data.append({
                "delays": delays,
                "channels": channels,
                "time_of_day": time_of_day
            })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single traffic sample."""
        sample = self.data[idx]

        # Extract fields
        delays = sample["delays"][:self.max_sequence_length]
        channels = sample["channels"][:self.max_sequence_length]
        time_of_day = sample["time_of_day"]

        # Pad if necessary
        seq_len = len(delays)
        if seq_len < self.max_sequence_length:
            delays = delays + [0.0] * (self.max_sequence_length - seq_len)
            channels = channels + [0] * (self.max_sequence_length - seq_len)

        return {
            "delays": torch.tensor(delays, dtype=torch.float32),
            "channels": torch.tensor(channels, dtype=torch.long),
            "time_of_day": torch.tensor(time_of_day, dtype=torch.float32),
            "sequence_length": torch.tensor(seq_len, dtype=torch.long)
        }


class GANTrainer:
    """
    Trainer for the Generator-Warden adversarial system.

    Implements the training loop for both models with support for:
    - Standard GAN training
    - WGAN-GP training
    - Checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        generator: Optional[TemporalPatternGenerator] = None,
        warden: Optional[DeepPacketInspectionWarden] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            generator: Pre-initialized generator (created if None)
            warden: Pre-initialized warden (created if None)
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create models if not provided
        self.generator = generator or TemporalPatternGenerator(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_channels=config.num_channels,
            max_sequence_length=config.max_sequence_length
        )

        self.warden = warden or DeepPacketInspectionWarden(
            num_channels=config.num_channels,
            hidden_dim=config.hidden_dim
        )

        # Move to device
        self.generator.to(self.device)
        self.warden.to(self.device)

        # Optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr,
            betas=(0.5, 0.999)
        )

        self.warden_optimizer = optim.Adam(
            self.warden.parameters(),
            lr=config.warden_lr,
            betas=(0.5, 0.999)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.metrics_history: list[TrainingMetrics] = []

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(
        self,
        real_batch: dict[str, torch.Tensor]
    ) -> TrainingMetrics:
        """
        Perform one training step (Warden + Generator update).

        Args:
            real_batch: Batch of real human traffic data

        Returns:
            TrainingMetrics for this step
        """
        batch_size = real_batch["delays"].size(0)

        # Move data to device
        real_delays = real_batch["delays"].to(self.device)
        real_channels = real_batch["channels"].to(self.device)
        time_of_day = real_batch["time_of_day"].to(self.device)
        seq_lengths = real_batch["sequence_length"].to(self.device)

        # Use the actual sequence length (not padded)
        actual_seq_len = seq_lengths[0].item()  # Assume batch has similar lengths

        # ==================== Train Warden ====================
        for _ in range(self.config.warden_steps):
            self.warden_optimizer.zero_grad()

            # Generate fake traffic
            z = sample_latent(batch_size, self.config.latent_dim, device=self.device)
            fake_schedule = self.generator(z, actual_seq_len, time_of_day)

            # Get Warden verdicts
            real_verdict = self.warden(real_delays[:, :actual_seq_len], real_channels[:, :actual_seq_len])

            fake_delays = fake_schedule.delays
            fake_channels = fake_schedule.sample_channels()
            fake_verdict = self.warden(fake_delays, fake_channels)

            # Compute Warden loss
            warden_loss = compute_warden_loss(real_verdict, fake_verdict)

            # Add gradient penalty if using WGAN-GP
            gradient_penalty = None
            if self.config.use_gradient_penalty:
                gradient_penalty = compute_gradient_penalty(
                    self.warden,
                    real_delays[:, :actual_seq_len],
                    fake_delays.detach(),
                    real_channels[:, :actual_seq_len],
                    fake_channels.detach(),
                    lambda_gp=self.config.lambda_gp
                )
                warden_loss = warden_loss + gradient_penalty

            # Backward pass
            warden_loss.backward()
            self.warden_optimizer.step()

        # ==================== Train Generator ====================
        self.generator_optimizer.zero_grad()

        # Generate new fake traffic
        z = sample_latent(batch_size, self.config.latent_dim, device=self.device)
        fake_schedule = self.generator(z, actual_seq_len, time_of_day)

        # Get Warden's verdict on fake data
        fake_channels = fake_schedule.sample_channels()
        fake_verdict = self.warden(fake_schedule.delays, fake_channels)

        # Compute Generator loss (wants to fool Warden)
        generator_loss = compute_generator_loss(fake_verdict.bot_probability)

        # Backward pass
        generator_loss.backward()
        self.generator_optimizer.step()

        # ==================== Collect Metrics ====================
        with torch.no_grad():
            real_verdict = self.warden(real_delays[:, :actual_seq_len], real_channels[:, :actual_seq_len])
            fake_verdict = self.warden(fake_schedule.delays, fake_channels)

            metrics = TrainingMetrics(
                epoch=self.current_epoch,
                step=self.global_step,
                generator_loss=generator_loss.item(),
                warden_loss=warden_loss.item(),
                real_bot_prob=real_verdict.bot_probability.mean().item(),
                fake_bot_prob=fake_verdict.bot_probability.mean().item(),
                generator_confidence=fake_schedule.confidence.mean().item(),
                gradient_penalty=gradient_penalty.item() if gradient_penalty is not None else None
            )

        self.global_step += 1
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: Optional[int] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> list[TrainingMetrics]:
        """
        Train the GAN for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs (uses config if None)
            callback: Optional callback function called after each step

        Returns:
            List of training metrics
        """
        num_epochs = num_epochs or self.config.num_epochs

        print(f"Starting GAN training for {num_epochs} epochs on {self.device}")
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Warden params: {sum(p.numel() for p in self.warden.parameters()):,}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = []

            for batch_idx, batch in enumerate(train_loader):
                # Train step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.metrics_history.append(metrics)

                # Logging
                if batch_idx % self.config.log_interval == 0:
                    self._log_metrics(metrics, batch_idx, len(train_loader))

                # Optional callback
                if callback is not None:
                    callback(metrics)

            # Save checkpoint at end of epoch
            self.save_checkpoint(f"epoch_{epoch:03d}")

            # Epoch summary
            avg_gen_loss = sum(m.generator_loss for m in epoch_metrics) / len(epoch_metrics)
            avg_warden_loss = sum(m.warden_loss for m in epoch_metrics) / len(epoch_metrics)
            avg_fake_prob = sum(m.fake_bot_prob for m in epoch_metrics) / len(epoch_metrics)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Generator Loss: {avg_gen_loss:.4f}")
            print(f"  Warden Loss: {avg_warden_loss:.4f}")
            print(f"  Avg Fake Bot Probability: {avg_fake_prob:.4f}")
            print("-" * 60)

        return self.metrics_history

    def _log_metrics(self, metrics: TrainingMetrics, batch_idx: int, total_batches: int):
        """Log training metrics to console."""
        print(
            f"[Epoch {metrics.epoch}][{batch_idx}/{total_batches}] "
            f"G_loss: {metrics.generator_loss:.4f} | "
            f"W_loss: {metrics.warden_loss:.4f} | "
            f"Real: {metrics.real_bot_prob:.3f} | "
            f"Fake: {metrics.fake_bot_prob:.3f} | "
            f"Conf: {metrics.generator_confidence:.3f}"
        )

    def save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "generator_state": self.generator.state_dict(),
            "warden_state": self.warden.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "warden_optimizer": self.warden_optimizer.state_dict(),
            "config": self.config,
            "metrics_history": self.metrics_history
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.warden.load_state_dict(checkpoint["warden_state"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        self.warden_optimizer.load_state_dict(checkpoint["warden_optimizer"])
        self.metrics_history = checkpoint["metrics_history"]

        print(f"Checkpoint loaded from epoch {self.current_epoch}")


def train_gan(
    data_path: Path,
    config: Optional[TrainingConfig] = None,
    save_final: bool = True
) -> GANTrainer:
    """
    Convenience function to train a GAN from scratch.

    Args:
        data_path: Path to training data JSON
        config: Training configuration (uses default if None)
        save_final: Whether to save final models

    Returns:
        Trained GANTrainer instance
    """
    config = config or TrainingConfig()

    # Create dataset and dataloader
    dataset = HumanTrafficDataset(data_path, config.max_sequence_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Create trainer
    trainer = GANTrainer(config)

    # Train
    trainer.train(train_loader)

    # Save final checkpoint
    if save_final:
        trainer.save_checkpoint("final")

    return trainer


if __name__ == "__main__":
    # Quick test
    print("Testing GANTrainer...")

    config = TrainingConfig(
        batch_size=8,
        num_epochs=2,
        log_interval=1,
        device="cpu"
    )

    # Create synthetic dataset
    data_path = Path("data/synthetic_traffic.json")
    dataset = HumanTrafficDataset(data_path, max_sequence_length=20)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Create trainer
    trainer = GANTrainer(config)

    # Train for a few steps
    print("\nRunning short training test...")
    trainer.train(train_loader, num_epochs=2)

    print("\n✓ Trainer test complete!")
    print(f"✓ Trained for {trainer.global_step} steps")
    print(f"✓ Final metrics: G_loss={trainer.metrics_history[-1].generator_loss:.4f}")

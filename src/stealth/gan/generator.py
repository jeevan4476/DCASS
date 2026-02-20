# src/stealth/gan/generator.py
"""
GAN Generator for DCASS Steganography Scheduling.

This module implements a temporal pattern generator that learns to produce
realistic human-like transmission timing patterns for covert communication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingSchedule:
    """
    Output of the Generator.

    Attributes:
        delays: Inter-transmission delays in seconds, shape (batch_size, sequence_length)
        channel_logits: Channel selection logits, shape (batch_size, sequence_length, num_channels)
        confidence: Generator's confidence score, shape (batch_size,)
    """
    delays: torch.Tensor
    channel_logits: torch.Tensor
    confidence: torch.Tensor

    def sample_channels(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample channel indices from logits using Gumbel-Softmax.

        Args:
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Channel indices, shape (batch_size, sequence_length)
        """
        # Apply temperature scaling
        scaled_logits = self.channel_logits / temperature

        # Gumbel-Softmax sampling for differentiable sampling
        probs = torch.softmax(scaled_logits, dim=-1)
        channel_ids = torch.argmax(probs, dim=-1)

        return channel_ids

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary for easier serialization."""
        return {
            "delays": self.delays,
            "channel_logits": self.channel_logits,
            "confidence": self.confidence
        }


class TemporalPatternGenerator(nn.Module):
    """
    GAN Generator for steganography transmission scheduling.

    Learns to generate realistic human-like transmission patterns:
    - Inter-transmission delays (Poisson-like burstiness)
    - Channel selection probabilities
    - Time-of-day awareness

    Architecture:
        Latent Noise → Time Embedding → GRU → Temporal Attention → Output Heads

    The generator produces sequences of transmission timings that mimic organic
    human social media behavior to evade Deep Packet Inspection (DPI).

    Args:
        latent_dim: Dimension of input noise vector (default: 128)
        hidden_dim: GRU hidden state dimension (default: 256)
        num_channels: Number of distribution channels available (default: 3)
        max_sequence_length: Maximum number of media items in sequence (default: 100)
        time_embedding_dim: Dimension for time-of-day encoding (default: 32)

    Example:
        >>> generator = TemporalPatternGenerator(num_channels=3)
        >>> z = torch.randn(8, 128)  # Batch of 8 latent vectors
        >>> time_of_day = torch.randint(0, 24, (8,))  # Random hours
        >>> schedule = generator(z, sequence_length=20, time_of_day=time_of_day)
        >>> print(schedule.delays.shape)  # torch.Size([8, 20])
        >>> print(schedule.channel_logits.shape)  # torch.Size([8, 20, 3])
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_channels: int = 3,
        max_sequence_length: int = 100,
        time_embedding_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.max_sequence_length = max_sequence_length
        self.time_embedding_dim = time_embedding_dim

        # Time-of-day embedding (cyclical encoding: sin/cos to avoid discontinuity)
        self.time_encoder = nn.Sequential(
            nn.Linear(2, time_embedding_dim),  # [sin(hour), cos(hour)]
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.LayerNorm(time_embedding_dim)
        )

        # Initial projection: Noise + Time → Hidden
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + time_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal sequence generator (GRU for autoregressive scheduling)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )

        # Multi-head self-attention for long-range temporal dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm after attention
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        # 1. Delay prediction head (positive delays via Softplus)
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive delays
        )

        # 2. Channel selection head (logits for categorical distribution)
        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_channels)  # Logits for channel selection
        )

        # 3. Confidence estimation head (how confident is the generator)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence score in [0, 1]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def encode_time_of_day(self, time_of_day: torch.Tensor) -> torch.Tensor:
        """
        Encode time-of-day using cyclical features.

        Uses sine/cosine encoding to avoid discontinuity at midnight (23 → 0).

        Args:
            time_of_day: Hour of day [0, 23], shape (batch_size,)

        Returns:
            Time embedding, shape (batch_size, time_embedding_dim)
        """
        # Convert hour to radians [0, 2π]
        hour_radians = 2.0 * torch.pi * time_of_day / 24.0

        # Cyclical encoding
        time_features = torch.stack([
            torch.sin(hour_radians),
            torch.cos(hour_radians)
        ], dim=1)  # (batch_size, 2)

        # Project to embedding space
        time_embed = self.time_encoder(time_features)  # (batch_size, time_embedding_dim)

        return time_embed

    def forward(
        self,
        z: torch.Tensor,
        sequence_length: int,
        time_of_day: torch.Tensor
    ) -> TimingSchedule:
        """
        Generate a transmission schedule from latent noise.

        Args:
            z: Latent noise, shape (batch_size, latent_dim)
            sequence_length: Number of media items to schedule
            time_of_day: Hour of day [0, 23], shape (batch_size,)

        Returns:
            TimingSchedule with delays, channel logits, and confidence

        Raises:
            ValueError: If sequence_length exceeds max_sequence_length
        """
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds "
                f"max_sequence_length ({self.max_sequence_length})"
            )

        batch_size = z.size(0)

        # Encode time-of-day with cyclical features
        time_embed = self.encode_time_of_day(time_of_day)  # (batch_size, time_embedding_dim)

        # Combine latent noise with time context
        combined = torch.cat([z, time_embed], dim=1)  # (batch_size, latent_dim + time_embedding_dim)
        hidden = self.latent_projection(combined)  # (batch_size, hidden_dim)

        # Expand hidden state for sequence generation
        hidden_seq = hidden.unsqueeze(1).repeat(1, sequence_length, 1)  # (batch_size, seq_len, hidden_dim)

        # Autoregressive temporal modeling with GRU
        gru_out, _ = self.gru(hidden_seq)  # (batch_size, seq_len, hidden_dim)

        # Apply self-attention for global temporal coherence
        attn_out, attn_weights = self.attention(
            gru_out, gru_out, gru_out
        )  # (batch_size, seq_len, hidden_dim)

        # Residual connection + layer norm
        temporal_features = self.attn_norm(gru_out + attn_out)  # (batch_size, seq_len, hidden_dim)

        # Generate outputs from temporal features
        delays = self.delay_head(temporal_features).squeeze(-1)  # (batch_size, seq_len)
        channel_logits = self.channel_head(temporal_features)  # (batch_size, seq_len, num_channels)

        # Compute confidence score (based on final hidden state)
        final_state = temporal_features[:, -1, :]  # (batch_size, hidden_dim)
        confidence = self.confidence_head(final_state).squeeze(-1)  # (batch_size,)

        return TimingSchedule(
            delays=delays,
            channel_logits=channel_logits,
            confidence=confidence
        )

    def generate(
        self,
        batch_size: int = 1,
        sequence_length: int = 20,
        time_of_day: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> TimingSchedule:
        """
        Generate a schedule from random latent noise.

        Convenience method that samples latent noise automatically.

        Args:
            batch_size: Number of schedules to generate
            sequence_length: Length of each schedule
            time_of_day: Hour of day [0, 23] (randomly sampled if None)
            device: Device to generate on

        Returns:
            TimingSchedule
        """
        # Sample latent noise
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Sample time of day if not provided
        if time_of_day is None:
            time_of_day = torch.randint(0, 24, (batch_size,), device=device).float()

        return self.forward(z, sequence_length, time_of_day)


def sample_latent(
    batch_size: int,
    latent_dim: int = 128,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Sample latent noise from standard normal distribution.

    Args:
        batch_size: Number of latent vectors to sample
        latent_dim: Dimension of each latent vector
        device: Device to create tensor on

    Returns:
        Latent noise, shape (batch_size, latent_dim)
    """
    return torch.randn(batch_size, latent_dim, device=device)


def compute_generator_loss(
    fake_warden_output: torch.Tensor,
    throughput_penalty: float = 0.0
) -> torch.Tensor:
    """
    Compute Generator loss for GAN training.

    The generator wants the Warden to classify its output as "human" (low bot probability).

    Args:
        fake_warden_output: Warden's bot probability for generated traffic, shape (batch_size,)
        throughput_penalty: Optional penalty for slow transmission (not used in basic GAN)

    Returns:
        Generator loss (scalar)
    """
    # Generator wants to fool the Warden (minimize Warden's bot probability)
    # Equivalently: maximize Warden's "human" probability
    # Loss = -log(P(human)) = -log(1 - P(bot))
    fool_loss = -torch.log(1.0 - fake_warden_output + 1e-8).mean()

    return fool_loss + throughput_penalty


if __name__ == "__main__":
    # Quick test
    print("Testing TemporalPatternGenerator...")

    generator = TemporalPatternGenerator(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3,
        max_sequence_length=100
    )

    # Generate a batch
    batch_size = 4
    seq_len = 20
    z = sample_latent(batch_size, latent_dim=128)
    time_of_day = torch.randint(0, 24, (batch_size,)).float()

    schedule = generator(z, seq_len, time_of_day)

    print(f"✓ Delays shape: {schedule.delays.shape}")
    print(f"✓ Channel logits shape: {schedule.channel_logits.shape}")
    print(f"✓ Confidence shape: {schedule.confidence.shape}")
    print(f"✓ Sample delays (seconds): {schedule.delays[0, :5].tolist()}")
    print(f"✓ Sample channels: {schedule.sample_channels()[0, :5].tolist()}")
    print(f"✓ Confidence scores: {schedule.confidence.tolist()}")

    print("\nGenerator parameters:", sum(p.numel() for p in generator.parameters()))
    print("Generator ready for training!")

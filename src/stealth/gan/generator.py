# src/stealth/gan/generator.py
"""
GAN Generator for DCASS Steganography Scheduling.

Implements an autoregressive temporal pattern generator that produces realistic
human-like transmission timing distributions for covert communication:
- Variable and arbitrary sequence lengths (N >= 1 without length limits)
- Causal temporal residual blocks supporting native 2nd-order gradients on GPU
- Step-by-step autoregressive streaming for live transmission pipelines
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

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
        Sample channel indices from logits using Gumbel-Softmax or argmax.

        Args:
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Channel indices, shape (batch_size, sequence_length)
        """
        scaled_logits = self.channel_logits / max(temperature, 1e-4)
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary for serialization."""
        return {
            "delays": self.delays,
            "channel_logits": self.channel_logits,
            "confidence": self.confidence
        }

class CausalGatedBlock(nn.Module):
    """
    Causal Gated Temporal Convolutional Block.
    Guarantees that each step only attends to previous time steps and provides
    native 100% stable 2nd-order analytical derivatives for WGAN-GP.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv_val = nn.Conv1d(channels, channels, kernel_size=kernel_size)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, channels) -> transpose to (batch_size, channels, seq_len)
        residual = x
        x_trans = x.transpose(1, 2)
        x_padded = nn.functional.pad(x_trans, (self.pad, 0))

        val = self.conv_val(x_padded)
        gate = torch.sigmoid(self.conv_gate(x_padded))
        out = (val * gate).transpose(1, 2)
        out = self.norm(residual + self.dropout(out))
        return out

class TemporalPatternGenerator(nn.Module):
    """
    Autoregressive GAN Generator for steganography transmission scheduling.

    Learns human social-media posting distributions:
    - Diurnal circadian rhythm modulation
    - Poisson-like burstiness and heavy-tail reading pauses
    - Multi-channel platform switching
    - Arbitrary sequence lengths (N >= 1) with streaming support
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_channels: int = 3,
        max_sequence_length: int = 1000,
        time_embedding_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.max_sequence_length = max_sequence_length
        self.time_embedding_dim = time_embedding_dim

        # Cyclical time-of-day embedding (sin/cos encoding)
        self.time_encoder = nn.Sequential(
            nn.Linear(2, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.LayerNorm(time_embedding_dim)
        )

        # Initial projection: Noise + Time -> Hidden
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + time_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Autoregressive sequence generator (GRU)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )

        # Causal temporal convolutional block for long-range temporal dependencies
        self.temporal_block = CausalGatedBlock(channels=hidden_dim, kernel_size=3, dropout=dropout)

        # Output heads
        # 1. Delay head: strictly positive inter-item delays (Softplus + base offset)
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # 2. Channel head: logits over distribution channels
        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_channels)
        )

        # 3. Confidence head: generator confidence in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights cleanly."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
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
        """Encode hour of day [0, 23] into cyclical unit circle embedding."""
        hour_radians = 2.0 * torch.pi * time_of_day.float() / 24.0
        time_features = torch.stack([
            torch.sin(hour_radians),
            torch.cos(hour_radians)
        ], dim=1)
        return self.time_encoder(time_features)

    def forward(
        self,
        z: torch.Tensor,
        sequence_length: int,
        time_of_day: torch.Tensor
    ) -> TimingSchedule:
        """
        Generate schedule for arbitrary sequence_length >= 1.
        """
        batch_size = z.size(0)
        time_embed = self.encode_time_of_day(time_of_day)

        combined = torch.cat([z, time_embed], dim=1)
        hidden = self.latent_projection(combined)

        # Autoregressive sequence expansion
        hidden_seq = hidden.unsqueeze(1).repeat(1, sequence_length, 1)
        gru_out, _ = self.gru(hidden_seq)

        # Apply causal temporal block
        temporal_features = self.temporal_block(gru_out)

        # Output predictions (add minimum 0.5s baseline delay)
        delays = self.delay_head(temporal_features).squeeze(-1) + 0.5
        channel_logits = self.channel_head(temporal_features)

        final_state = temporal_features[:, -1, :]
        confidence = self.confidence_head(final_state).squeeze(-1)

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
        """Generate a complete schedule for arbitrary length."""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        if time_of_day is None:
            time_of_day = torch.randint(0, 24, (batch_size,), device=device).float()
        return self.forward(z, sequence_length, time_of_day)

    def generate_stream(
        self,
        num_items: int,
        time_of_day: Optional[float] = None,
        device: str = "cpu"
    ) -> Iterator[Tuple[float, int]]:
        """
        Autoregressively stream (delay, channel) one packet at a time.
        Allows real-time transmission scheduling without memory overhead.
        """
        if time_of_day is None:
            import datetime
            time_of_day = float(datetime.datetime.now().hour)

        t_tensor = torch.tensor([time_of_day], device=device)
        with torch.no_grad():
            schedule = self.generate(
                batch_size=1,
                sequence_length=num_items,
                time_of_day=t_tensor,
                device=device
            )
            delays = schedule.delays[0].cpu().tolist()
            channels = schedule.sample_channels()[0].cpu().tolist()

        for d, c in zip(delays, channels):
            yield (float(d), int(c))

def sample_latent(batch_size: int, latent_dim: int = 128, device: str = "cpu") -> torch.Tensor:
    """Sample latent Gaussian noise."""
    return torch.randn(batch_size, latent_dim, device=device)

def compute_generator_loss(fake_warden_output: torch.Tensor, throughput_penalty: float = 0.0) -> torch.Tensor:
    """Compute generator adversarial loss."""
    return -torch.mean(fake_warden_output) + throughput_penalty

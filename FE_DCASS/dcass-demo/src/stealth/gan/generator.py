"""
GAN Generator for DCASS Steganography Scheduling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingSchedule:
    delays: torch.Tensor
    channel_logits: torch.Tensor
    confidence: torch.Tensor

    def sample_channels(self, temperature: float = 1.0) -> torch.Tensor:
        scaled_logits = self.channel_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        channel_ids = torch.argmax(probs, dim=-1)
        return channel_ids

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "delays": self.delays,
            "channel_logits": self.channel_logits,
            "confidence": self.confidence,
        }


class TemporalPatternGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_channels: int = 3,
        max_sequence_length: int = 100,
        time_embedding_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.max_sequence_length = max_sequence_length
        self.time_embedding_dim = time_embedding_dim

        self.time_encoder = nn.Sequential(
            nn.Linear(2, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.LayerNorm(time_embedding_dim),
        )

        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + time_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_channels),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def encode_time_of_day(self, time_of_day: torch.Tensor) -> torch.Tensor:
        hour_radians = 2.0 * torch.pi * time_of_day / 24.0
        time_features = torch.stack(
            [torch.sin(hour_radians), torch.cos(hour_radians)], dim=1
        )
        return self.time_encoder(time_features)

    def forward(
        self, z: torch.Tensor, sequence_length: int, time_of_day: torch.Tensor
    ) -> TimingSchedule:
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"sequence_length exceeds max_sequence_length")

        time_embed = self.encode_time_of_day(time_of_day)
        combined = torch.cat([z, time_embed], dim=1)
        hidden = self.latent_projection(combined)

        hidden_seq = hidden.unsqueeze(1).repeat(1, sequence_length, 1)
        gru_out, _ = self.gru(hidden_seq)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        temporal_features = self.attn_norm(gru_out + attn_out)

        delays = self.delay_head(temporal_features).squeeze(-1)
        channel_logits = self.channel_head(temporal_features)
        final_state = temporal_features[:, -1, :]
        confidence = self.confidence_head(final_state).squeeze(-1)

        return TimingSchedule(
            delays=delays, channel_logits=channel_logits, confidence=confidence
        )

    def generate(
        self,
        batch_size: int = 1,
        sequence_length: int = 20,
        time_of_day: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> TimingSchedule:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        if time_of_day is None:
            time_of_day = torch.randint(0, 24, (batch_size,), device=device).float()
        return self.forward(z, sequence_length, time_of_day)


def sample_latent(batch_size, latent_dim=128, device="cpu"):
    return torch.randn(batch_size, latent_dim, device=device)


def compute_generator_loss(fake_warden_output, throughput_penalty=0.0):
    fool_loss = -torch.log(1.0 - fake_warden_output + 1e-8).mean()
    return fool_loss + throughput_penalty

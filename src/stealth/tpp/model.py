"""Conditional Autoregressive Log-Normal Mixture TPP Generator.

Models the joint conditional distribution
    p(delay_t, channel_t | history_<t, hour_t)
      = p(delay_t | h_t) * p(channel_t | h_t, delay_t)

where log(delay_t) follows a K-component Gaussian mixture (intensity-free
Log-Normal Mixture TPP, Shchur et al., ICLR 2020) and channel_t follows a
categorical distribution conditioned on both the recurrent history and the
sampled inter-arrival gap.

During deployment, every step enforces per-channel cooldowns and feeds the
actually routed (delay_t, channel_t) back into the recurrent state so the
generator adapts smoothly to cooldown back-offs without drifting off the
human timing manifold.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.checkpoint_io import load_trained_checkpoint

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class TPPConfig:
    """Architecture and range configuration for TemporalPointProcessGenerator."""

    num_channels: int = 3
    num_components: int = 8
    hidden_dim: int = 128
    num_layers: int = 2
    time_embed_dim: int = 32
    channel_embed_dim: int = 16
    dropout: float = 0.1
    min_delay: float = 0.25
    max_delay: float = 120.0
    log_delay_mean: float = 1.5
    log_delay_std: float = 1.2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TPPConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class TemporalPointProcessGenerator(nn.Module):
    """Autoregressive Log-Normal Mixture TPP with diurnal and channel conditioning."""

    def __init__(self, config: Optional[TPPConfig] = None):
        super().__init__()
        self.config = config or TPPConfig()
        c = self.config

        self.num_channels = c.num_channels
        self.num_components = c.num_components
        self.hidden_dim = c.hidden_dim
        self.min_delay = float(c.min_delay)
        self.max_delay = float(c.max_delay)

        # Register empirical log-delay normalization constants as buffers so
        # they travel with state_dict and device moves cleanly.
        self.register_buffer(
            "log_delay_mean", torch.tensor(float(c.log_delay_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "log_delay_std",
            torch.tensor(max(float(c.log_delay_std), 0.1), dtype=torch.float32),
        )

        # Extra token index `num_channels` serves as the start-of-sequence (BOS) token.
        self.bos_channel_idx = c.num_channels
        self.channel_embedding = nn.Embedding(c.num_channels + 1, c.channel_embed_dim)

        # Cyclical time-of-day encoder (sin/cos of hour + 2nd harmonic for sharper
        # morning/evening transitions).
        self.time_encoder = nn.Sequential(
            nn.Linear(4, c.time_embed_dim),
            nn.GELU(),
            nn.Linear(c.time_embed_dim, c.time_embed_dim),
            nn.LayerNorm(c.time_embed_dim),
        )

        # Step input: [norm_prev_log_delay (1), is_bos (1), channel_embed, time_embed]
        step_in_dim = 2 + c.channel_embed_dim + c.time_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(step_in_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )

        self.gru = nn.GRU(
            input_size=c.hidden_dim,
            hidden_size=c.hidden_dim,
            num_layers=c.num_layers,
            batch_first=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
        )
        self.post_norm = nn.LayerNorm(c.hidden_dim)

        # Log-Normal Mixture delay head: outputs (mix_logits, means, raw_scales)
        self.delay_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, 3 * c.num_components),
        )

        # Channel head conditioned on both GRU context and realized log-delay
        self.channel_head = nn.Sequential(
            nn.Linear(c.hidden_dim + 1, c.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim // 2, c.num_channels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)

        # Spread initial mixture means across [-1.5, +1.5] standardized log-delay
        # so components specialize naturally instead of starting identical.
        last_linear = self.delay_head[-1]
        k = self.num_components
        with torch.no_grad():
            spread = torch.linspace(-1.5, 1.5, k)
            last_linear.bias[k : 2 * k].copy_(spread)
            last_linear.bias[2 * k : 3 * k].fill_(0.0)

    def encode_time_of_day(self, hours: torch.Tensor) -> torch.Tensor:
        """Encode hour in [0, 24) with 1st and 2nd circular harmonics."""
        angle = 2.0 * math.pi * (hours.float() % 24.0) / 24.0
        feats = torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2.0 * angle),
                torch.cos(2.0 * angle),
            ],
            dim=-1,
        )
        return self.time_encoder(feats)

    def _norm_log_delay(self, delays: torch.Tensor) -> torch.Tensor:
        log_d = torch.log(delays.clamp(min=self.min_delay))
        return (log_d - self.log_delay_mean) / self.log_delay_std

    def _decode_delay_params(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map recurrent features h (..., H) to (mix_logits, means, scales)."""
        raw = self.delay_head(h)
        mix_logits, raw_means, raw_scales = torch.chunk(raw, 3, dim=-1)
        means = self.log_delay_mean + self.log_delay_std * raw_means
        scales = F.softplus(raw_scales).clamp(min=0.08, max=2.5)
        return mix_logits, means, scales

    def predict_channel_logits(
        self, h: torch.Tensor, delays: torch.Tensor
    ) -> torch.Tensor:
        """Predict channel logits conditioned on recurrent state h and delay."""
        norm_ld = self._norm_log_delay(delays).unsqueeze(-1)
        return self.channel_head(torch.cat([h, norm_ld], dim=-1))

    def forward(
        self,
        delays: torch.Tensor,
        channels: torch.Tensor,
        time_of_day: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass over a batch of sequences.

        Args:
            delays: (B, T) positive inter-arrival delays in seconds.
            channels: (B, T) integer channel indices in [0, num_channels - 1].
            time_of_day: (B,) starting hour of day in [0, 24).

        Returns:
            mix_logits: (B, T, K)
            means: (B, T, K) log-delay component means
            scales: (B, T, K) log-delay component stds
            channel_logits: (B, T, C)
        """
        bsz, seq_len = delays.shape
        device = delays.device

        # Shift inputs right by 1 step with BOS at t=0.
        prev_delays = torch.zeros_like(delays)
        if seq_len > 1:
            prev_delays[:, 1:] = delays[:, :-1]
        norm_prev_ld = self._norm_log_delay(prev_delays.clamp(min=self.min_delay))
        norm_prev_ld[:, 0] = 0.0

        is_bos = torch.zeros((bsz, seq_len), device=device, dtype=delays.dtype)
        is_bos[:, 0] = 1.0

        prev_channels = torch.full(
            (bsz, seq_len), self.bos_channel_idx, device=device, dtype=torch.long
        )
        if seq_len > 1:
            prev_channels[:, 1:] = channels[:, :-1].clamp(0, self.num_channels - 1)
        ch_embed = self.channel_embedding(prev_channels)

        # Track elapsed hour smoothly across the sequence.
        cum_elapsed_sec = torch.cumsum(prev_delays, dim=1)
        step_hours = (time_of_day.float().unsqueeze(1) + cum_elapsed_sec / 3600.0) % 24.0
        time_embed = self.encode_time_of_day(step_hours)

        step_in = torch.cat(
            [norm_prev_ld.unsqueeze(-1), is_bos.unsqueeze(-1), ch_embed, time_embed],
            dim=-1,
        )
        proj = self.input_proj(step_in)
        gru_out, _ = self.gru(proj)
        h = self.post_norm(gru_out)

        mix_logits, means, scales = self._decode_delay_params(h)
        channel_logits = self.predict_channel_logits(h, delays)
        return mix_logits, means, scales, channel_logits

    def compute_loss(
        self,
        delays: torch.Tensor,
        channels: torch.Tensor,
        time_of_day: torch.Tensor,
        mask: torch.Tensor,
        channel_weight: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute masked negative log-likelihood for delays and channels."""
        mix_logits, means, scales, channel_logits = self.forward(
            delays, channels, time_of_day
        )

        # Target log-delays clamped strictly inside the modeled range.
        clamped_delays = delays.clamp(min=self.min_delay, max=self.max_delay)
        log_d = torch.log(clamped_delays).unsqueeze(-1)  # (B, T, 1)

        log_weights = F.log_softmax(mix_logits, dim=-1)
        comp_log_prob = (
            -0.5 * ((log_d - means) / scales).pow(2)
            - torch.log(scales)
            - 0.5 * _LOG_2PI
        )
        # Exact log-density of Log-Normal mixture in log-space; subtracting log_d
        # yields exact density in seconds.
        log_p_delay = torch.logsumexp(log_weights + comp_log_prob, dim=-1)
        delay_nll_per_step = -log_p_delay + log_d.squeeze(-1)

        ch_nll_per_step = F.cross_entropy(
            channel_logits.reshape(-1, self.num_channels),
            channels.clamp(0, self.num_channels - 1).reshape(-1),
            reduction="none",
        ).reshape(delays.shape)

        denom = mask.sum().clamp(min=1.0)
        delay_nll = (delay_nll_per_step * mask).sum() / denom
        channel_nll = (ch_nll_per_step * mask).sum() / denom
        total_loss = delay_nll + float(channel_weight) * channel_nll
        return total_loss, delay_nll, channel_nll

    def step(
        self,
        prev_delay: Optional[torch.Tensor],
        prev_channel: Optional[torch.Tensor],
        hour: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the autoregressive GRU by one step.

        Args:
            prev_delay: (B,) float tensor or None for step t=0 (BOS).
            prev_channel: (B,) long tensor or None for step t=0 (BOS).
            hour: (B,) current hour of day in [0, 24).
            hidden: Optional GRU hidden state (num_layers, B, H).

        Returns:
            mix_logits: (B, K)
            means: (B, K)
            scales: (B, K)
            h_t: (B, H) normalized recurrent context
            next_hidden: (num_layers, B, H)
        """
        bsz = hour.size(0)
        device = hour.device

        if prev_delay is None or prev_channel is None:
            norm_ld = torch.zeros(bsz, device=device, dtype=torch.float32)
            is_bos = torch.ones(bsz, device=device, dtype=torch.float32)
            ch_idx = torch.full(
                (bsz,), self.bos_channel_idx, device=device, dtype=torch.long
            )
        else:
            norm_ld = self._norm_log_delay(prev_delay.clamp(min=self.min_delay))
            is_bos = torch.zeros(bsz, device=device, dtype=torch.float32)
            ch_idx = prev_channel.long().clamp(0, self.num_channels - 1)

        ch_embed = self.channel_embedding(ch_idx)
        time_embed = self.encode_time_of_day(hour)

        step_in = torch.cat(
            [norm_ld.unsqueeze(-1), is_bos.unsqueeze(-1), ch_embed, time_embed],
            dim=-1,
        ).unsqueeze(1)  # (B, 1, D)

        proj = self.input_proj(step_in)
        gru_out, next_hidden = self.gru(proj, hidden)
        h_t = self.post_norm(gru_out.squeeze(1))
        mix_logits, means, scales = self._decode_delay_params(h_t)
        return mix_logits, means, scales, h_t, next_hidden

    @torch.no_grad()
    def generate_routed_schedule(
        self,
        num_items: int,
        time_of_day: float = 14.0,
        cooldowns: Optional[Dict[int, float]] = None,
        pause_cap_s: Optional[float] = None,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        device: Optional[str] = None,
    ) -> Tuple[list[float], list[int]]:
        """Generate a complete cooldown-safe (delays, channels) schedule online.

        At every step, enforces per-channel cooldowns and feeds the realized
        (delay, channel) back into the recurrent state.
        """
        if num_items <= 0:
            return [], []

        self.eval()
        dev = torch.device(device) if device is not None else next(self.parameters()).device
        temp = max(float(temperature), 1e-3)
        upper_cap = float(pause_cap_s) if pause_cap_s is not None else self.max_delay
        upper_cap = max(self.min_delay, min(upper_cap, self.max_delay))

        cd_table = (
            {c: 6.0 for c in range(self.num_channels)}
            if cooldowns is None
            else dict(cooldowns)
        )
        last_landing = {c: -1000.0 for c in range(self.num_channels)}
        current_landing = 0.0

        delays_out: list[float] = []
        channels_out: list[int] = []

        hidden: Optional[torch.Tensor] = None
        prev_delay_t: Optional[torch.Tensor] = None
        prev_ch_t: Optional[torch.Tensor] = None

        for _ in range(num_items):
            curr_hour = (float(time_of_day) + current_landing / 3600.0) % 24.0
            hour_t = torch.tensor([curr_hour], dtype=torch.float32, device=dev)

            mix_logits, means, scales, h_t, hidden = self.step(
                prev_delay_t, prev_ch_t, hour_t, hidden
            )

            # Sample delay from Log-Normal mixture
            probs = F.softmax(mix_logits[0] / temp, dim=-1).cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()
            if rng is not None:
                comp = int(rng.choice(self.num_components, p=probs))
                z = float(rng.standard_normal())
            else:
                comp = int(np.random.choice(self.num_components, p=probs))
                z = float(np.random.standard_normal())

            log_d = float(means[0, comp].item()) + float(scales[0, comp].item()) * z
            delay = float(np.clip(math.exp(np.clip(log_d, -6.0, 8.0)), self.min_delay, upper_cap))

            # Predict channel logits given h_t and proposed delay
            delay_tensor = torch.tensor([delay], dtype=torch.float32, device=dev)
            ch_logits = self.predict_channel_logits(h_t, delay_tensor)[0].cpu().numpy().astype(np.float64)

            # Cooldown-aware routing in float64
            projected = current_landing + delay
            available = [
                c
                for c in range(self.num_channels)
                if projected - last_landing[c] >= cd_table.get(c, 6.0)
            ]
            if available:
                avail_logits = np.array([ch_logits[c] for c in available], dtype=np.float64) / temp
                avail_logits = avail_logits - np.max(avail_logits)
                w = np.exp(avail_logits)
                w = w / w.sum()
                if rng is not None:
                    chosen = int(rng.choice(available, p=w))
                else:
                    chosen = int(np.random.choice(available, p=w))
            else:
                chosen = min(
                    range(self.num_channels),
                    key=lambda c: last_landing[c] + cd_table.get(c, 6.0),
                )
                wait = last_landing[chosen] + cd_table.get(chosen, 6.0) - projected
                delay = float(delay + max(0.0, wait))
                projected = current_landing + delay

            current_landing = projected
            last_landing[chosen] = current_landing
            delays_out.append(delay)
            channels_out.append(chosen)

            # Autoregressive feedback of the realized (delay, channel)
            prev_delay_t = torch.tensor([delay], dtype=torch.float32, device=dev)
            prev_ch_t = torch.tensor([chosen], dtype=torch.long, device=dev)

        return delays_out, channels_out

    def save_checkpoint(self, path: Path | str, extra: Optional[dict] = None) -> None:
        """Save weights and plain-dict config (strict weights_only loader compatible)."""
        payload = {
            "model_type": "lognormal_mixture_tpp",
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(out))

    @classmethod
    def load_checkpoint(
        cls, path: Path | str, device: str = "cpu"
    ) -> "TemporalPointProcessGenerator":
        ckpt = load_trained_checkpoint(str(path), map_location=device)
        cfg = TPPConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model

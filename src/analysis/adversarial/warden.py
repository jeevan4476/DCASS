# src/analysis/adversarial/warden.py
"""
Adversarial Warden - Deep Packet Inspection Simulator for DCASS.

This module implements a deep learning-based traffic classifier that acts as
an advanced network firewall to detect steganographic communication patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class WardenVerdict:
    """
    Output of the Warden's traffic analysis.

    Attributes:
        bot_probability: Probability that traffic is bot/steganography, shape (batch_size,)
        anomaly_scores: Per-timestamp anomaly scores, shape (batch_size, sequence_length)
        feature_importance: Dictionary of attention weights for interpretability
    """

    bot_probability: torch.Tensor
    anomaly_scores: torch.Tensor
    feature_importance: dict[str, torch.Tensor]

    def is_suspicious(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Check if traffic is suspicious based on threshold.

        Args:
            threshold: Bot probability threshold (default: 0.5)

        Returns:
            Boolean tensor indicating suspicious traffic, shape (batch_size,)
        """
        return self.bot_probability > threshold

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary for easier serialization."""
        return {
            "bot_probability": self.bot_probability,
            "anomaly_scores": self.anomaly_scores,
            **self.feature_importance,
        }


class DeepPacketInspectionWarden(nn.Module):
    """
    Adversarial Warden - DPI-style Traffic Classifier.

    Analyzes transmission timing patterns to detect steganographic behavior.
    Acts as the Discriminator in the GAN framework and provides adversarial
    feedback to the RL agent.

    Key Detection Features:
    - Inter-arrival time distribution (human = Poisson-like, bot = regular)
    - Burst analysis (coefficient of variation)
    - Channel switching patterns
    - Circadian rhythm compliance

    Architecture:
        Traffic Timeline → Feature Extraction → BiLSTM → Transformer Encoder → Classification

    The Warden is trained on both real human traffic and generated steganographic
    traffic to learn distinguishing patterns.

    Args:
        num_channels: Number of distribution channels (default: 3)
        hidden_dim: LSTM hidden dimension (default: 256)
        num_transformer_layers: Depth of transformer encoder (default: 4)
        feature_dim: Manual statistical feature dimension (default: 16)
        dropout: Dropout probability (default: 0.3)

    Example:
        >>> warden = DeepPacketInspectionWarden(num_channels=3)
        >>> delays = torch.rand(8, 20) * 10  # Random delays
        >>> channels = torch.randint(0, 3, (8, 20))  # Random channels
        >>> verdict = warden(delays, channels)
        >>> print(verdict.bot_probability)  # Probabilities for each sample
        >>> print(verdict.is_suspicious().sum().item(), "suspicious samples detected")
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        feature_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Channel embedding (learn channel-specific patterns)
        self.channel_embedding = nn.Embedding(num_channels, 32)

        # Time-delay embedding (log-scale for wide range of delays)
        self.delay_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

        # Combine embeddings: delay + channel + statistical features
        input_dim = 64 + 32 + feature_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM for temporal modeling
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        # Transformer encoder for global pattern recognition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Anomaly detection head (per-timestamp)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Anomaly score [0, 1]
        )

        # Classification head (global verdict).
        # NOTE: No sigmoid here — this is an unbounded Wasserstein critic score.
        # For bot-probability interpretation, use torch.sigmoid(raw_score) externally.
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # No Sigmoid: unbounded critic output required for WGAN-GP Wasserstein loss.
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def extract_statistical_features(self, delays: torch.Tensor) -> torch.Tensor:
        """
        Extract handcrafted statistical features from delay sequence.

        These features capture the statistical fingerprint of traffic patterns
        that distinguish human behavior from bots.

        Features extracted:
        1. Mean, std, min, max of delays
        2. Coefficient of variation (CV = std/mean) - bots have low CV
        3. Skewness (third moment) - human traffic is often right-skewed
        4. Kurtosis (fourth moment) - measures tail heaviness
        5. Delay range (max - min)
        6. Autocorrelation at lag 1 - regularity indicator

        Args:
            delays: Inter-transmission delays, shape (batch_size, sequence_length)

        Returns:
            Statistical features, shape (batch_size, feature_dim)
        """
        batch_size = delays.size(0)
        eps = 1e-8  # Numerical stability

        # Basic statistics
        mean_delay = delays.mean(dim=1, keepdim=True)
        std_delay = delays.std(dim=1, keepdim=True)
        min_delay = delays.min(dim=1, keepdim=True)[0]
        max_delay = delays.max(dim=1, keepdim=True)[0]

        # Coefficient of variation (bot traffic often has very low CV)
        cv = std_delay / (mean_delay + eps)

        # Centered delays for moment calculations
        centered = delays - mean_delay

        # Skewness (third moment) - human traffic often right-skewed
        skewness = (centered**3).mean(dim=1, keepdim=True) / (std_delay**3 + eps)

        # Kurtosis (fourth moment) - measures tail heaviness
        kurtosis = (centered**4).mean(dim=1, keepdim=True) / (std_delay**4 + eps)

        # Delay range
        delay_range = max_delay - min_delay

        # Autocorrelation at lag 1 (regularity indicator)
        if delays.size(1) > 1:
            delays_t = delays[:, :-1]
            delays_t1 = delays[:, 1:]
            # Pearson correlation
            mean_t = delays_t.mean(dim=1, keepdim=True)
            mean_t1 = delays_t1.mean(dim=1, keepdim=True)
            cov = ((delays_t - mean_t) * (delays_t1 - mean_t1)).mean(
                dim=1, keepdim=True
            )
            std_t = delays_t.std(dim=1, keepdim=True)
            std_t1 = delays_t1.std(dim=1, keepdim=True)
            autocorr = cov / (std_t * std_t1 + eps)
        else:
            autocorr = torch.zeros(batch_size, 1, device=delays.device)

        # Median absolute deviation (robust measure of variability)
        median_delay = delays.median(dim=1, keepdim=True)[0]
        mad = (delays - median_delay).abs().median(dim=1, keepdim=True)[0]

        # Combine all features
        features = torch.cat(
            [
                mean_delay,
                std_delay,
                min_delay,
                max_delay,
                cv,
                skewness,
                kurtosis,
                delay_range,
                autocorr,
                median_delay,
                mad,
            ],
            dim=1,
        )

        # Pad or truncate to feature_dim
        if features.size(1) < self.feature_dim:
            padding = torch.zeros(
                batch_size, self.feature_dim - features.size(1), device=delays.device
            )
            features = torch.cat([features, padding], dim=1)
        else:
            features = features[:, : self.feature_dim]

        return features

    def forward(
        self,
        delays: torch.Tensor,
        channel_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> WardenVerdict:
        """
        Analyze traffic timeline and classify as Human vs Bot.

        Args:
            delays: Inter-transmission delays (seconds), shape (batch_size, sequence_length)
            channel_ids: Channel indices [0, num_channels-1] (long, shape
                (batch_size, sequence_length)) OR soft channel probabilities
                (float, shape (batch_size, sequence_length, num_channels))
                for differentiable straight-through training.
            timestamps: Absolute Unix timestamps (optional), shape (batch_size, sequence_length)

        Returns:
            WardenVerdict with bot probability and anomaly scores

        Raises:
            ValueError: If delays and channel_ids have mismatched shapes
        """
        batch_size, seq_len = delays.size()

        if channel_ids.dtype in (torch.long, torch.int):
            if delays.shape != channel_ids.shape:
                raise ValueError(
                    f"delays shape {delays.shape} != channel_ids shape {channel_ids.shape}"
                )
        else:
            # Soft distribution path (straight-through Gumbel-Softmax)
            if (
                channel_ids.dim() != 3
                or channel_ids.size(0) != batch_size
                or channel_ids.size(1) != seq_len
            ):
                raise ValueError(
                    f"soft channel probs shape {tuple(channel_ids.shape)} does not "
                    f"match delays shape {tuple(delays.shape)}"
                )

        # Extract statistical features (global features for the entire sequence)
        stat_features = self.extract_statistical_features(
            delays
        )  # (batch_size, feature_dim)
        stat_features_expanded = stat_features.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # (batch_size, seq_len, feature_dim)

        # Embed delays (log-transform for wide range of values)
        log_delays = torch.log(delays + 1.0).unsqueeze(-1)  # (batch_size, seq_len, 1)
        delay_embed = self.delay_embedding(log_delays)  # (batch_size, seq_len, 64)

        # Embed channels: hard indices via lookup, soft probabilities as a
        # weighted sum of embeddings (differentiable back to the generator's
        # channel head).
        if channel_ids.dtype in (torch.long, torch.int):
            channel_embed = self.channel_embedding(
                channel_ids
            )  # (batch_size, seq_len, 32)
        else:
            channel_embed = channel_ids @ self.channel_embedding.weight

        # Combine all embeddings
        combined = torch.cat(
            [delay_embed, channel_embed, stat_features_expanded], dim=2
        )  # (batch_size, seq_len, 64+32+feature_dim)

        # Project to hidden dimension
        sequence_features = self.input_projection(
            combined
        )  # (batch_size, seq_len, hidden_dim)

        # Temporal modeling with BiLSTM
        lstm_out, _ = self.bilstm(
            sequence_features
        )  # (batch_size, seq_len, hidden_dim)

        # Global pattern recognition with Transformer
        transformer_out = self.transformer(
            lstm_out
        )  # (batch_size, seq_len, hidden_dim)

        # Per-timestamp anomaly scores
        anomaly_scores = self.anomaly_head(transformer_out).squeeze(
            -1
        )  # (batch_size, seq_len)

        # Global classification using dual pooling
        mean_pool = transformer_out.mean(dim=1)  # (batch_size, hidden_dim)
        max_pool = transformer_out.max(dim=1)[0]  # (batch_size, hidden_dim)
        global_features = torch.cat(
            [mean_pool, max_pool], dim=1
        )  # (batch_size, hidden_dim*2)

        # Final classification: unbounded Wasserstein critic score.
        raw_critic_score = self.classification_head(global_features).squeeze(
            -1
        )  # (batch_size,)

        # Sigmoid-bounded bot probability for RL reward / interpretability.
        # During GAN training use raw_critic_score directly (from feature_importance).
        bot_probability = torch.sigmoid(raw_critic_score)

        # Feature importance (use anomaly scores as interpretability proxy)
        feature_importance = {
            "anomaly_scores": anomaly_scores,
            "global_attention": torch.softmax(anomaly_scores, dim=1),
            "statistical_features": stat_features,
            "raw_critic_score": raw_critic_score,  # unbounded Wasserstein critic value
        }

        return WardenVerdict(
            bot_probability=bot_probability,
            anomaly_scores=anomaly_scores,
            feature_importance=feature_importance,
        )


def compute_warden_loss(
    real_verdict: WardenVerdict,
    fake_verdict: WardenVerdict,
    label_smoothing: float = 0.1,  # kept for API compatibility, unused in WGAN-GP
) -> torch.Tensor:
    """
    Compute Warden (Discriminator) loss for WGAN-GP training.

    Wasserstein critic loss: L_D = E[D(fake)] - E[D(real)]
    Minimising this maximises E[D(real)] - E[D(fake)], i.e. the Wasserstein-1
    distance lower bound between real and generated distributions.

    Requires:
        - Warden classification_head has NO sigmoid activation (unbounded output).
        - Gradient penalty added separately via compute_gradient_penalty().

    Args:
        real_verdict: Warden output for real human traffic.
        fake_verdict: Warden output for generated (fake) traffic.
        label_smoothing: Unused. Kept for backward-compatible call signatures.

    Returns:
        Wasserstein critic loss (scalar, negate w.r.t. real data minus fake data).
    """
    real_scores = real_verdict.feature_importance["raw_critic_score"]
    fake_scores = fake_verdict.feature_importance["raw_critic_score"]

    # Wasserstein critic loss: minimise E[D(fake)] - E[D(real)]
    return fake_scores.mean() - real_scores.mean()


def compute_warden_loss_bce(
    real_verdict: WardenVerdict,
    fake_verdict: WardenVerdict,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Deprecated BCE-based Warden loss. Kept for reference and test backward compat.

    DO NOT use for WGAN-GP training — use compute_warden_loss() instead.
    """
    # Real labels (with smoothing): 0 → label_smoothing
    # Fake labels (with smoothing): 1 → 1 - label_smoothing
    real_target = torch.full_like(real_verdict.bot_probability, label_smoothing)
    fake_target = torch.full_like(fake_verdict.bot_probability, 1.0 - label_smoothing)

    # Binary cross-entropy loss
    real_loss = nn.functional.binary_cross_entropy(
        real_verdict.bot_probability, real_target
    )

    fake_loss = nn.functional.binary_cross_entropy(
        fake_verdict.bot_probability, fake_target
    )

    # Total discriminator loss
    return (real_loss + fake_loss) / 2.0


def compute_gradient_penalty(
    warden: DeepPacketInspectionWarden,
    real_delays: torch.Tensor,
    fake_delays: torch.Tensor,
    real_channels: torch.Tensor,
    fake_channels: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP training.

    This enforces the Lipschitz constraint for Wasserstein GAN training.

    Args:
        warden: The Warden model
        real_delays: Real human traffic delays
        fake_delays: Generated traffic delays
        real_channels: Real channel assignments
        fake_channels: Generated channel assignments
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss (scalar)
    """
    batch_size = real_delays.size(0)
    device = real_delays.device

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolate between real and fake
    interpolated_delays = alpha * real_delays + (1 - alpha) * fake_delays
    interpolated_delays.requires_grad_(True)

    # For channels, use real channels (since they're discrete)
    interpolated_channels = real_channels

    # Get Warden verdict on interpolated data with math attention & CuDNN disabled for double backwards
    with (
        torch.backends.cudnn.flags(enabled=False),
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ),
    ):
        verdict = warden(interpolated_delays, interpolated_channels)

        # Compute gradients on the RAW critic score, not the sigmoid view.
        # WGAN-GP constrains ||grad D|| -> 1; the sigmoid squashes gradients
        # by <= 0.25x making the constraint nearly unsatisfiable.
        raw_critic = verdict.feature_importance["raw_critic_score"]
        gradients = torch.autograd.grad(
            outputs=raw_critic.sum(),
            inputs=interpolated_delays,
            create_graph=True,
            retain_graph=True,
        )[0]

    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()

    return gradient_penalty


if __name__ == "__main__":
    # Quick test
    print("Testing DeepPacketInspectionWarden...")

    warden = DeepPacketInspectionWarden(
        num_channels=3, hidden_dim=256, num_transformer_layers=4
    )

    # Create synthetic traffic data
    batch_size = 8
    seq_len = 20

    # Real human traffic (Poisson-like, variable delays)
    real_delays = torch.abs(torch.randn(batch_size, seq_len) * 5 + 10)
    real_channels = torch.randint(0, 3, (batch_size, seq_len))

    # Bot traffic (more regular, lower variance)
    bot_delays = torch.abs(torch.randn(batch_size, seq_len) * 1 + 8)
    bot_channels = torch.randint(0, 3, (batch_size, seq_len))

    # Test Warden
    real_verdict = warden(real_delays, real_channels)
    bot_verdict = warden(bot_delays, bot_channels)

    print(f"✓ Real traffic bot probability: {real_verdict.bot_probability.mean():.3f}")
    print(f"✓ Bot traffic bot probability: {bot_verdict.bot_probability.mean():.3f}")
    print(f"✓ Anomaly scores shape: {real_verdict.anomaly_scores.shape}")
    print(
        f"✓ Suspicious real samples: {real_verdict.is_suspicious().sum().item()}/{batch_size}"
    )
    print(
        f"✓ Suspicious bot samples: {bot_verdict.is_suspicious().sum().item()}/{batch_size}"
    )

    # Test loss computation
    loss = compute_warden_loss(real_verdict, bot_verdict)
    print(f"✓ Warden loss: {loss.item():.4f}")

    print("\nWarden parameters:", sum(p.numel() for p in warden.parameters()))
    print("Warden ready for training!")

# tests/test_stealth/test_gan.py
"""
Unit and Integration tests for upgraded DCASS WGAN-GP Temporal Pattern Generator.
Tests arbitrary sequence lengths, causal temporal blocks, streaming generator, and Warden critic.
"""

import pytest
import torch
from pathlib import Path
from src.stealth.gan.generator import TemporalPatternGenerator, sample_latent
from src.analysis.adversarial.warden import DeepPacketInspectionWarden, compute_gradient_penalty
from src.stealth.stealth_scheduler import StealthScheduler

def test_temporal_pattern_generator_arbitrary_lengths():
    """Verify generator produces valid schedules for various sequence lengths (short, medium, long)."""
    gen = TemporalPatternGenerator(
        latent_dim=128,
        hidden_dim=64,
        num_channels=3,
        time_embedding_dim=16
    )

    time_of_day = torch.tensor([14, 2])

    for seq_len in [1, 5, 25, 120, 350]:
        z = torch.randn(2, 128)
        schedule = gen(z, sequence_length=seq_len, time_of_day=time_of_day)

        assert schedule.delays.shape == (2, seq_len)
        assert schedule.channel_logits.shape == (2, seq_len, 3)
        assert schedule.confidence.shape == (2,)

        # Delays must be strictly >= 0.5s baseline
        assert (schedule.delays >= 0.5).all()

        channels = schedule.sample_channels()
        assert channels.shape == (2, seq_len)
        assert (channels >= 0).all() and (channels < 3).all()

def test_temporal_pattern_generator_streaming():
    """Verify generate_stream yields one packet delay/channel at a time without memory blowup."""
    gen = TemporalPatternGenerator(
        latent_dim=128,
        hidden_dim=64,
        num_channels=3
    )

    stream = gen.generate_stream(num_items=15, time_of_day=14.0)
    items_yielded = list(stream)

    assert len(items_yielded) == 15
    for delay, channel in items_yielded:
        assert isinstance(delay, float)
        assert delay >= 0.5
        assert isinstance(channel, int)
        assert 0 <= channel < 3

def test_warden_forward():
    """Verify Warden discriminator forward pass."""
    batch_size = 4
    seq_len = 20
    num_channels = 3

    warden = DeepPacketInspectionWarden(
        num_channels=num_channels,
        hidden_dim=64,
        num_transformer_layers=2,
        feature_dim=16
    )

    delays = torch.rand(batch_size, seq_len) * 5.0 + 1.0
    channels = torch.randint(0, num_channels, (batch_size, seq_len))

    verdict = warden(delays, channels)

    assert verdict.bot_probability.shape == (batch_size,)
    assert verdict.anomaly_scores.shape == (batch_size, seq_len)
    assert (verdict.bot_probability >= 0.0).all() and (verdict.bot_probability <= 1.0).all()

def test_wgan_gradient_penalty():
    """Verify WGAN-GP gradient penalty computation on Warden."""
    batch_size = 4
    seq_len = 10
    num_channels = 3

    warden = DeepPacketInspectionWarden(
        num_channels=num_channels,
        hidden_dim=32,
        num_transformer_layers=1,
        feature_dim=8
    )

    real_delays = torch.rand(batch_size, seq_len) * 4.0 + 1.0
    fake_delays = torch.rand(batch_size, seq_len) * 4.0 + 1.0
    real_channels = torch.randint(0, num_channels, (batch_size, seq_len))
    fake_channels = torch.randint(0, num_channels, (batch_size, seq_len))

    gp = compute_gradient_penalty(
        warden,
        real_delays,
        fake_delays,
        real_channels,
        fake_channels
    )

    assert isinstance(gp, torch.Tensor)
    assert gp.ndim == 0
    assert gp.item() >= 0.0

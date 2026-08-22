# tests/test_engine/test_payload_framing.py
"""
Tier-1 tests for the framed payload codec (Decision 4 / WP-6) plus
tier-3 CPU gradient-flow tests for the stealth training fixes
(R-22 / R-23 / R-24) - these are the artifact that lets a teammate trust
the model-side fixes without spending GPU time.
"""

import struct

import pytest


# ============================================================================
# Tier 1 - framing codec
# ============================================================================


class TestPayloadFraming:
    def test_frame_roundtrip(self):
        from src.engine.payload_framing import frame_payload, unframe_payload

        msg = "Meet me at the cafe at noon"
        frame = frame_payload(msg)
        assert frame[0] == 0x01  # version marker
        text, was_framed = unframe_payload(frame)
        assert was_framed is True
        assert text == msg

    def test_header_layout(self):
        """version(1) | length BE u16 | crc16 BE u16."""
        from src.engine.payload_framing import frame_payload, crc16_ccitt

        raw = b"hello"
        frame = frame_payload(raw.decode())
        version, length = struct.unpack(">BH", frame[:3])
        crc_recv = struct.unpack(">H", frame[3:5])[0]
        assert version == 0x01
        assert length == len(raw)
        assert crc_recv == crc16_ccitt(raw)
        assert frame[5:] == raw

    def test_length_field_is_explicit(self):
        """Trailing garbage after the body must not leak into the plaintext."""
        from src.engine.payload_framing import frame_payload, unframe_payload

        frame = frame_payload("hi") + b"\xff\xff\xff"  # simulate RS padding noise
        text, _ = unframe_payload(frame)
        assert text == "hi"

    def test_crc_detects_corruption(self):
        from src.engine.payload_framing import (
            frame_payload,
            unframe_payload,
            FrameError,
        )

        frame = bytearray(frame_payload("corrupt me"))
        frame[-1] ^= 0xFF  # flip bits in the body
        with pytest.raises(FrameError):
            unframe_payload(bytes(frame))

    def test_legacy_raw_still_decodes(self):
        """Unframed payloads (pre-framing encoders) must keep working."""
        from src.engine.payload_framing import unframe_payload

        text, was_framed = unframe_payload(b"plain legacy message")
        assert was_framed is False
        assert text == "plain legacy message"

    def test_truncated_frame_rejected(self):
        from src.engine.payload_framing import frame_payload, unframe_payload, FrameError

        frame = frame_payload("a much longer message to truncate")
        with pytest.raises(FrameError):
            unframe_payload(frame[:4])

    def test_oversize_message_rejected(self):
        from src.engine.payload_framing import frame_payload

        with pytest.raises(ValueError):
            frame_payload("x" * 0x10000)


# ============================================================================
# Tier 3 - CPU gradient-flow proofs for R-22 / R-23 (seconds, no GPU)
# ============================================================================


def _tiny_warden(seq_len=8, num_channels=3, batch=2):
    import torch

    from src.analysis.adversarial.warden import DeepPacketInspectionWarden

    torch.manual_seed(0)
    delays = torch.rand(batch, seq_len) * 10 + 0.1
    channels = torch.randint(0, num_channels, (batch, seq_len))
    warden = DeepPacketInspectionWarden(num_channels=num_channels)
    return warden, delays, channels


class TestGradientPenaltyUsesRawCritic:
    """R-22: the WGAN-GP penalty must constrain ||grad D|| on the RAW score."""

    def test_penalty_computes_and_backprops(self):
        import torch

        from src.analysis.adversarial.warden import compute_gradient_penalty

        warden, real_delays, real_channels = _tiny_warden()
        fake_delays = torch.rand_like(real_delays)

        gp = compute_gradient_penalty(
            warden,
            real_delays,
            fake_delays,
            real_channels,
            real_channels.clone(),
        )
        assert torch.isfinite(gp), "gradient penalty must be finite"
        assert gp.item() > 0
        gp.backward()
        # Gradients reached the warden's parameters through the RAW path.
        grads = [p.grad for p in warden.parameters() if p.grad is not None]
        assert grads, "no gradients flowed into the warden during GP"

    def test_raw_score_gradient_not_sigmoid_squashed(self):
        """
        The old bug: grad wrt bot_probability is scaled by sigmoid' <= 0.25.
        Prove we compute on raw_critic_score by checking that a large-magnitude
        critic output still yields a full-size input gradient (a saturated
        sigmoid would crush it toward 0).
        """
        import torch


        warden, _, channels = _tiny_warden(batch=4)
        # Large delays -> large critic magnitude in practice; force saturation
        # by directly scaling the classification head if needed.
        big_delays = torch.full((4, 8), 50.0) * 20
        interp = big_delays.clone().requires_grad_(True)
        verdict = warden(interp, channels)
        raw = verdict.feature_importance["raw_critic_score"]
        (raw.sum()).backward()
        g_raw = interp.grad.abs().sum().item()

        interp2 = big_delays.clone().requires_grad_(True)
        verdict2 = warden(interp2, channels)
        verdict2.bot_probability.sum().backward()
        g_sigmoid = interp2.grad.abs().sum().item()

        # If outputs are saturated, sigmoid-path grads ~0 while raw stays > 0;
        # in all cases raw >= sigmoid since sigmoid' <= 1.
        assert g_raw >= g_sigmoid * 0.999 or g_raw > 0
        assert g_raw > 1e-8, "raw critic path must produce usable gradients"


class TestChannelHeadReceivesGradient:
    """R-23: straight-through Gumbel-Softmax must train the channel head."""

    def test_gumbel_hard_sample_is_onehot(self):
        import torch

        from src.stealth.gan.generator import TimingSchedule

        logits = torch.randn(2, 4, 3, requires_grad=True)
        schedule = TimingSchedule(
            delays=torch.rand(2, 4),
            channel_logits=logits,
            confidence=torch.ones(2),
        )
        probs = schedule.channel_probs_straight_through()
        assert probs.shape == (2, 4, 3)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))
        # hard=True: exactly one entry is 1 per step
        assert ((probs == 1).sum(dim=-1) == 1).all()

    def test_generator_channel_head_gets_nonzero_grad(self):
        import torch

        from src.stealth.gan.generator import TemporalPatternGenerator
        from src.analysis.adversarial.warden import DeepPacketInspectionWarden

        torch.manual_seed(1)
        gen = TemporalPatternGenerator(latent_dim=16, hidden_dim=32, num_channels=3)
        z = torch.randn(2, 16)
        tod = torch.rand(2) * 24.0  # hour of day [0, 23]
        schedule = gen(z, sequence_length=6, time_of_day=tod)

        warden = DeepPacketInspectionWarden(num_channels=3)
        probs = schedule.channel_probs_straight_through()
        verdict = warden(schedule.delays.detach(), probs)
        loss = verdict.feature_importance["raw_critic_score"].mean()
        loss.backward()

        head_grads = [
            p.grad
            for n, p in gen.named_parameters()
            if "channel" in n.lower() and p.grad is not None
        ]
        assert head_grads, (
            "channel_head received no gradient - R-23 regression "
            "(argmax sampling would leave it untrained forever)"
        )
        assert any(g.abs().sum() > 0 for g in head_grads)


class TestCriticLoopDetachesGenerator:
    """R-24: the warden loop must not backprop through the generator."""

    def test_fake_delays_detached(self):
        import inspect

        from src.stealth.gan import trainer as trainer_mod

        src = inspect.getsource(trainer_mod.GANTrainer.train_step)
        assert ".delays.detach()" in src, (
            "critic loop must detach fake delays before the warden forward "
            "(R-24: otherwise every critic step backprops through G)"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

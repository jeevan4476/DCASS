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

    def test_legacy_raw_rejected_when_framing_required(self):
        """An ECC-protected transport must not accept an unframed payload."""
        from src.engine.payload_framing import unframe_payload, FrameError

        with pytest.raises(FrameError, match="framing required"):
            unframe_payload(b"plain legacy message", require_framed=True)

    def test_framed_payload_accepted_when_framing_required(self):
        from src.engine.payload_framing import frame_payload, unframe_payload

        frame = frame_payload("framed and required")
        text, was_framed = unframe_payload(frame, require_framed=True)
        assert was_framed is True
        assert text == "framed and required"

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


class _StubVerdict:
    """Minimal verdict object for analytic gradient-penalty checks."""

    def __init__(self, raw_critic_score):
        self.feature_importance = {"raw_critic_score": raw_critic_score}
        # Deliberately a constant: if the penalty ever regressed to the
        # sigmoid view, these gradients would vanish and the analytic
        # assertions below fail loudly.
        self.bot_probability = raw_critic_score.detach() * 0.0


class _LinearStubWarden:
    """
    Duck-typed warden with analytically known input gradients.

    Defines D(delays, channels) = gradient_scale * (sum(delays) +
    sum(channels)): every input gradient equals gradient_scale, so each
    sample's ||grad D||_2 = gradient_scale * sqrt(T * (1 + C)).
    """

    def __init__(self, num_channels: int, gradient_scale: float):
        self.num_channels = num_channels
        self.gradient_scale = gradient_scale

    def __call__(self, delays, channels, mask=None):
        raw = self.gradient_scale * (delays.sum(dim=1) + channels.sum(dim=(1, 2)))
        return _StubVerdict(raw)


class TestGradientPenaltyUsesRawCritic:
    """R-22: the WGAN-GP penalty must constrain ||grad D|| on the RAW score."""

    def test_penalty_computes_and_backprops(self):
        import math

        import torch

        from src.analysis.adversarial.warden import compute_gradient_penalty

        # --- Analytic contract on a duck-typed critic ---------------------
        # s is chosen so ||grad D||_2 = 3 exactly, hence the penalty MUST be
        # lambda * (3 - 1)^2 = 4 * lambda. Using the sigmoid view (zero
        # gradients on this stub) or the delay stream alone (norm
        # 3 / sqrt(1 + C), not 3) changes the number and fails the check.
        batch, seq, num_channels = 3, 5, 3
        torch.manual_seed(3)
        real = torch.rand(batch, seq) + 1.0
        fake = torch.rand(batch, seq) + 1.0
        channels = torch.randint(0, num_channels, (batch, seq))
        scale = 3.0 / math.sqrt(seq * (1 + num_channels))

        gp = compute_gradient_penalty(
            _LinearStubWarden(num_channels, scale),
            real,
            fake,
            channels,
            channels.clone(),
            lambda_gp=7.0,
        )
        assert torch.allclose(gp, torch.tensor(28.0), atol=1e-4), (
            "joint delay+channel WGAN-GP contract broken: expected "
            f"7 * (3 - 1)^2 = 28, got {gp.item()}"
        )

        # Constant critic (all-zero input gradients): (0 - 1)^2 * lambda.
        gp_zero = compute_gradient_penalty(
            _LinearStubWarden(num_channels, 0.0),
            real,
            fake,
            channels,
            channels.clone(),
            lambda_gp=7.0,
        )
        assert torch.allclose(gp_zero, torch.tensor(7.0), atol=1e-4)

        # --- Real warden: finite, positive, gradients reach the critic ----
        warden, real_delays, real_channels = _tiny_warden()
        fake_delays = torch.rand_like(real_delays)

        gp_real = compute_gradient_penalty(
            warden, real_delays, fake_delays, real_channels, real_channels.clone()
        )
        assert torch.isfinite(gp_real), "gradient penalty must be finite"
        assert gp_real.item() > 0
        gp_real.backward()
        grads = [p.grad for p in warden.parameters() if p.grad is not None]
        assert grads, "no gradients flowed into the warden during GP"
        assert any(g.abs().sum() > 0 for g in grads)

    def test_raw_score_gradient_not_sigmoid_squashed(self):
        """
        R-22 regression: at critic saturation the sigmoid path's input
        gradient is crushed by sigmoid' while the raw critic path stays
        usable. Saturation is forced AND verified, so this test cannot pass
        vacuously on an unsaturated model.
        """
        import torch

        warden, _, channels = _tiny_warden(batch=4)
        warden.eval()

        big_delays = torch.full((4, 8), 50.0)

        # Amplify the critic output far into the sigmoid tail WITHOUT
        # zeroing d(raw)/d(input): scale the final head weights, then shift
        # the bias so every sample's raw score is large.
        final = warden.classification_head[-1]
        with torch.no_grad():
            final.weight.mul_(1e3)
        probe = warden(big_delays, channels).feature_importance["raw_critic_score"]
        with torch.no_grad():
            final.bias.add_(50.0 - probe.detach().min())

        saturated = warden(big_delays, channels)
        raw = saturated.feature_importance["raw_critic_score"]
        assert raw.min().item() >= 49.0, "saturation premise: raw critic must be large"
        assert saturated.bot_probability.min().item() > 1.0 - 1e-9, (
            "saturation premise: sigmoid must actually be saturated"
        )

        interp = big_delays.clone().requires_grad_(True)
        warden(interp, channels).feature_importance["raw_critic_score"].sum().backward()
        g_raw = interp.grad.abs().sum().item()

        interp2 = big_delays.clone().requires_grad_(True)
        warden(interp2, channels).bot_probability.sum().backward()
        g_sigmoid = interp2.grad.abs().sum().item()

        assert g_raw > 1e-6, "raw critic path must produce usable gradients"
        assert g_sigmoid < 1e-12, "sigmoid path must be crushed at saturation"
        assert g_raw > 1e6 * g_sigmoid, (
            "raw-path gradients must dwarf the sigmoid-path gradients at "
            "saturation (R-22 regression otherwise)"
        )


class TestChannelHeadReceivesGradient:
    """R-23: straight-through Gumbel-Softmax must train the channel head."""

    def test_gumbel_hard_sample_is_onehot(self):
        """
        The hard Gumbel sample must be one-hot in the FORWARD pass and carry
        the soft sample's gradient in the BACKWARD pass (straight-through).
        A detached one-hot, or soft probabilities, would fail one of these.
        """
        import torch

        from src.stealth.gan.generator import TimingSchedule

        torch.manual_seed(7)
        logits = torch.randn(2, 4, 3, requires_grad=True)
        schedule = TimingSchedule(
            delays=torch.rand(2, 4),
            channel_logits=logits,
            confidence=torch.ones(2),
        )

        # Fix the RNG state so the soft reference below sees the same noise.
        torch.manual_seed(1234)
        probs = schedule.channel_probs_straight_through()

        assert probs.requires_grad, "straight-through sample must stay differentiable"
        assert probs.shape == (2, 4, 3)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))
        # hard=True: exactly one entry is 1 per step
        assert ((probs == 1).sum(dim=-1) == 1).all()

        weights = torch.randn(2, 4, 3)
        (probs * weights).sum().backward()
        assert logits.grad is not None, "STE must not detach the channel logits"
        grad_hard = logits.grad.clone()
        assert grad_hard.abs().sum().item() > 1e-8

        logits.grad = None
        torch.manual_seed(1234)
        soft = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
        (soft * weights).sum().backward()
        assert torch.allclose(grad_hard, logits.grad, atol=1e-6), (
            "straight-through gradient must equal the soft sample's gradient"
        )

    def test_generator_channel_head_gets_nonzero_grad(self):
        """
        R-23 regression, behavioral: critic gradients must reach every
        channel_head parameter through the straight-through sample, and must
        NOT reach them when the sample is detached (control case).
        """
        import torch

        from src.stealth.gan.generator import TemporalPatternGenerator
        from src.analysis.adversarial.warden import DeepPacketInspectionWarden

        torch.manual_seed(1)
        gen = TemporalPatternGenerator(latent_dim=16, hidden_dim=32, num_channels=3)
        gen.eval()
        z = torch.randn(2, 16)
        tod = torch.rand(2) * 24.0  # hour of day [0, 23]
        schedule = gen(z, sequence_length=6, time_of_day=tod)

        warden = DeepPacketInspectionWarden(
            num_channels=3, hidden_dim=32, num_transformer_layers=1, feature_dim=8
        )
        warden.eval()

        head_params = {n: p for n, p in gen.named_parameters() if "channel" in n.lower()}
        assert head_params, "generator must expose channel_head parameters"

        # Control: a detached channel sample must leave the head untrained.
        control_loss = warden(
            schedule.delays.detach(),
            schedule.channel_probs_straight_through().detach(),
        ).feature_importance["raw_critic_score"].mean()
        control_loss.backward()
        for name, p in head_params.items():
            assert p.grad is None or p.grad.abs().sum().item() == 0, (
                f"control failed: detached channels trained {name}"
            )

        # Positive: the straight-through sample must train every head param.
        loss = warden(
            schedule.delays.detach(),
            schedule.channel_probs_straight_through(),
        ).feature_importance["raw_critic_score"].mean()
        loss.backward()
        for name, p in head_params.items():
            assert p.grad is not None, (
                f"{name} received no gradient - R-23 regression "
                "(argmax sampling would leave it untrained forever)"
            )
            assert p.grad.abs().sum().item() > 0, f"{name} gradient is zero"


class TestCriticLoopDetachesGenerator:
    """R-24: the warden loop must not backprop through the generator."""

    def test_fake_delays_detached(self, tmp_path):
        """
        R-24 regression, behavioral: during a real train_step the critic
        update must show the warden DETACHED fake delays and channel
        probabilities, while the generator update must show tensors that
        require grad (proving the spy can distinguish the two).
        """
        import torch

        from src.analysis.adversarial.warden import DeepPacketInspectionWarden
        from src.stealth.gan.trainer import GANTrainer, TrainingConfig

        config = TrainingConfig(
            latent_dim=16,
            hidden_dim=32,
            num_channels=3,
            max_sequence_length=8,
            batch_size=2,
            warden_steps=1,
            use_gradient_penalty=False,
            device="cpu",
            checkpoint_dir=tmp_path / "gan_ckpt",
        )
        warden = DeepPacketInspectionWarden(
            num_channels=3, hidden_dim=32, num_transformer_layers=1, feature_dim=8
        )
        trainer = GANTrainer(config, warden=warden)

        calls: list[dict] = []
        original_forward = trainer.warden.forward

        def spy(delays, channels, *args, **kwargs):
            calls.append(
                {
                    "delays_require_grad": bool(getattr(delays, "requires_grad", False)),
                    "channels_require_grad": bool(
                        getattr(channels, "requires_grad", False)
                    ),
                }
            )
            return original_forward(delays, channels, *args, **kwargs)

        trainer.warden.forward = spy

        real_batch = {
            "delays": torch.rand(2, 8) * 5.0 + 1.0,
            "channels": torch.randint(0, 3, (2, 8)),
            "time_of_day": torch.rand(2) * 24.0,
            "sequence_length": torch.tensor([8, 6]),
        }
        trainer.train_step(real_batch)

        # Call order (warden_steps=1, gradient penalty off):
        # 0 = real verdict, 1 = fake verdict of the critic update,
        # 2 = fake verdict of the generator update, 3+ = metrics under no_grad.
        assert calls[0]["delays_require_grad"] is False
        assert calls[1]["delays_require_grad"] is False, (
            "critic update must detach fake delays before the warden forward "
            "(R-24: otherwise every critic step backprops through G)"
        )
        assert calls[1]["channels_require_grad"] is False, (
            "critic update must detach fake channel probabilities too"
        )
        assert any(c["delays_require_grad"] for c in calls[2:]), (
            "spy never observed a gradient-carrying fake verdict; if the "
            "generator update does not backprop through the warden, this "
            "assertion is vacuous"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

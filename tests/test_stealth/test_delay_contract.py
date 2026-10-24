# tests/test_stealth/test_delay_contract.py
"""
Regression tests for the repaired generator path.

The generator had three structural defects, each measured in
`docs/benchmarking/C3_GENERATOR_ADVERSARY.md`:

  * the delay mapping left a freshly initialised generator with no
    gradient at all, because the usable delay range is a narrow slice of
    the delay head's output range and every output landed at the clip;
  * the critic was fed the generator's raw output while the real
    traffic was in seconds, so it never saw the traffic the scheduler
    sends;
  * the critic's score had no anchor, and the offset ran away.

These tests pin the fixes:

  * the generator's raw output is always inside the band the canonical
    mapping can use, so the mapping is never clipped;
  * the gradient that reaches the delay head through the mapping is
    never zero;
  * the trainer maps the generated traffic before the critic sees it;
  * the critic's scoring configuration survives a checkpoint round trip.
"""

import numpy as np
import pytest
import torch

from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.delay_contract import (
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    GAN_RAW_MAX,
    GAN_RAW_MIN,
    canonical_delay_mapping,
    canonical_delay_mapping_torch,
    raw_delay_band_transform,
)
from src.stealth.gan.generator import TemporalPatternGenerator
from src.stealth.gan.trainer import GANTrainer, TrainingConfig, apply_critic_config

SEQUENCE_LENGTH = 20


def _generator(seed: int = 0) -> TemporalPatternGenerator:
    torch.manual_seed(seed)
    return TemporalPatternGenerator(
        latent_dim=16,
        hidden_dim=32,
        num_channels=3,
        max_sequence_length=SEQUENCE_LENGTH,
    )


def test_the_usable_raw_band_is_inside_the_mapping_s_window():
    """The band the generator emits into must not be clipped."""
    assert GAN_RAW_MIN < GAN_RAW_MAX
    # The band's ends must map to the deployment window's ends.
    assert canonical_delay_mapping([GAN_RAW_MIN])[0] == pytest.approx(GAN_DELAY_MIN)
    assert canonical_delay_mapping([GAN_RAW_MAX])[0] == pytest.approx(GAN_DELAY_MAX)


def test_the_generator_emits_inside_the_usable_band():
    """A fresh generator's raw output must never fall outside the band."""
    generator = _generator()
    z = torch.randn(64, 16)
    tod = torch.full((64,), 14.0)
    with torch.no_grad():
        raw = generator(z, SEQUENCE_LENGTH, tod).delays
    assert float(raw.min()) >= GAN_RAW_MIN
    assert float(raw.max()) <= GAN_RAW_MAX


def test_the_delay_mapping_is_never_saturated_by_the_generator():
    """If the mapping is never clipped, no delay is pinned to a bound."""
    generator = _generator(seed=3)
    z = torch.randn(64, 16)
    tod = torch.full((64,), 14.0)
    with torch.no_grad():
        raw = generator(z, SEQUENCE_LENGTH, tod).delays
    mapped = canonical_delay_mapping(raw.numpy())
    assert float(mapped.min()) > GAN_DELAY_MIN
    assert float(mapped.max()) < GAN_DELAY_MAX


def test_the_gradient_reaches_the_delay_head():
    """The generator must be trainable through the canonical mapping."""
    generator = _generator(seed=5)
    z = torch.randn(32, 16, requires_grad=True)
    tod = torch.full((32,), 14.0)
    raw = generator(z, SEQUENCE_LENGTH, tod).delays
    mapped = canonical_delay_mapping_torch(raw)
    grad = torch.autograd.grad(mapped.sum(), raw, retain_graph=True)[0]
    assert float((grad == 0).float().mean()) == 0.0


def test_the_band_transform_is_monotone_and_bounded():
    """The transform folds the head's output into the band, in order."""
    head = torch.linspace(0.0, 6.0, 64)
    band = raw_delay_band_transform(head)
    # The comparison is in float32, so the band is reached to within the
    # representation's precision rather than exactly.
    assert float(band.min()) == pytest.approx(GAN_RAW_MIN, abs=1e-6)
    assert float(band.max()) <= GAN_RAW_MAX + 1e-6
    assert torch.all(band[1:] >= band[:-1])


class _RecordingCritic(torch.nn.Module):
    """A stand-in critic that records the delays it is given."""

    def __init__(self):
        super().__init__()
        self.seen: list[torch.Tensor] = []
        # A parameter, so the trainer's optimiser has something to hold.
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, delays, channels, mask=None):
        self.seen.append(delays.detach().clone())
        from src.analysis.adversarial.warden import WardenVerdict

        score = delays.sum(dim=1) * self.scale
        return WardenVerdict(
            bot_probability=torch.sigmoid(score),
            anomaly_scores=torch.zeros_like(delays),
            feature_importance={
                "anomaly_scores": torch.zeros_like(delays),
                "global_attention": torch.zeros_like(delays),
                "statistical_features": torch.zeros_like(delays),
                "raw_critic_score": score,
            },
        )


def test_the_trainer_maps_the_generated_traffic_before_the_critic():
    """The critic must see the delays the scheduler sends.

    The trainer used to give the critic the generator's raw output while
    the real traffic was in seconds, so the critic discriminated a unit
    range rather than a timing shape.
    """
    config = TrainingConfig(
        latent_dim=16,
        hidden_dim=32,
        num_channels=3,
        max_sequence_length=SEQUENCE_LENGTH,
        batch_size=4,
        device="cpu",
        use_gradient_penalty=False,
    )
    critic = _RecordingCritic()
    trainer = GANTrainer(config, warden=critic)

    delays = torch.rand(4, SEQUENCE_LENGTH) * 20.0 + 0.5
    batch = {
        "delays": delays,
        "channels": torch.randint(0, 3, (4, SEQUENCE_LENGTH)),
        "time_of_day": torch.full((4,), 14.0),
        "sequence_length": torch.full((4,), SEQUENCE_LENGTH),
    }
    trainer.train_step(batch)

    assert critic.seen, "the critic should have been called"
    # The first call is with the real traffic, the second with the
    # generated traffic.
    generated = critic.seen[1]
    assert float(generated.min()) >= GAN_DELAY_MIN
    assert float(generated.max()) <= GAN_DELAY_MAX
    # The generated traffic must not be the generator's raw output, which
    # lives in a band around 0.5 and is not in seconds.
    assert not torch.allclose(generated, torch.full_like(generated, 0.5))


def test_the_critics_scoring_configuration_survives_a_checkpoint():
    """A bounded critic must not be read as an unbounded one."""
    bounded = DeepPacketInspectionWarden(
        num_channels=3, hidden_dim=32, bounded_critic=True, critic_score_bound=7.0
    )
    checkpoint = {
        "warden_state": bounded.state_dict(),
        "critic_bounded": True,
        "critic_score_bound": 7.0,
    }
    loaded = DeepPacketInspectionWarden(num_channels=3, hidden_dim=32)
    loaded.load_state_dict(checkpoint["warden_state"])
    apply_critic_config(loaded, checkpoint)
    assert loaded.bounded_critic is True
    assert loaded.critic_score_bound == 7.0


def test_an_older_checkpoint_keeps_the_unbounded_default():
    """A checkpoint written before the bound existed is unbounded."""
    loaded = DeepPacketInspectionWarden(num_channels=3, hidden_dim=32)
    apply_critic_config(loaded, {"warden_state": {}})
    assert loaded.bounded_critic is False


def test_the_bounded_critic_cannot_run_away():
    """A bounded critic's score stays inside its bound."""
    warden = DeepPacketInspectionWarden(
        num_channels=3, hidden_dim=32, bounded_critic=True, critic_score_bound=5.0
    )
    delays = torch.rand(8, 20) * 100.0
    channels = torch.randint(0, 3, (8, 20))
    with torch.no_grad():
        verdict = warden(delays, channels)
    score = verdict.feature_importance["raw_critic_score"]
    assert float(score.abs().max()) <= 5.0


def test_the_two_critics_agree_on_small_scores():
    """The bound is close to the identity where the critic works."""
    unbounded = DeepPacketInspectionWarden(num_channels=3, hidden_dim=32).eval()
    bounded = DeepPacketInspectionWarden(
        num_channels=3, hidden_dim=32, bounded_critic=True, critic_score_bound=100.0
    ).eval()
    bounded.load_state_dict(unbounded.state_dict())
    delays = torch.rand(8, 20) * 10.0
    channels = torch.randint(0, 3, (8, 20))
    with torch.no_grad():
        a = unbounded(delays, channels).feature_importance["raw_critic_score"]
        b = bounded(delays, channels).feature_importance["raw_critic_score"]
    assert np.allclose(a.numpy(), b.numpy(), atol=1e-3)

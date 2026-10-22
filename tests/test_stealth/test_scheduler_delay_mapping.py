# tests/test_stealth/test_scheduler_delay_mapping.py
"""
Regression tests for the unified WGAN delay mapping (remediation item B1).

The deployed GAN path, the deployed hybrid path, and the Component 3
benchmark used to disagree about how a raw generator output becomes an
inter-item delay. The benchmark applied `(raw - 0.5) * 1500 + 1.5` while
the deployed GAN path returned the raw value, so the recorded benchmark row
did not describe the shipped path.

These tests pin the contract:
  * one canonical mapping function, shared by every consumer;
  * the deployed GAN path applies it;
  * the deployed hybrid path applies the same mapping;
  * a pinned hour reaches the generator;
  * the benchmark imports the scheduler's mapping instead of keeping a copy.

A stub generator is injected so the tests are deterministic and fast; the
real checkpoint is exercised by the C3 benchmark itself.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from src.stealth.gan.generator import TimingSchedule
from src.stealth.stealth_scheduler import (
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    StealthScheduler,
    canonical_delay_mapping,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _StubGenerator:
    """Minimal stand-in for TemporalPatternGenerator.

    Returns fixed raw delays so the mapping under test is the only thing
    that determines the scheduled delays.
    """

    latent_dim = 4

    def __init__(self, raw_delays=None):
        self._raw = (
            np.array([0.5010, 0.5020, 0.5030, 0.5015, 0.5040], dtype=np.float32)
            if raw_delays is None
            else np.asarray(raw_delays, dtype=np.float32)
        )
        self.seen_hours = []

    def eval(self):
        return self

    def generate(self, batch_size=1, sequence_length=20, time_of_day=None, device="cpu"):
        hour = float(time_of_day.reshape(-1)[0].item())
        self.seen_hours.append(hour)
        n = sequence_length
        delays = np.resize(self._raw, n)
        return TimingSchedule(
            delays=torch.tensor(delays, dtype=torch.float32).unsqueeze(0),
            channel_logits=torch.zeros(1, n, 3, dtype=torch.float32),
            confidence=torch.zeros(1, dtype=torch.float32),
        )


def _stub_scheduler(raw_delays=None) -> tuple[StealthScheduler, _StubGenerator]:
    scheduler = StealthScheduler(num_channels=3, device="cpu")
    generator = _StubGenerator(raw_delays)
    # _load_generator short-circuits when a generator is already present.
    scheduler._generator = generator
    return scheduler, generator


# ----------------------------------------------------------------------
# The mapping itself
# ----------------------------------------------------------------------


def test_canonical_mapping_matches_documented_formula():
    raw = np.array([0.5, 0.501, 0.51, 0.6], dtype=np.float64)
    expected = np.clip((raw - 0.5) * 1500.0 + 1.5, 0.5, 30.0)
    np.testing.assert_allclose(canonical_delay_mapping(raw), expected)


def test_canonical_mapping_clips_to_bounds():
    raw = np.array([-10.0, 0.5, 100.0], dtype=np.float64)
    mapped = canonical_delay_mapping(raw)
    assert mapped.min() >= GAN_DELAY_MIN
    assert mapped.max() <= GAN_DELAY_MAX


def test_canonical_mapping_is_monotonic():
    raw = np.linspace(0.5, 0.6, 50)
    mapped = canonical_delay_mapping(raw)
    assert np.all(np.diff(mapped) >= 0)


def test_canonical_mapping_expands_the_generator_band():
    """The generator's raw output is concentrated just above 0.5. Without
    the mapping that band collapses to a near-constant half-second
    cadence, which is the periodic rhythm the warden is built to catch."""
    raw = np.array([0.5010, 0.5020, 0.5030])
    mapped = canonical_delay_mapping(raw)
    # A 2 ms spread in generator units must not survive as a 2 ms spread
    # in the deployed schedule.
    assert mapped.max() - mapped.min() > 1.0
    assert np.all(mapped >= 1.5)


# ----------------------------------------------------------------------
# The deployed paths
# ----------------------------------------------------------------------


def test_deployed_gan_path_applies_the_canonical_mapping():
    """The deployed GAN schedule must be the mapped delays, not the raw
    generator output."""
    scheduler, _ = _stub_scheduler()
    media_ids = [f"doc_{i}" for i in range(5)]

    schedule = scheduler.schedule(
        media_ids, mode="gan", time_of_day=12.0
    )

    assert schedule["mode_used"] == "gan"
    raw = np.array(schedule["raw_delays"], dtype=np.float64)
    mapped = canonical_delay_mapping(raw)
    np.testing.assert_allclose(
        np.array(schedule["delays"], dtype=np.float64), mapped, rtol=1e-9
    )


def test_deployed_gan_path_is_not_a_constant_half_second_cadence():
    """Regression guard for the original bug: the deployed GAN path used to
    emit the raw generator values, i.e. an almost-constant 0.5 s cadence."""
    scheduler, _ = _stub_scheduler()
    media_ids = [f"doc_{i}" for i in range(5)]

    schedule = scheduler.schedule(media_ids, mode="gan", time_of_day=12.0)
    delays = np.array(schedule["delays"], dtype=np.float64)

    assert delays.min() >= GAN_DELAY_MIN
    assert delays.max() <= GAN_DELAY_MAX
    # A raw band of a few milliseconds must not survive as a
    # sub-millisecond spread in the deployed schedule.
    assert delays.std() > 1e-3
    # And the raw values themselves were in the narrow 0.5 s band.
    raw = np.array(schedule["raw_delays"], dtype=np.float64)
    assert raw.std() < 0.01


def test_pinned_hour_reaches_the_generator():
    """A pinned hour-of-day must be forwarded to the generator, so a
    benchmark can compare two hours under one identical mapping."""
    scheduler, generator = _stub_scheduler()
    media_ids = [f"doc_{i}" for i in range(5)]

    scheduler.schedule(media_ids, mode="gan", time_of_day=14.0)
    scheduler.schedule(media_ids, mode="gan", time_of_day=3.0)

    assert generator.seen_hours == [14.0, 3.0]


def test_hybrid_path_applies_the_same_mapping():
    """The hybrid path must use the canonical mapping too, so the two
    WGAN-based modes cannot drift apart."""
    scheduler, _ = _stub_scheduler()
    media_ids = [f"doc_{i}" for i in range(5)]

    schedule = scheduler.schedule(media_ids, mode="hybrid", time_of_day=12.0)
    assert schedule["mode_used"] == "hybrid"
    delays = np.array(schedule["delays"], dtype=np.float64)
    assert delays.min() >= GAN_DELAY_MIN
    assert delays.max() <= GAN_DELAY_MAX


def test_benchmark_shares_the_scheduler_mapping():
    """The Component 3 benchmark must import the scheduler's mapping
    rather than keeping its own copy of the formula."""
    script = _PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
    spec = importlib.util.spec_from_file_location("bench_c3", script)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules, so the
    # module has to be registered before it is executed.
    sys.modules["bench_c3"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("bench_c3", None)

    raw = np.array([0.5005, 0.5015, 0.5025], dtype=np.float32)
    np.testing.assert_allclose(
        module.canonical_delay_mapping(raw),
        canonical_delay_mapping(raw).astype(np.float32),
        rtol=1e-6,
    )


def test_schedule_contract_holds_for_wgan_modes():
    """Adding the raw-delay diagnostic key must not break the payload
    contract the validator enforces."""
    for mode in ("gan", "hybrid"):
        scheduler, _ = _stub_scheduler()
        media_ids = [f"doc_{i}" for i in range(6)]

        schedule = scheduler.schedule(media_ids, mode=mode, time_of_day=9.0)
        assert len(schedule["items"]) == 6
        assert len(schedule["delays"]) == 6
        assert len(schedule["channels"]) == 6
        assert all(d >= 0 for d in schedule["delays"])

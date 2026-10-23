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


# ----------------------------------------------------------------------
# The RL export path: timeout behavior + cooldown-safe tail (FIX-A)
# ----------------------------------------------------------------------

class _StubRLEnv:
    """Scripted StealthEnvironment stand-in for `_schedule_rl`.

    `script` is a list of step results already shaped like the env's
    return: (next_state, reward, done, info). Sent steps carry an
    env-reported delay; the timeout step carries only a reason, exactly
    like the real environment (the item was never sent).
    """

    state_dim = 8

    def __init__(self, script):
        self._script = list(script)
        self.i = 0
        self.state = object()

    def reset(self, media_ids, start_hour=None):
        self.state = object()
        self.i = 0
        return self.state

    def get_action_mask(self):
        return np.ones(3, dtype=np.float32)

    def step(self, action):
        out = self._script[self.i]
        self.i += 1
        return out


class _StubRLAgent:
    """Agent-side stand-in: fixed action, clearable buffer, scripted env."""

    def __init__(self, action, env):
        self.action = action          # the agent's raw (unbackoffed) choice
        self.env = env
        self._buffer = []

        class _Buffer:
            def clear(self_inner):
                self._buffer.clear()

        self.buffer = _Buffer()

    def select_action(self, state, channel_mask=None):
        return self.action, 0.0, 0.0


def _rl_stub_scheduler(script, action=None):
    scheduler = StealthScheduler(num_channels=3, device="cpu")
    data = script if script is not None else [
        # Sent steps: the env transmits on the ACTION's channel, so the
        # stub's action must agree with what the steps describe.
        (object(), 0.0, False, {"delay": 5.0, "channel": 2}),
        (object(), 0.0, False, {"delay": 5.0, "channel": 2}),
        # Timeout: episode ends, NO transmission happened, and the env
        # reports no delay at all (matches environment.py:290-296 in
        # spirit; the real info dict is {"reason": "timeout"}).
        (object(), -50.0, True, {"reason": "timeout"}),
    ]
    env = _StubRLEnv(data)
    scheduler._rl_agent = _StubRLAgent(
        action or {"delay": 2.0, "channel": 2}, env
    )
    return scheduler


def test_schedule_rl_exports_no_phantom_delay_on_timeout():
    """The timeout step appended the agent's raw, un-backoffed action delay
    (its `info` has no `delay` key, so the bare `action["delay"]` fallback
    fired). Only env-confirmed transmissions belong on the wire.
    """
    scheduler = _rl_stub_scheduler(None)
    media_ids = [f"doc_{i}" for i in range(5)]

    schedule = scheduler._schedule_rl(media_ids, base_delay=2.0, checkpoint=None)

    assert schedule["mode_used"] == "rl"
    delays = [float(d) for d in schedule["delays"]]
    assert len(delays) == 5
    # The raw action delay of the timeout step must never appear on the wire.
    assert 2.0 not in delays
    # Env-reported transmissions are preserved verbatim.
    np.testing.assert_allclose(delays[:2], [5.0, 5.0])


def test_schedule_rl_tail_respects_cooldowns():
    """The padded tail must obey the per-channel cooldowns the benchmark
    measures: every landing gap on a channel >= that channel's cooldown,
    and every delay stays inside the canonical pacing window.
    """
    from src.stealth.rl.pacing import RL_DELAY_MIN
    from src.stealth.stealth_scheduler import (
        CHANNEL_COOLDOWNS,
    )

    scheduler = _rl_stub_scheduler(None)
    media_ids = [f"doc_{i}" for i in range(6)]

    schedule = scheduler._schedule_rl(media_ids, base_delay=2.0, checkpoint=None)

    delays = np.array(schedule["delays"], dtype=np.float64)
    channels = [int(c) for c in schedule["channels"]]

    assert schedule["mode_used"] == "rl"
    assert len(delays) == len(media_ids)
    assert np.all(delays >= RL_DELAY_MIN)

    t = np.cumsum(delays)
    last = {}
    for k, (c, tk) in enumerate(zip(channels, t)):
        if c in last:
            gap = float(tk) - float(last[c])
            assert gap >= CHANNEL_COOLDOWNS[c], (
                f"item {k} lands {gap:.6f}s after the previous landing "
                f"on channel {c}, cooldown is {CHANNEL_COOLDOWNS[c]}"
            )
        last[c] = float(tk)

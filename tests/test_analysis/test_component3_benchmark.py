"""Tests for the component 3 warden decision function.

Properties pinned here are the ones the audit left implicit in the legacy
record: the rule priority order, the soft branch's ability to reach the
flag threshold, rate-limit diagnostics that are recorded but never scored,
and the log-bin entropy recalibration that stopped the warden flagging its
own surrogate reference.

The soft branch changed shape in Phase 1. It used to be
`clip(0.10 * anomaly + 0.35 * ks, 0, 0.45)` against a 0.50 flag
threshold, so its maximum was 0.45 and it could never flag any traffic for
any input: the recorded 0.0 percent interception for the WGAN and Hybrid
rows was a property of the arithmetic, not of the traffic. It is now the
trained detector's bot probability, against a threshold calibrated on
held-out human traffic. The KS statistic is still recorded per trial but is
not part of the score.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "benchmark_3_traffic_evasion.py"
)


def _load_module():
    name = "benchmark_3_traffic_evasion"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _StubVerdict:
    def __init__(self, value: float):
        self.anomaly_scores = torch.full((1, 4), value)
        self.bot_probability = torch.full((1,), value)


class _StubWarden:
    """Constant detector, mirroring a detector that sees one thing only."""

    def __init__(self, value: float = 0.62493):
        self.value = value

    def __call__(self, delays, channels, mask=None):
        return _StubVerdict(self.value)


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(mod, "DEVICE", "cpu")
    yield


def _human(n: int = 500) -> np.ndarray:
    return mod.generate_human_baseline_delays(n)


def _soft_branch_delays() -> np.ndarray:
    """Non-periodic, non-burst delays spanning five log-bin magnitudes."""
    pattern = np.array([1.6, 2.8, 4.1, 5.5, 6.9, 3.2])
    return np.concatenate([np.tile(pattern, 8), pattern[:2]])


def test_constant_clock_triggers_periodic_cv():
    res = mod.evaluate_dpi_warden(
        np.full(50, 3.0), np.arange(50) % 3, _StubWarden(), _human()
    )
    assert res["rule"] == "periodic_cv"
    assert res["score"] == 0.98
    assert res["flagged"] is True
    assert res["cv"] == 0.0


def test_low_log_entropy_nonconstant_triggers_periodic_entropy():
    delays = np.concatenate([np.full(48, 3.0), np.full(2, 100.0)])
    res = mod.evaluate_dpi_warden(
        delays, np.arange(50) % 3, _StubWarden(), _human()
    )
    assert res["rule"] == "periodic_entropy"
    assert res["score"] == 0.98
    assert res["cv"] > 0.20
    assert res["entropy_log"] < 1.0


def test_periodic_rule_precedes_burst():
    res = mod.evaluate_dpi_warden(
        np.full(50, 0.5), np.arange(50) % 3, _StubWarden(), _human()
    )
    assert res["max_burst"] >= 6
    assert res["rule"] == "periodic_cv"
    assert res["score"] == 0.98


def test_burst_rule_scores_between_floor_and_cap():
    delays = np.concatenate(
        [np.full(12, 0.2), np.full(12, 0.8), np.full(13, 8.0), np.full(13, 80.0)]
    )
    res = mod.evaluate_dpi_warden(
        delays, np.arange(50) % 3, _StubWarden(), _human()
    )
    assert res["rule"] == "burst"
    assert res["cv"] >= 0.20
    assert res["entropy_log"] >= 1.0
    assert 0.65 <= res["score"] <= 0.95
    assert res["flagged"] is True


def test_single_channel_hammer_scores_one():
    res = mod.evaluate_dpi_warden(
        _soft_branch_delays(), np.zeros(50, dtype=np.int64), _StubWarden(), _human()
    )
    assert res["rule"] == "hammer"
    assert res["score"] == 1.0
    assert res["flagged"] is True


def test_soft_branch_can_reach_the_flag_threshold():
    """The old branch was capped at 0.45 against a 0.50 threshold, so it
    could never flag anything. The branch must now be able to."""
    # A ramp is non-periodic, non-bursting, and multi-channel, so it reaches
    # the soft branch.
    delays = np.linspace(0.5, 200.0, 50)
    res = mod.evaluate_dpi_warden(
        delays, np.arange(50) % 3, _StubWarden(0.99), _human()
    )
    assert res["rule"] == "soft"
    # The score is the trained detector's bot probability.
    assert res["score"] == pytest.approx(res["bot_probability"], abs=1e-9)
    assert res["score"] > 0.45, "the branch must be able to exceed the old cap"
    assert res["flagged"] is True


def test_soft_branch_score_is_bounded_by_one():
    """The detector's bot probability is a probability, so the score cannot
    exceed 1."""
    res = mod.evaluate_dpi_warden(
        _soft_branch_delays(), np.arange(50) % 3, _StubWarden(1.0), _human()
    )
    assert res["score"] <= 1.0


def test_ks_statistic_is_recorded_but_does_not_score():
    """The KS statistic has a large null: a burst drawn from the reference
    itself scores about 0.12, and a whole human session about 0.28. It is
    recorded per trial and reported in the table, but it is not part of
    the score, because adding it to a calibrated detector dilutes the
    detector."""
    res_a = mod.evaluate_dpi_warden(
        _soft_branch_delays(), np.arange(50) % 3, _StubWarden(0.30), _human()
    )
    res_b = mod.evaluate_dpi_warden(
        np.linspace(0.5, 200.0, 50), np.arange(50) % 3, _StubWarden(0.30), _human()
    )
    # The two differ in their KS statistic...
    assert res_a["ks_stat"] != res_b["ks_stat"]
    # ...but the soft-branch score depends only on the detector.
    assert res_a["score"] == pytest.approx(res_b["score"], abs=1e-9)


def test_rate_limit_violations_are_recorded_and_not_scored():
    # Delays cycle [3, 4, 5] so that every channel reuse gap equals 12.0 s,
    # clearing the per-channel cooldowns {0: 6.0, 1: 12.0, 2: 4.0}. The dirty
    # channel pattern pairs each channel back to back instead.
    delays = np.concatenate([np.tile(np.array([3.0, 4.0, 5.0, 3.0, 4.0, 5.0]), 8), [3.0, 4.0]])
    clean = np.concatenate([np.tile(np.array([0, 1, 2, 0, 1, 2]), 8), [0, 1]])
    dirty = np.concatenate([np.tile(np.array([0, 0, 1, 1, 2, 2]), 8), [0, 1]])
    res_clean = mod.evaluate_dpi_warden(delays, clean, _StubWarden(), _human())
    res_dirty = mod.evaluate_dpi_warden(delays, dirty, _StubWarden(), _human())
    assert res_clean["violations"] == 0
    assert res_dirty["violations"] > 0
    assert res_clean["rule"] == res_dirty["rule"] == "soft"
    assert res_clean["score"] == res_dirty["score"]


def test_constant_clock_scores_zero_entropy_under_both_estimators():
    delays = np.full(50, 3.0)
    assert mod.compute_delay_entropy(delays) == 0.0
    assert mod.compute_log_bin_entropy(delays) == 0.0


def test_log_bin_entropy_separates_scales_that_linear_bins_collapse():
    delays = np.concatenate(
        [np.full(17, 0.05), np.full(16, 3.0), np.full(17, 3000.0)]
    )
    assert mod.compute_log_bin_entropy(delays) > mod.compute_delay_entropy(delays)


def test_surrogate_entropy_reads_higher_under_log_bins():
    human = mod.generate_human_baseline_delays(2000)
    assert mod.compute_log_bin_entropy(human) > mod.compute_delay_entropy(human)

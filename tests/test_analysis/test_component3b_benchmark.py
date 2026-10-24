"""Tests for the component 3b warden score summarizer.

The 3b record turns the stored per-trial scores into quantiles and flagged
counts, and labels the earlier recorded PPO run as a historical artifact. The
soft branch's threshold is calibrated per run, so 3b reads it from the
Component 3 artifact rather than assuming one, and the historical row is
judged against the 0.50 threshold that was in force when it was recorded.
These tests pin the threshold semantics and the config labeling so a rerun
cannot silently mix the two configurations.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "benchmark_3b_warden_score_distributions.py"
)


def _load_module():
    name = "benchmark_3b_warden_score_distributions"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_summarize_reports_quantiles_and_threshold_flags():
    s = mod.summarize("primary", "X", [0.1, 0.2, 0.3, 0.6, 0.7, 0.9], 3.5)
    assert s.num_trials == 6
    assert s.minimum == 0.1
    assert s.maximum == 0.9
    assert s.median == pytest.approx(0.45)
    assert s.p25 == pytest.approx(0.225)
    assert s.p75 == pytest.approx(0.675)
    assert s.flagged == 3
    assert s.interception_rate_pct == 50.0
    assert s.mean_latency_s == 3.5


def test_flag_boundary_is_inclusive():
    s = mod.summarize("primary", "X", [0.499, 0.5], 1.0)
    assert s.flagged == 1


def test_load_summaries_labels_historical_operating_point(tmp_path, monkeypatch):
    primary = {
        "schedulers": [
            {"scheduler": name, "warden_scores": [0.2] * 50, "mean_latency": 3.0}
            for name in [
                "Static Cadence",
                "Stochastic Poisson",
                "DCASS Temporal WGAN",
                "DCASS Hybrid Engine",
                "DCASS PPO RL Agent",
            ]
        ]
    }
    baseline = {
        "schedulers": [
            {
                "scheduler": "DCASS PPO RL Agent",
                "warden_scores": [0.3] * 49 + [0.98],
                "mean_latency": 44.07,
            },
            {"scheduler": "Static Cadence", "warden_scores": [0.98] * 50, "mean_latency": 3.0},
        ]
    }
    (tmp_path / "component_3_results.json").write_text(json.dumps(primary))
    (tmp_path / "component_3_results_baseline.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(mod, "BENCHMARK_DIR", tmp_path)

    summaries = mod.load_summaries()
    assert len(summaries) == 6
    assert [s.config for s in summaries] == ["primary"] * 5 + ["historical"]
    historical = summaries[-1]
    assert historical.scheduler == "PPO RL Agent, conservative config"
    assert historical.mean_latency_s == 44.07
    assert historical.flagged == 1
    # The historical row keeps the 0.50 threshold that was in force when it
    # was recorded.
    assert historical.threshold == mod.FLAG_THRESHOLD
    # The primary rows use the threshold recorded in the artifact.
    assert all(s.threshold == mod.FLAG_THRESHOLD for s in summaries[:5])


def test_primary_threshold_is_read_from_the_artifact(tmp_path, monkeypatch):
    """The soft branch's threshold is calibrated per run, so 3b must not
    assume it. An artifact that records a calibrated threshold must be
    honoured."""
    threshold = 0.1923
    primary = {
        "soft_branch_calibration": {"threshold": threshold},
        "schedulers": [
            {
                "scheduler": name,
                "warden_scores": [0.1] * 50,
                "mean_latency": 3.0,
            }
            for name in ["Static Cadence", "DCASS PPO RL Agent"]
        ],
    }
    baseline = {"schedulers": []}
    (tmp_path / "component_3_results.json").write_text(json.dumps(primary))
    (tmp_path / "component_3_results_baseline.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(mod, "BENCHMARK_DIR", tmp_path)

    assert mod.primary_flag_threshold(primary) == threshold
    summaries = mod.load_summaries()
    assert all(s.threshold == threshold for s in summaries)
    # 0.1 is below the calibrated threshold, so nothing is flagged.
    assert all(s.flagged == 0 for s in summaries)

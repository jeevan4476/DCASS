"""Tests for the component 2b benchmark protocol (schema v2).

Properties pinned here are the ones the audit found missing in the legacy
artifact: the mechanism null must replay the observed chunk schedule instead
of a plain multinomial draw, chi-square and max-z must carry permutation
calibrated p-values, and repeat rates must be measured with entropy on
unique media.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "benchmark_2b_transmitted_cluster_usage.py"
)


def _load_module():
    name = "benchmark_2b_transmitted_cluster_usage"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()
NUM_CLUSTERS = mod.NUM_CLUSTERS


def _family_probs_for_test():
    """Two families: {0,1} with 0.75/0.25, and {2,3} with 0.5/0.5."""
    cum_0 = np.zeros(NUM_CLUSTERS)
    cum_0[0] = 0.75
    cum_0[1:] = 1.0
    cum_1 = np.zeros(NUM_CLUSTERS)
    cum_1[2] = 0.25
    cum_1[3:] = 1.0
    return {"image": (cum_0, cum_1), "audio": (cum_0, cum_1)}


class TestStructuredSimulation:
    def test_replays_schedule_and_stays_inside_families(self):
        rng = np.random.default_rng(0)
        counts = mod.simulate_structured_counts(
            ["image"] * 4, _family_probs_for_test(), reps=50, rng=rng
        )
        assert counts.shape == (50, NUM_CLUSTERS)
        assert np.all(counts.sum(axis=1) == 4)
        assert counts[:, 4:].sum() == 0

    def test_density_weighting_is_respected(self):
        rng = np.random.default_rng(1)
        counts = mod.simulate_structured_counts(
            ["image"] * 2000, _family_probs_for_test(), reps=1, rng=rng
        )
        # Family 0 is picked half the time; inside it cluster 0 has 0.75 mass.
        # Family 1 contributes clusters 2 and 3 at 0.5 each.
        assert counts[0, 0] > counts[0, 1]
        assert counts[0, 2] > 0 and counts[0, 3] > 0

    def test_deterministic_given_seed(self):
        first = mod.simulate_structured_counts(
            ["image", "audio"] * 10, _family_probs_for_test(), 5, np.random.default_rng(7)
        )
        second = mod.simulate_structured_counts(
            ["image", "audio"] * 10, _family_probs_for_test(), 5, np.random.default_rng(7)
        )
        assert np.array_equal(first, second)

    def test_empty_schedule_gives_zero_counts(self):
        rng = np.random.default_rng(0)
        counts = mod.simulate_structured_counts([], _family_probs_for_test(), reps=3, rng=rng)
        assert counts.shape == (3, NUM_CLUSTERS)
        assert counts.sum() == 0

    def test_missing_family_distribution_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="family distribution"):
            mod.simulate_structured_counts(["text"], _family_probs_for_test(), 2, rng)


class TestNullStatistics:
    def _setup(self):
        rng = np.random.default_rng(3)
        expected = np.full(NUM_CLUSTERS, 4.0)
        reps = rng.multinomial(1024, np.ones(NUM_CLUSTERS) / NUM_CLUSTERS, size=100)
        return reps, expected

    def test_entropy_band_matches_rowwise_entropy(self):
        reps, expected = self._setup()
        stats = mod.null_statistics(reps, expected, chi2_observed=0.0, max_z_observed=0.0)
        row_entropies = mod.entropy_bits_rows(reps)
        assert stats["entropy_mean"] == pytest.approx(float(row_entropies.mean()))
        assert stats["entropy_p2_5"] <= stats["entropy_mean"] <= stats["entropy_p97_5"]

    def test_permutation_pvalues_are_in_range_and_monotone(self):
        reps, expected = self._setup()
        observed_chi2 = float(((reps.mean(axis=0) - expected) ** 2 / expected).sum())
        low = mod.null_statistics(reps, expected, observed_chi2, max_z_observed=0.0)
        high = mod.null_statistics(reps, expected, observed_chi2 + 10_000.0, max_z_observed=99.0)
        assert low["chi2_perm_pvalue"] > high["chi2_perm_pvalue"]
        assert low["max_z_perm_pvalue"] > high["max_z_perm_pvalue"]
        for key in ("chi2_perm_pvalue", "max_z_perm_pvalue"):
            assert 0.0 < low[key] <= 1.0
            assert 0.0 < high[key] <= 1.0

    def test_entropy_bits_rows_matches_scalar_entropy(self):
        rows = np.array([[1, 1, 2, 0], [4, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)
        row_entropies = mod.entropy_bits_rows(rows)
        for i in range(rows.shape[0]):
            assert row_entropies[i] == pytest.approx(mod.entropy_bits(rows[i]))


class TestRepeatStatistics:
    def test_counts_slots_unique_and_within_message_repeats(self):
        records = [
            {"message": 0, "modality": "image", "media_id": "a", "cluster_id": 0},
            {"message": 0, "modality": "image", "media_id": "a", "cluster_id": 0},
            {"message": 1, "modality": "image", "media_id": "b", "cluster_id": 1},
            {"message": 1, "modality": "text", "media_id": "c", "cluster_id": 2},
        ]
        stats = mod.repeat_statistics(records)
        assert stats["slots_total"] == 4
        assert stats["unique_media_total"] == 3
        assert stats["reuse_factor_total"] == pytest.approx(4 / 3)
        assert stats["within_message_repeat_slots"] == 1
        assert stats["max_uses_total"] == 2
        assert stats["repeated_media_total"] == 1
        assert stats["per_modality"]["image"]["unique_media"] == 2
        assert stats["per_modality"]["text"]["reuse_factor"] == pytest.approx(1.0)


class TestMechanismExpectationGuards:
    def test_empty_family_raises(self):
        class _Mapper:
            def symbol_for_media_id(self, media_id):
                return None

        class _Index:
            metadata = {"image": [{"id": "a"}], "text": [], "audio": []}

        class _Engine:
            _vcp_mapper = _Mapper()
            index = _Index()

        with pytest.raises(RuntimeError, match="fail-loud"):
            mod.build_mechanism_expectation(
                _Engine(), {"image": 10, "text": 0, "audio": 0}
            )

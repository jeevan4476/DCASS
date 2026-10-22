"""Tests for the component 2 benchmark protocol (schema v2).

Properties pinned here are the ones the audit found violated in the legacy
artifact: the reference sample must be disjoint from every carrier media id,
the multivariate tests must be permutation calibrated, the KS family must
match scipy per dimension and carry multiplicity control, and the fail-loud
ledger must refuse to bless a run with silent drops.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "benchmark_2_semantic_distribution.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_2_semantic_distribution", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _pooled_matrices(sample_a, sample_b):
    """Pooled distance and RBF kernel matrices with group A first."""
    pooled = np.vstack([sample_a, sample_b]).astype(np.float64)
    D = torch.from_numpy(cdist(pooled, pooled))
    squared = D**2
    sigma2 = float(squared.median())
    K = torch.exp(-squared / (2.0 * sigma2))
    return D, K


class TestEnergyDistanceAndMMD:
    def test_energy_matches_reference_u_statistic(self):
        rng = np.random.default_rng(3)
        sample_a = rng.normal(size=(60, 5))
        sample_b = rng.normal(size=(40, 5)) + 0.15
        D, K = _pooled_matrices(sample_a, sample_b)
        energy, _ = mod.energy_mmd_observed(D, K, n_a=len(sample_a))
        reference = (
            2.0 * cdist(sample_a, sample_b).mean()
            - cdist(sample_a, sample_a).sum() / (60 * 59)
            - cdist(sample_b, sample_b).sum() / (40 * 39)
        )
        assert energy == pytest.approx(float(reference), rel=1e-9)

    def test_mmd_matches_reference_u_statistic(self):
        rng = np.random.default_rng(23)
        sample_a = rng.normal(size=(50, 6))
        sample_b = rng.normal(size=(70, 6)) + 0.2
        D, K = _pooled_matrices(sample_a, sample_b)
        _, mmd = mod.energy_mmd_observed(D, K, n_a=len(sample_a))
        sigma2 = float((D**2).median())
        k_aa = np.exp(-cdist(sample_a, sample_a) ** 2 / (2 * sigma2))
        np.fill_diagonal(k_aa, 0.0)
        k_bb = np.exp(-cdist(sample_b, sample_b) ** 2 / (2 * sigma2))
        np.fill_diagonal(k_bb, 0.0)
        k_ab = np.exp(-cdist(sample_a, sample_b) ** 2 / (2 * sigma2))
        reference = (
            k_aa.sum() / (50 * 49)
            + k_bb.sum() / (70 * 69)
            - 2 * k_ab.sum() / (50 * 70)
        )
        assert mmd == pytest.approx(float(reference), rel=1e-9)

    def test_permutation_null_is_not_extreme(self):
        rng = np.random.default_rng(5)
        sample_a = rng.normal(size=(40, 4))
        sample_b = rng.normal(size=(40, 4))
        D, K = _pooled_matrices(sample_a, sample_b)
        result = mod.two_sample_permutation_test(D, K, n_a=40, n_perm=200, seed=7)
        assert result["energy"]["p_perm"] > 0.05
        assert result["mmd"]["p_perm"] > 0.05

    def test_permutation_under_separation_is_minimal(self):
        rng = np.random.default_rng(7)
        sample_a = rng.normal(size=(40, 4))
        sample_b = sample_a + 3.0
        D, K = _pooled_matrices(sample_a, sample_b)
        result = mod.two_sample_permutation_test(D, K, n_a=40, n_perm=100, seed=11)
        assert result["energy"]["p_perm"] == pytest.approx(1.0 / 101.0)
        assert result["mmd"]["p_perm"] == pytest.approx(1.0 / 101.0)

    def test_permutation_is_deterministic_given_seed(self):
        rng = np.random.default_rng(13)
        sample_a = rng.normal(size=(30, 3))
        sample_b = rng.normal(size=(30, 3))
        D, K = _pooled_matrices(sample_a, sample_b)
        first = mod.two_sample_permutation_test(D, K, n_a=30, n_perm=50, seed=17)
        second = mod.two_sample_permutation_test(D, K, n_a=30, n_perm=50, seed=17)
        assert first == second


class TestKSFamily:
    def test_per_dimension_matches_scipy_with_tie_blocks(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(size=(80, 6))
        carriers = rng.normal(size=(50, 6))
        carriers[10:20] = carriers[0:10]
        family = mod.KSFamPermutation(
            torch.from_numpy(reference), torch.from_numpy(carriers)
        )
        per_dimension = family.per_dimension().numpy()
        for dim in range(reference.shape[1]):
            stat, _ = ks_2samp(reference[:, dim], carriers[:, dim])
            # The float32 statistic rounds at ~1e-8; tie blocks must still match exactly.
            assert per_dimension[dim] == pytest.approx(stat, abs=1e-6)

    def test_familywise_p_rejects_shift(self):
        rng = np.random.default_rng(11)
        reference = rng.normal(size=(60, 4))
        carriers = rng.normal(size=(60, 4)) + 1.5
        family = mod.KSFamPermutation(torch.from_numpy(reference), torch.from_numpy(carriers))
        result = family.run(n_perm=50, seed=3)
        assert result["dims"] == 4
        assert result["familywise_max"] == pytest.approx(
            float(result["per_dimension"].max())
        )
        assert result["familywise_p"] == pytest.approx(1.0 / 51.0)

    def test_familywise_null_is_not_extreme(self):
        rng = np.random.default_rng(13)
        reference = rng.normal(size=(60, 4))
        carriers = rng.normal(size=(60, 4))
        family = mod.KSFamPermutation(torch.from_numpy(reference), torch.from_numpy(carriers))
        result = family.run(n_perm=100, seed=19)
        assert result["familywise_p"] > 0.05


class TestBenjaminiHochberg:
    def test_known_example(self):
        p_values = np.array([0.001, 0.02, 0.03, 0.9])
        reject, count, qvals = mod.bh_fdr(p_values, q=0.05)
        assert count == 3
        assert reject.tolist() == [True, True, True, False]
        assert qvals.shape == p_values.shape

    def test_qvalues_are_monotone_in_p_order(self):
        p_values = np.array([0.04, 0.01, 0.03, 0.02, 0.5])
        _, _, qvals = mod.bh_fdr(p_values, q=0.05)
        order = np.argsort(p_values)
        assert np.all(np.diff(qvals[order]) >= -1e-12)
        assert np.all(qvals <= 1.0)

    def test_rejects_nothing_when_all_pvalues_large(self):
        reject, count, _ = mod.bh_fdr(np.full(10, 0.8), q=0.05)
        assert count == 0
        assert not reject.any()


class TestDisjointOffsets:
    def test_excludes_carrier_ids_and_missing_ids(self):
        media_ids = ["a", "b", None, "c", "d", "e"]
        rng = np.random.default_rng(0)
        chosen, stats = mod.disjoint_offsets(media_ids, {"b", "d"}, sample_size=2, rng=rng)
        assert set(chosen.tolist()).issubset({0, 3, 5})
        assert stats["n_total"] == 6
        assert stats["n_missing_id"] == 1
        assert stats["n_excluded_carrier_media"] == 2
        assert stats["n_candidates"] == 3
        assert len(stats["offsets_sha256_16"]) == 16

    def test_raises_when_candidate_pool_is_too_small(self):
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError):
            mod.disjoint_offsets(["a", "b", None], {"a"}, sample_size=3, rng=rng)

    def test_deterministic_given_seed(self):
        media_ids = [f"m{i}" for i in range(50)]
        first, _ = mod.disjoint_offsets(
            media_ids, {"m3"}, sample_size=10, rng=np.random.default_rng(4)
        )
        second, _ = mod.disjoint_offsets(
            media_ids, {"m3"}, sample_size=10, rng=np.random.default_rng(4)
        )
        assert first.tolist() == second.tolist()
        assert "m3" not in {media_ids[i] for i in first}


class TestRepeatStatistics:
    def test_counts_slots_unique_and_within_message_repeats(self):
        records = [
            {"draw": 0, "message_index": 0, "modality": "image", "media_id": "a"},
            {"draw": 0, "message_index": 0, "modality": "image", "media_id": "a"},
            {"draw": 0, "message_index": 1, "modality": "image", "media_id": "b"},
            {"draw": 1, "message_index": 0, "modality": "text", "media_id": "a"},
            {"draw": 1, "message_index": 0, "modality": "text", "media_id": "c"},
        ]
        stats = mod.repeat_statistics(records)
        assert stats["slots_total"] == 5
        assert stats["unique_media_total"] == 3
        assert stats["reuse_factor_total"] == pytest.approx(5 / 3)
        assert stats["repeated_media_total"] == 1
        assert stats["within_message_repeat_slots"] == 1
        assert stats["max_uses_total"] == 3
        assert stats["per_modality"]["image"]["slots"] == 3
        assert stats["per_modality"]["image"]["unique_media"] == 2
        assert stats["per_modality"]["image"]["max_uses"] == 2
        assert stats["per_modality"]["text"]["reuse_factor"] == pytest.approx(1.0)
        assert len(stats["per_draw_unique_fractions"]) == 2

    def test_no_repeats(self):
        records = [
            {"draw": 0, "message_index": 0, "modality": "audio", "media_id": f"x{i}"}
            for i in range(4)
        ]
        stats = mod.repeat_statistics(records)
        assert stats["reuse_factor_total"] == pytest.approx(1.0)
        assert stats["within_message_repeat_slots"] == 0


class TestClusterEntropy:
    def test_entropy_bits_known_values(self):
        assert mod.entropy_bits(np.ones(256)) == pytest.approx(8.0)
        assert mod.entropy_bits(np.array([5])) == pytest.approx(0.0)
        assert mod.entropy_bits(np.array([1, 1, 2])) == pytest.approx(1.5)
        assert mod.entropy_bits(np.zeros(4)) == pytest.approx(0.0)

    def test_unique_media_entropy_dedupes(self):
        records = [
            {"modality": "image", "media_id": "a", "cluster_id": 0},
            {"modality": "image", "media_id": "a", "cluster_id": 0},
            {"modality": "image", "media_id": "b", "cluster_id": 1},
        ]
        summary = mod.cluster_entropy_summary(records)
        assert summary["slots_entropy_bits"] == pytest.approx(0.9182958340544896, abs=1e-9)
        assert summary["unique_entropy_bits"] == pytest.approx(1.0)
        assert summary["max_entropy_bits"] == pytest.approx(8.0)
        assert summary["per_modality"]["image"]["unique_entropy_ratio"] == pytest.approx(1 / 8)


class TestDropLedger:
    def test_clean_ledger_passes(self):
        ledger = mod.DropLedger()
        ledger.bump("messages_requested", 3)
        ledger.bump("carriers_scored", 12)
        ledger.raise_if_dirty("test")
        assert ledger.dirty() == []

    def test_any_hard_drop_fails_loudly(self):
        ledger = mod.DropLedger()
        ledger.bump("messages_encoded", 2)
        ledger.bump("messages_failed", 1)
        ledger.note_failure("draw 0 message 3: RuntimeError: boom")
        with pytest.raises(RuntimeError, match="fail-loud"):
            ledger.raise_if_dirty("draw 0")
        assert ledger.dirty() == ["messages_failed"]

    def test_to_dict_is_sorted_and_capped(self):
        ledger = mod.DropLedger()
        for i in range(15):
            ledger.note_failure(f"failure {i}")
        ledger.bump("carriers_scored", 1)
        payload = ledger.to_dict()
        assert list(payload["counters"]) == sorted(payload["counters"])
        assert len(payload["failure_samples"]) == 10


class TestBand:
    def test_band_reports_percentiles(self):
        result = mod.band([1.0, 2.0, 3.0, 4.0])
        assert result["mean"] == pytest.approx(2.5)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)
        assert result["p2_5"] <= result["p97_5"]

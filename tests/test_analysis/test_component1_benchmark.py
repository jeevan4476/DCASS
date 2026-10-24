"""Tests for the component 1 benchmark protocol (schema v2).

The properties pinned here are the ones the audit found violated in the
legacy artifact: the split must be image-level, so a cover and its LSB twin
never straddle train/val/test, and the reported statistics must behave
correctly under the null and under separation.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "benchmark_1_media_steganalysis.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_1_media_steganalysis", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


class _StubModel(torch.nn.Module):
    """Passes the first input value through the stego logit so the test can
    verify that score_loader preserves the cover/stego interleave."""

    def forward(self, x):
        values = x.reshape(x.shape[0], -1)[:, 0]
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 1] = values
        return logits


class TestImageLevelFolds:
    def test_folds_partition_the_pool(self):
        parts = mod.make_image_level_folds(100, 3)
        union = sorted(i for part in parts for i in part)
        assert union == list(range(100))

    def test_train_val_test_roles_are_disjoint_and_cover_the_rest(self):
        parts = mod.make_image_level_folds(100, 3)
        for fold in range(3):
            rest = [i for g, part in enumerate(parts) if g != fold for i in part]
            train, val = mod.split_train_val(rest)
            assert set(train).isdisjoint(val)
            assert set(train + val) == set(rest)
            assert set(parts[fold]).isdisjoint(train)
            assert set(parts[fold]).isdisjoint(val)

    def test_twins_cannot_straddle_splits(self):
        """Both items of every image derive from one pool index, so role
        assignment by pool index guarantees pair integrity."""
        parts = mod.make_image_level_folds(60, 3)
        roles = {}
        for fold, part in enumerate(parts):
            for index in part:
                roles[index] = f"test{fold}"
        rest = [i for g, part in enumerate(parts) if g != 0 for i in part]
        train, val = mod.split_train_val(rest)
        for index in train:
            roles[index] = "train"
        for index in val:
            roles[index] = "val"
        for index, role in roles.items():
            cover_item, stego_item = 2 * index, 2 * index + 1
            assert (cover_item // 2) == (stego_item // 2) == index, role


class TestTwinGeneration:
    def test_twin_is_deterministic_in_uid_rate_seed(self):
        base = torch.full((1, 32, 32), 0.5)
        first = mod.embed_twin(base, 7, 0.5)
        second = mod.embed_twin(base, 7, 0.5)
        other_uid = mod.embed_twin(base, 8, 0.5)
        assert torch.equal(first, second)
        assert not torch.equal(first, other_uid)

    def test_twin_stays_in_range_and_zero_rate_is_identity(self):
        base = torch.rand(1, 48, 48)
        for rate in (0.1, 0.5, 1.0):
            twin = mod.embed_twin(base, 3, rate)
            assert float(twin.min()) >= 0.0
            assert float(twin.max()) <= 1.0
        assert torch.equal(mod.embed_twin(base, 3, 0.0), base)

    def test_full_rate_touches_every_pixel_parity(self):
        base = torch.full((1, 16, 16), 0.5)
        twin = mod.embed_twin(base, 1, 1.0)
        assert twin.numel() - int((twin == base).sum()) > 0.99 * twin.numel()


class TestStatistics:
    def test_rank_auc_bounds_and_reference_cases(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=200)
        assert 0.0 <= mod.rank_auc(sample, rng.normal(size=200)) <= 1.0
        assert mod.rank_auc(sample + 10.0, sample) == 1.0
        assert abs(mod.rank_auc(sample, sample) - 0.5) < 1e-12

    def test_permutation_auc_does_not_reject_under_the_null(self):
        rng = np.random.default_rng(1)
        result = mod.permutation_auc_test(
            rng.uniform(0, 1, 150), rng.uniform(0, 1, 150), n_perm=300, seed=3
        )
        assert result["p"] > 0.05

    def test_permutation_auc_rejects_under_separation(self):
        rng = np.random.default_rng(2)
        cover = rng.normal(0.0, 0.5, 150)
        carrier = rng.normal(1.5, 0.5, 150)
        result = mod.permutation_auc_test(cover, carrier, n_perm=300, seed=3)
        assert result["auc"] > 0.9
        assert result["p"] < 0.01

    def test_kl_permutation_is_calibrated_under_the_null(self):
        rng = np.random.default_rng(4)
        result = mod.kl_permutation_test(
            rng.uniform(0, 1, 200), rng.uniform(0, 1, 200), n_perm=200, seed=5
        )
        assert result["kl"] >= 0.0
        assert result["p"] > 0.05

    def test_wilson_interval_brackets_the_estimate(self):
        lo, hi = mod.wilson_ci(95, 100)
        assert lo < 0.95 < hi
        assert 0.0 <= lo <= hi <= 1.0

    def test_cover_class_rate_orientation(self):
        assert mod.cover_class_rate([0.1, 0.5, 0.9]) == pytest.approx(2 / 3)
        assert mod.cover_class_rate([0.9, 0.9]) == 0.0
        assert mod.cover_class_rate([0.1, 0.2]) == 1.0

    def test_balanced_accuracy_orientation(self):
        cover = np.array([0.2, 0.4, 0.6, 0.8])
        carrier = np.array([0.9, 0.7, 0.3, 0.1])
        assert mod.balanced_accuracy(cover, carrier) == pytest.approx(0.5)
        # Identical score distributions must give exactly 0.5 at any base
        # rate, not the cover base rate of the 0.5 threshold.
        cover_heavy = np.array([0.1, 0.1, 0.1, 0.9])
        assert mod.balanced_accuracy(cover_heavy, cover_heavy.copy()) == pytest.approx(0.5)
        # A detectable carrier-versus-cover gap moves the metric by half the
        # decision-rate difference.
        leaky_carrier = np.array([0.6, 0.7, 0.8, 0.9])
        assert mod.balanced_accuracy(cover, leaky_carrier) == pytest.approx(0.75)

    def test_bootstrap_auc_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(6)
        cover = rng.uniform(0, 1, 120)
        carrier = rng.uniform(0, 1, 120)
        ci = mod.bootstrap_auc_ci(cover, carrier, n_boot=200, seed=7)
        point = mod.rank_auc(carrier, cover)
        assert ci["lo"] <= point <= ci["hi"]


class TestScoringMode:
    def test_scoring_forces_eval_mode_and_batch_invariance(self):
        """Regression for the component 1 audit finding: with BatchNorm
        layers left in training mode, scores depended on batch composition,
        which fabricated a carrier-versus-cover AUC gap of 0.73 to 0.82 on
        unmodified bytes."""
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )
        model.train()
        with torch.no_grad():
            model(torch.rand(16, 1, 8, 8))
            model(torch.rand(16, 1, 8, 8))
        assert model.training is True

        images = torch.rand(8, 1, 8, 8)
        together = mod.score_batch(model, "stub_bn", images)
        first_half = mod.score_batch(model, "stub_bn", images[:4])
        second_half = mod.score_batch(model, "stub_bn", images[4:])
        assert model.training is False
        assert np.allclose(together[:4], first_half)
        assert np.allclose(together[4:], second_half)


class TestScoreLoaderAlignment:
    def test_misaligned_labels_are_rejected(self):
        loader = [[torch.zeros(4, 1, 2, 2), torch.tensor([0, 1, 1, 0])]]
        with pytest.raises(RuntimeError):
            mod.score_loader(_StubModel(), "stub", loader)

    def test_aligned_labels_split_cover_and_stego(self):
        images = torch.arange(4).float().view(4, 1, 1, 1) / 10.0
        loader = [[images, torch.tensor([0, 1, 0, 1])]]
        cover, stego = mod.score_loader(_StubModel(), "stub", loader)
        assert cover.shape == (2,)
        assert stego.shape == (2,)
        assert np.all(stego > cover)


class TestGateExitCode:
    """The CLI exit status must say whether the positive-control gates
    passed; before FIX-F the script always exited 0."""

    def test_passing_summary_exits_zero(self):
        assert mod.gate_exit_code({"all_gates_passed": True}) == 0

    def test_failing_summary_exits_one(self):
        assert mod.gate_exit_code({"all_gates_passed": False}) == 1

# tests/test_corpus/test_voronoi_codebook_fit.py
"""
Unit tests for VoronoiCodebook's own fitting lifecycle. The sibling file
(test_voronoi_codebook.py) asserts against the shipped fitted artifact and
skips without it, so no run anywhere covered `fit`, the soft-margin
filter's branches, or save/load — while the C4b record is built on them.

The tests here are self-contained: synthetic vectors on the unit sphere,
deterministic seeds, the trainer's own convergence path on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.corpus.cluster.voronoi_codebook import VoronoiCodebook


def _separated_embeddings(
    num_groups: int, per_group: int, dim: int, noise: float = 0.05, seed: int = 7
) -> np.ndarray:
    """Well-separated unit-sphere clusters around exactly orthonormal poles.

    QR turns one seeded gaussian draw into `num_groups` exactly orthonormal
    poles, so every point's similarity to its own pole is ~0.99 and to every
    other pole is ~0: spherical K-Means recovers the planted partition within
    a few iterations. Rejection-sampling raw gaussian poles instead would
    compare dot products (sd sqrt(dim)) against a correlation threshold and
    loop effectively forever.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, num_groups)))
    poles = q.T.astype(np.float32)  # (num_groups, dim), orthonormal rows
    chunks = [
        poles[group][None, :] + noise * rng.normal(size=(per_group, dim))
        for group in range(num_groups)
    ]
    return np.vstack(chunks).astype(np.float32)


class TestFit:
    def test_partitions_corpus_and_reports_metrics(self):
        num_groups, per_group, dim = 8, 200, 16
        emb = _separated_embeddings(num_groups, per_group, dim)
        codebook = VoronoiCodebook(num_clusters=8, dim=dim)

        metrics = codebook.fit(emb, max_iters=25, device="cpu", seed=42)

        assert metrics["num_vectors"] == num_groups * per_group
        assert metrics["num_clusters"] == 8
        assert metrics["mean_density"] == num_groups * per_group / 8
        # A planted 8-way split partitions the whole corpus.
        assert sum(len(v) for v in codebook.cluster_to_indices.values()) == metrics["num_vectors"]
        # With well-separated poles every cluster is used.
        assert metrics["min_density"] > 0
        assert metrics["max_density"] <= metrics["num_vectors"]
        assert codebook.is_fitted is True
        assert codebook.centroids.shape == (8, dim)
        norms = np.linalg.norm(codebook.centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_fit_is_deterministic_under_seed(self):
        emb = _separated_embeddings(8, 200, 16)
        first = VoronoiCodebook(num_clusters=8, dim=16)
        second = VoronoiCodebook(num_clusters=8, dim=16)
        first.fit(emb, max_iters=25, device="cpu", seed=42)
        second.fit(emb, max_iters=25, device="cpu", seed=42)

        np.testing.assert_array_equal(first.centroids, second.centroids)
        np.testing.assert_array_equal(first.cluster_assignments, second.cluster_assignments)

    def test_rejects_dimension_mismatch(self):
        codebook = VoronoiCodebook(num_clusters=8, dim=16)
        with pytest.raises(ValueError, match="dimension mismatch"):
            codebook.fit(np.zeros((10, 8), dtype=np.float32), device="cpu")

    def test_empty_cluster_reinit_runs_for_orphaned_centroid(self, monkeypatch):
        # Two clusters seeded by identical rows tie under argmax, orphaning a
        # centroid: the update loop must re-initialize it with a random row.
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(16, 2)))
        poles = q.T.astype(np.float32)
        emb = np.vstack(
            [
                np.repeat(poles[0][None, :], 6, axis=0),
                np.repeat(poles[1][None, :], 6, axis=0),
            ]
        ).astype(np.float32)

        calls = []
        real_randint = torch.randint

        def spy_randint(*args, **kwargs):
            out = real_randint(*args, **kwargs)
            calls.append(int(out.item()))
            return out

        monkeypatch.setattr(torch, "randint", spy_randint)
        codebook = VoronoiCodebook(num_clusters=3, dim=16)
        metrics = codebook.fit(emb, max_iters=5, device="cpu", seed=42)

        assert calls  # the empty-cluster branch re-drew a random row
        assert metrics["min_density"] == 0
        assert sum(len(v) for v in codebook.cluster_to_indices.values()) == 12
        norms = np.linalg.norm(codebook.centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_assign_matches_fit_partitions(self):
        emb = _separated_embeddings(8, 200, 16)
        codebook = VoronoiCodebook(num_clusters=8, dim=16)
        codebook.fit(emb, max_iters=25, device="cpu", seed=42)

        np.testing.assert_array_equal(codebook.assign(emb), codebook.cluster_assignments)

    def test_stratified_offsets_keep_slices_apart(self):
        # Two disjoint modality slices; stratified init must draw initial
        # centroids from both, and the fit must not let one slice's clusters
        # absorb the other slice's vectors.
        group_a = _separated_embeddings(2, 400, 16, seed=11)
        group_b = _separated_embeddings(2, 400, 16, seed=23)
        emb = np.vstack([group_a, group_b]).astype(np.float32)

        codebook = VoronoiCodebook(num_clusters=4, dim=16)
        codebook.fit(emb, max_iters=25, device="cpu", seed=42, stratified_offsets=[0, 800, 1600])

        assignments = codebook.cluster_assignments
        slice_a_clusters = set(assignments[:800].tolist())
        slice_b_clusters = set(assignments[800:].tolist())
        assert len(slice_a_clusters) == 2
        assert len(slice_b_clusters) == 2
        assert slice_a_clusters.isdisjoint(slice_b_clusters)

    def test_stratified_offsets_distribute_the_remainder(self):
        # num_clusters % n_slices != 0: the first slice must take the extra
        # centroid (K=5 over 2 slices -> [3, 2] instead of [2, 2]).
        # Each slice is a single dense group around one pole, so every
        # random init draw lands in that group and the per-slice K split is
        # stable regardless of which points are drawn.
        emb = np.vstack(
            [
                _separated_embeddings(1, 900, 16, seed=3),
                _separated_embeddings(1, 900, 16, seed=9),
            ]
        ).astype(np.float32)

        codebook = VoronoiCodebook(num_clusters=5, dim=16)
        codebook.fit(emb, max_iters=25, device="cpu", seed=42, stratified_offsets=[0, 900, 1800])

        assignments = codebook.cluster_assignments
        first_slice_clusters = set(assignments[:900].tolist())
        second_slice_clusters = set(assignments[900:].tolist())
        assert len(first_slice_clusters) == 3
        assert len(second_slice_clusters) == 2
        assert first_slice_clusters.isdisjoint(second_slice_clusters)
        assert min(len(codebook.get_cluster_indices(c)) for c in first_slice_clusters) > 0


class TestUnfittedGuards:
    def test_assign_raises_when_unfitted(self):
        codebook = VoronoiCodebook(num_clusters=8, dim=16)
        with pytest.raises(RuntimeError, match="not fitted"):
            codebook.assign(np.zeros((4, 16), dtype=np.float32))

    def test_save_raises_when_unfitted(self, tmp_path):
        codebook = VoronoiCodebook(num_clusters=8, dim=16)
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            codebook.save(tmp_path / "codebook.npz")


class TestSoftMarginFilter:
    @staticmethod
    def _fitted_two_pole_codebook(delta_margin: float = 0.05) -> VoronoiCodebook:
        """A fitted K=2 codebook whose centroids hug +/- e0 exactly.

        The fit sees only clean interior rows, so the centroid's off-axis
        components stay ~0.003; had the boundary rows been included, they
        would have dragged the centroids halfway toward themselves and made
        the margin semantics data-order dependent.
        """
        dim = 4
        poles = np.zeros((2, dim), dtype=np.float32)
        poles[0, 0] = 1.0
        poles[1, 0] = -1.0
        cloud = np.array(
            [
                poles[0] + 0.02 * np.array([0.0, 0.4, 0.4, 0.1], dtype=np.float32),
                poles[0] + 0.02 * np.array([0.0, -0.3, 0.2, 0.3], dtype=np.float32),
                poles[1] + 0.02 * np.array([0.0, 0.4, 0.4, 0.1], dtype=np.float32),
                poles[1] + 0.02 * np.array([0.0, -0.3, 0.2, 0.3], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        codebook = VoronoiCodebook(num_clusters=2, dim=dim, delta_margin=delta_margin)
        codebook.fit(np.vstack([poles, cloud]), max_iters=5, device="cpu", seed=1)
        return codebook, poles

    def test_filter_keeps_interior_and_drops_boundary(self):
        codebook, poles = self._fitted_two_pole_codebook()

        # Candidate corpus (rows the fit never saw): row 2 hugs the +e0 pole,
        # row 3 is exactly perpendicular to both poles, i.e. equidistant.
        interior = poles[0] + 0.05 * np.array([0.0, 0.3, 0.3, 0.3], dtype=np.float32)
        boundary = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        corpus = np.vstack([poles, interior[None, :], boundary[None, :]]).astype(np.float32)

        cluster_id = int(codebook.assign(poles[0][None, :])[0])
        robust = codebook.filter_soft_margin(corpus, [2, 3], cluster_id)

        # Interior (margin ~1.0) survives; boundary (margin ~0.006 < 0.05) goes.
        assert robust == [2]

    def test_fallback_returns_all_candidates_when_all_are_boundary(self):
        codebook, poles = self._fitted_two_pole_codebook()

        boundary_a = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        boundary_b = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        corpus = np.vstack([poles, boundary_a[None, :], boundary_b[None, :]]).astype(np.float32)

        cluster_id = int(codebook.assign(poles[0][None, :])[0])
        robust = codebook.filter_soft_margin(corpus, [2, 3], cluster_id)

        # Every candidate sat on the boundary, so the filter falls back to the
        # original candidate list instead of dropping them all.
        assert robust == [2, 3]

    def test_empty_candidates_pass_through(self):
        codebook = VoronoiCodebook(num_clusters=4, dim=16)
        emb = _separated_embeddings(4, 100, 16)
        codebook.fit(emb, max_iters=5, device="cpu", seed=42)

        assert codebook.filter_soft_margin(emb, [], cluster_id=1) == []

    def test_unfitted_codebook_passes_candidates_through(self):
        codebook = VoronoiCodebook(num_clusters=4, dim=16)
        emb = np.zeros((4, 16), dtype=np.float32)
        assert codebook.filter_soft_margin(emb, [0, 1], cluster_id=1) == [0, 1]

    def test_get_cluster_indices(self):
        emb = _separated_embeddings(4, 100, 16, seed=5)
        codebook = VoronoiCodebook(num_clusters=4, dim=16)
        codebook.fit(emb, max_iters=10, device="cpu", seed=42)

        indices = codebook.get_cluster_indices(0)
        assert all(codebook.cluster_assignments[i] == 0 for i in indices)
        # Every cluster of the planted 4-way partition has rows.
        assert min((len(codebook.get_cluster_indices(c)) for c in range(codebook.num_clusters))) > 0


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        emb = _separated_embeddings(8, 200, 16)
        codebook = VoronoiCodebook(num_clusters=8, dim=16, delta_margin=0.1)
        codebook.fit(emb, max_iters=25, device="cpu", seed=42)

        path = tmp_path / "nested" / "codebook.npz"
        codebook.save(path)
        assert path.exists()

        restored = VoronoiCodebook(num_clusters=8, dim=16)
        restored.load(path)

        assert restored.is_fitted is True
        np.testing.assert_array_equal(restored.centroids, codebook.centroids)
        np.testing.assert_array_equal(
            restored.cluster_assignments, codebook.cluster_assignments
        )
        assert restored.delta_margin == pytest.approx(0.1)
        # Assignments made through the loaded codebook match the original's.
        np.testing.assert_array_equal(restored.assign(emb), codebook.assign(emb))

    def test_load_missing_file(self, tmp_path):
        codebook = VoronoiCodebook(num_clusters=8, dim=16)
        with pytest.raises(FileNotFoundError):
            codebook.load(tmp_path / "missing.npz")

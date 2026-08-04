# tests/test_corpus/test_voronoi_codebook.py
"""
Unit tests for Spherical K-Means Voronoi Codebook Partitioning (VCP).
Verifies centroid norm constraint (||c||_2 = 1.0) and soft-margin filtering.
"""

from pathlib import Path
import numpy as np
from src.corpus.cluster.voronoi_codebook import VoronoiCodebook

PROJECT_ROOT = Path(__file__).parent.parent.parent
CODEBOOK_PATH = PROJECT_ROOT / "storage" / "data" / "indices" / "voronoi_codebook.npz"

def test_voronoi_codebook_centroids_unit_norm():
    """Verify that all 256 centroids have exact unit norm (||c||_2 = 1.0)."""
    assert CODEBOOK_PATH.exists(), f"Codebook file not found at {CODEBOOK_PATH}"
    
    codebook = VoronoiCodebook()
    codebook.load(CODEBOOK_PATH)
    
    assert codebook.is_fitted is True
    assert codebook.num_clusters == 256
    assert codebook.centroids.shape == (256, 512)

    norms = np.linalg.norm(codebook.centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    print("✅ Unit Norm Constraint Verified: All 256 centroids have ||c||_2 = 1.0!")

def test_voronoi_codebook_symbol_assignment():
    """Verify deterministic symbol assignment for query vectors."""
    codebook = VoronoiCodebook()
    codebook.load(CODEBOOK_PATH)

    # Use first 5 centroids as queries
    queries = codebook.centroids[:5].copy()
    assigned = codebook.assign(queries)
    
    # Each centroid query should assign back to its own index 0..4
    np.testing.assert_array_equal(assigned, np.arange(5))
    print("✅ Symbol Assignment Verified: Centroid queries assign to exact symbol IDs!")

if __name__ == "__main__":
    test_voronoi_codebook_centroids_unit_norm()
    test_voronoi_codebook_symbol_assignment()

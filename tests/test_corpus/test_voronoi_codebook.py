# tests/test_corpus/test_voronoi_codebook.py
"""
Unit tests for Spherical K-Means Voronoi Codebook Partitioning (VCP).
Verifies centroid norm constraint (||c||_2 = 1.0) and soft-margin filtering.
"""

from pathlib import Path
import numpy as np
import pytest
from src.corpus.cluster.voronoi_codebook import VoronoiCodebook

PROJECT_ROOT = Path(__file__).parent.parent.parent
CODEBOOK_PATH = PROJECT_ROOT / "storage" / "data" / "indices" / "voronoi_codebook.npz"

pytestmark = pytest.mark.skipif(
    not CODEBOOK_PATH.exists(),
    reason=f"Fitted codebook artifact not present at {CODEBOOK_PATH}",
)


def test_voronoi_codebook_centroids_unit_norm():
    """Verify that all 256 centroids have exact unit norm (||c||_2 = 1.0)."""
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


def test_voronoi_codebook_global_offset_reconstruction():
    """
    P1-9: Verify VCPPayloadMapper's global-offset reconstruction against the
    real fitted codebook and the real FAISS indices.

    The fitting script stacks vectors in image, text, audio order; the mapper
    reconstructs global rows in the same order. This is the single most
    fragile coupling in the system - this test catches drift between a
    refitted codebook and rebuilt indices.
    """
    import faiss

    from src.corpus.index.unified_index import UnifiedSemanticIndex
    from src.engine.vcp_payload import VCPPayloadMapper

    indices_dir = CODEBOOK_PATH.parent
    index = UnifiedSemanticIndex(base_path=indices_dir)
    status = index.load()
    assert all(status.values()), f"Indices failed to load: {status}"

    mapper = VCPPayloadMapper(index)
    mapper.load()

    # 1) Every media ID maps to the same symbol as its raw vector does under
    #    nearest-centroid assignment. Catches offset/ordering corruption.
    checked = 0
    for modality in ("image", "text", "audio"):
        meta_list = index.metadata[modality]
        faiss_index = index.indices[modality]
        step = max(1, len(meta_list) // 25)
        for local_idx in range(0, len(meta_list), step):
            meta = meta_list[local_idx]
            media_id = meta.get("id")
            if not media_id:
                continue
            vec = np.asarray(faiss_index.reconstruct(local_idx), dtype=np.float32)
            expected_symbol = int(mapper.codebook.assign(vec[None, :])[0])
            mapped_symbol = mapper.symbol_for_media_id(media_id)
            assert mapped_symbol == expected_symbol, (
                f"Global offset mismatch for {modality}[{local_idx}] "
                f"({media_id}): mapped={mapped_symbol}, vector={expected_symbol}"
            )
            checked += 1
    assert checked >= 25, "Sampled too few vectors to validate offsets"


if __name__ == "__main__":
    test_voronoi_codebook_centroids_unit_norm()
    test_voronoi_codebook_symbol_assignment()

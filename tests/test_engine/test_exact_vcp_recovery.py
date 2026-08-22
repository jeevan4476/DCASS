# tests/test_engine/test_exact_vcp_recovery.py
"""
P1-8: Within-capacity Reed-Solomon recovery exercised THROUGH the exact_vcp
carrier path (not just at the ECC layer).

Corrupts 1..t carrier IDs (swapping in carriers from different clusters, i.e.
wrong payload bytes) and asserts the decoder recovers the original message
exactly. Uses a fake index/codebook so it runs without artifacts.
"""

import numpy as np
import pytest

from src.corpus.index.unified_index import UnifiedSemanticIndex
from src.corpus.cluster.voronoi_codebook import VoronoiCodebook
from src.engine.decoder import SemanticDecoder
from src.engine.encoder import SemanticEncoder
from src.engine.vcp_payload import VCPPayloadMapper


NUM_CLUSTERS = 256
DIM = 16


class FakeIndex(UnifiedSemanticIndex):
    """Minimal in-memory index: 8 items per cluster across one modality."""

    def __init__(self):
        from src.corpus.index.unified_index import ScoreNormalizer

        self.base_path = None
        self.device = "cpu"
        self.enabled_modalities = ["text"]
        self.indices = {}
        self.metadata = {}
        self._id_lookup = {}
        self.normalizer = ScoreNormalizer()
        # Build a deterministic codebook: centroid c = one-hot-ish vector.
        codebook = VoronoiCodebook(num_clusters=NUM_CLUSTERS, dim=DIM)
        centroids = np.zeros((NUM_CLUSTERS, DIM), dtype=np.float32)
        for c in range(NUM_CLUSTERS):
            centroids[c] = np.cos(np.arange(DIM) * (c + 1) * 0.37)
            centroids[c] /= np.linalg.norm(centroids[c])
        codebook.centroids = centroids
        codebook.cluster_assignments = np.zeros(0, dtype=np.int64)
        codebook._fitted = True

        vectors = []
        metas = []
        for symbol in range(NUM_CLUSTERS):
            for j in range(8):
                vec = centroids[symbol] + np.linspace(-0.01, 0.01, DIM).astype(
                    np.float32
                ) * (j - 4)
                vec /= np.linalg.norm(vec)
                vectors.append(vec)
                metas.append(
                    {
                        "id": f"txt_{symbol:02x}_{j}",
                        "content": f"carrier {symbol:02x}.{j}",
                        "path": "",
                        "text": f"carrier {symbol:02x}.{j}",
                    }
                )
        assignments = []
        for v in vectors:
            np.dot(np.array(vectors), np.array(v))
            assignments.append(
                int(
                    np.argmax(
                        np.stack([np.dot(v, centroids[c]) for c in range(NUM_CLUSTERS)])
                    )
                )
            )
        self.codebook = codebook
        self._vcp_codebook = codebook  # consumed by VCPPayloadMapper
        self._vectors = np.array(vectors, dtype=np.float32)

        class _Flat:
            def __init__(self, vectors):
                import faiss

                self.inner = faiss.IndexFlatIP(vectors.shape[1])
                self.inner.add(vectors.astype("float32"))
                self._vectors = vectors

            @property
            def ntotal(self):
                return self.inner.ntotal

            def reconstruct(self, i):
                return self._vectors[i]

            def search(self, q, k):
                return self.inner.search(q, k)

        flat = _Flat(self._vectors)
        # VCP requires all canonical modalities; mirror the corpus per
        # modality with distinct ID prefixes so offsets are exercised too.
        global_offset = 0
        for modality in ("image", "text", "audio"):
            metas = [
                {
                    "id": f"{modality}_{symbol:02x}_{j}",
                    "content": f"carrier {modality} {symbol:02x}.{j}",
                    "path": "",
                    "text": f"carrier {modality} {symbol:02x}.{j}",
                }
                for symbol in range(NUM_CLUSTERS)
                for j in range(8)
            ]
            self.indices[modality] = flat
            self.metadata[modality] = metas
            for local_idx, meta in enumerate(metas):
                self._id_lookup[meta["id"]] = (modality, local_idx)
            codebook.cluster_assignments = np.concatenate(
                [
                    codebook.cluster_assignments,
                    np.array(
                        [s for s in range(NUM_CLUSTERS) for _ in range(8)],
                        dtype=np.int64,
                    ),
                ]
            )
            global_offset += len(metas)

    def load(self, modalities=None):
        return {"image": True, "text": True, "audio": True}

    def _encode_query(self, text, modality):
        # Deterministic pseudo-query embedding in the same space.
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(DIM).astype(np.float32)
        return (v / np.linalg.norm(v))[None, :]


@pytest.fixture()
def setup():
    index = FakeIndex()
    encoder = SemanticEncoder(index=index)
    encoder._loaded = True
    decoder = SemanticDecoder(index=index)
    decoder._loaded = True
    mapper = VCPPayloadMapper(index)
    mapper.codebook = index.codebook
    mapper.load()
    return encoder, decoder, mapper


def test_clean_roundtrip_through_vcp(setup):
    encoder, decoder, _ = setup
    msg = "Meet at noon"
    result = encoder.encode(msg, payload_mode="exact_vcp", use_ecc=True)
    decoded = decoder.decode(result.media_ids, payload_mode="exact_vcp", use_ecc=True)
    assert decoded.ecc_success
    assert decoded.reconstructed_meaning == msg


def test_rs_recovery_through_corrupted_carriers(setup):
    """Swap up to t=4 carriers with wrong-cluster IDs; RS must recover."""
    encoder, decoder, mapper = setup
    msg = "Attack at dawn"
    result = encoder.encode(msg, payload_mode="exact_vcp", use_ecc=True)
    ids = list(result.media_ids)

    # Build a lookup of substitute carriers from a DIFFERENT cluster.
    by_symbol = {}
    for mid, sym in mapper._id_to_symbol.items():
        by_symbol.setdefault(sym, []).append(mid)
    used = set(ids)

    corrupted = list(ids)
    t = 4
    positions = [0, len(ids) // 3, 2 * len(ids) // 3, len(ids) - 1][:t]
    for pos in positions:
        orig_sym = mapper.symbol_for_media_id(ids[pos])
        replacement = next(
            m
            for s, candidates in by_symbol.items()
            if s != orig_sym
            for m in candidates
            if m not in used
        )
        used.discard(ids[pos])
        used.add(replacement)
        corrupted[pos] = replacement

    decoded = decoder.decode(corrupted, payload_mode="exact_vcp", use_ecc=True)
    assert decoded.ecc_success, "RS-ECC should correct 4 corrupted carriers"
    assert decoded.reconstructed_meaning == msg


def test_beyond_capacity_reports_failure(setup):
    """More than t corruptions must NOT silently pass as success."""
    encoder, decoder, mapper = setup
    msg = "A longer message that gives us room to corrupt many bytes safely."
    result = encoder.encode(msg, payload_mode="exact_vcp", use_ecc=True)
    ids = list(result.media_ids)

    by_symbol = {}
    for mid, sym in mapper._id_to_symbol.items():
        by_symbol.setdefault(sym, []).append(mid)
    used = set(ids)

    corrupted = list(ids)
    n_corrupt = 9  # > t = 4; statistically near-certain to exceed capacity
    positions = list(range(0, min(n_corrupt * 2, len(ids)), 2))[:n_corrupt]
    for pos in positions:
        orig_sym = mapper.symbol_for_media_id(ids[pos])
        replacement = next(
            m
            for s, candidates in by_symbol.items()
            if s != orig_sym
            for m in candidates
            if m not in used
        )
        used.discard(ids[pos])
        used.add(replacement)
        corrupted[pos] = replacement

    decoded = decoder.decode(corrupted, payload_mode="exact_vcp", use_ecc=True)
    # Beyond-capacity corruption must be detected (no silent wrong output).
    assert not decoded.ecc_success or decoded.reconstructed_meaning == msg, (
        "Decoder claimed success on a beyond-capacity corruption"
    )


if __name__ == "__main__":
    import sys

    s = setup.__wrapped__(None) if hasattr(setup, "__wrapped__") else None
    sys.exit(pytest.main([__file__, "-v"]))

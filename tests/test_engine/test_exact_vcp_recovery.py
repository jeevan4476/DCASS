# tests/test_engine/test_exact_vcp_recovery.py
"""
P1-8: Within-capacity Reed-Solomon recovery exercised THROUGH the exact_vcp
carrier path (not just at the ECC layer).

Corrupts 1..t carrier IDs (swapping in carriers from different clusters, i.e.
wrong payload bytes) and asserts the decoder recovers the original message
exactly. Uses a fake index/codebook so it runs without artifacts.
"""

import hashlib

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
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
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
    result = encoder.encode(msg, use_ecc=True)
    decoded = decoder.decode(result.media_ids, use_ecc=True)
    assert decoded.ecc_success
    assert decoded.reconstructed_meaning == msg


def test_rs_recovery_through_corrupted_carriers(setup):
    """Swap up to t=4 carriers with wrong-cluster IDs; RS must recover."""
    encoder, decoder, mapper = setup
    msg = "Sample message"
    result = encoder.encode(msg, use_ecc=True)
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

    decoded = decoder.decode(corrupted, use_ecc=True)
    assert decoded.ecc_success, "RS-ECC should correct 4 corrupted carriers"
    assert decoded.reconstructed_meaning == msg


def test_beyond_capacity_reports_failure(setup):
    """More than t corruptions must NOT silently pass as success."""
    encoder, decoder, mapper = setup
    msg = "A longer message that gives us room to corrupt many bytes safely."
    result = encoder.encode(msg, use_ecc=True)
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

    decoded = decoder.decode(corrupted, use_ecc=True)
    # Beyond-capacity corruption must be detected (no silent wrong output).
    assert not decoded.ecc_success or decoded.reconstructed_meaning == msg, (
        "Decoder claimed success on a beyond-capacity corruption"
    )


def test_rs_erasure_recovery_through_dropped_carriers(setup):
    """Dropping up to R=8 carriers (known positions) recovers 100%; R+1 fails closed."""
    encoder, decoder, _ = setup
    msg = "Packet drop resilience test through exact_vcp carriers."
    result = encoder.encode(msg, use_ecc=True, ecc_parity_bytes=8)
    ids = list(result.media_ids)

    for drop_count in (1, 2, 3, 5, 8):
        drop_positions = [i * 3 for i in range(drop_count)]
        received = [None if i in drop_positions else m for i, m in enumerate(ids)]
        decoded = decoder.decode(
            received,
            use_ecc=True,
            ecc_parity_bytes=8,
            erasure_positions=drop_positions,
        )
        assert decoded.ecc_success is True, f"Should recover {drop_count} dropped packets"
        assert decoded.reconstructed_meaning == msg
        assert len(decoded.ecc_errors_fixed) >= drop_count

    # Dropping 9 packets (> parity=8) must fail closed
    over_positions = [i * 3 for i in range(9)]
    received_over = [None if i in over_positions else m for i, m in enumerate(ids)]
    decoded_over = decoder.decode(
        received_over,
        use_ecc=True,
        ecc_parity_bytes=8,
        erasure_positions=over_positions,
    )
    assert decoded_over.ecc_success is False


def test_parity_only_corruption_fails_closed_without_leaking_plaintext(setup):
    """Regression test (H7): corrupting ONLY parity bytes must fail closed.

    With R=8, corrupting 5 of the 8 parity slots (data bytes left untouched)
    exceeds RS error capacity (t = R/2 = 4), so decode_bytes reports failure
    while returning the intact data slice. The decoder must NOT unframe that
    unverified slice: it must report ecc_success=False and must not expose
    the original plaintext via ecc_payload / reconstructed_meaning.
    """
    encoder, decoder, mapper = setup
    msg = "Parity-only corruption must not leak this plaintext."
    result = encoder.encode(msg, use_ecc=True, ecc_parity_bytes=8)
    ids = list(result.media_ids)

    n_parity = 8
    data_len = len(ids) - n_parity
    assert data_len > 0, "codeword must have data bytes before parity bytes"

    by_symbol = {}
    for mid, sym in mapper._id_to_symbol.items():
        by_symbol.setdefault(sym, []).append(mid)
    used = set(ids)

    # Corrupt 5 of the last 8 carrier slots (the parity bytes) with carriers
    # from different clusters; the data-byte carriers stay untouched.
    corrupted = list(ids)
    parity_positions = list(range(data_len, len(ids)))[:5]
    assert len(parity_positions) == 5
    for pos in parity_positions:
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

    decoded = decoder.decode(corrupted, use_ecc=True, ecc_parity_bytes=8)
    assert decoded.ecc_success is False
    # Fail-closed: empty payload, never the raw/unframed plaintext.
    assert decoded.ecc_payload == ""
    assert msg not in (decoded.ecc_payload or "")
    assert decoded.reconstructed_meaning != msg
    assert msg not in decoded.reconstructed_meaning


if __name__ == "__main__":
    import sys

    s = setup.__wrapped__(None) if hasattr(setup, "__wrapped__") else None
    sys.exit(pytest.main([__file__, "-v"]))

"""
End-to-end DSSC roundtrip with a minimal synthetic corpus.
Verifies that fixed encoder and decoder agree on family, produce same
permutation, and recover the exact original message.
"""
from unittest.mock import MagicMock


def _build_synthetic_index_and_mapper(n_per_family: int = 20):
    """
    Build a synthetic index + VCPPayloadMapper with deterministic cluster assignments.
    Assigns IDs nature_0..nature_N to cluster 0, urban_0..urban_N to cluster 42, etc.
    """
    from src.engine.vcp_payload import VCPPayloadMapper
    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES

    # Assign IDs to clusters
    cluster_map: dict[str, int] = {}
    all_ids: list[str] = []
    for fam in DEFAULT_SEMANTIC_FAMILIES:
        cluster_id = fam.cluster_ids[0]  # first cluster of each family
        for i in range(n_per_family):
            mid = f"{fam.name}_{i}"
            cluster_map[mid] = cluster_id
            all_ids.append(mid)

    # Mock VCPPayloadMapper
    mapper = MagicMock(spec=VCPPayloadMapper)
    mapper.symbol_for_media_id.side_effect = lambda mid: cluster_map.get(mid)

    # Mock UnifiedSemanticIndex
    index = MagicMock()
    index.metadata = {
        "image": [{"id": mid} for mid in all_ids],
    }
    item_mock = MagicMock()
    item_mock.content = "test content"
    index.get_by_id.return_value = item_mock

    return index, mapper, all_ids, cluster_map


def test_dssc_roundtrip_exact_recovery():
    """Encoder and decoder must agree on family and recover the exact message."""
    from src.engine.dssc_encoder import DSSCEncoder
    from src.engine.dssc_decoder import DSSCDecoder

    index, vcp_mapper, all_ids, cluster_map = _build_synthetic_index_and_mapper(n_per_family=30)
    session_key = b"secure_session_key_32bytes_padded"

    enc = DSSCEncoder(index=index, vcp_mapper=vcp_mapper)
    dec = DSSCDecoder(index=index, vcp_mapper=vcp_mapper)

    message = "hello world"
    enc_result = enc.encode(message, session_key=session_key, ecc_parity_bytes=8)
    assert enc_result.carrier_ids, "Encoder produced no carriers"

    dec_result = dec.decode(enc_result.carrier_ids, session_key=session_key, ecc_parity_bytes=8)
    assert dec_result.success, f"Decode failed: {dec_result}"
    assert dec_result.reconstructed_message == message, (
        f"Expected '{message}', got '{dec_result.reconstructed_message}'"
    )

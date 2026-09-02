"""
Unit tests for DSSC (Dynamic Semantic State-Space Coding).

Verifies:
1. BitStreamReader and BitStreamWriter bit-exact roundtrip
2. DSSCStateSpace capacity and symbol-to-media-id mapping
3. End-to-end DSSC encoding and decoding exact recovery
4. Reed-Solomon error correction under simulated carrier substitution
5. Session-key security (different key prevents decoding)
"""

from unittest.mock import MagicMock
from src.engine.dssc_state_space import (
    DSSCStateSpace,
    derive_session_permutation,
    DEFAULT_SEMANTIC_FAMILIES,
)
from src.engine.dssc_encoder import DSSCEncoder, BitStreamReader
from src.engine.dssc_decoder import DSSCDecoder, BitStreamWriter
from src.engine.vcp_payload import VCPPayloadMapper
from src.corpus.index.unified_index import MediaItem


def test_bitstream_roundtrip():
    data = b"Hello, DSSC World! 1234567890"
    reader = BitStreamReader(data)
    writer = BitStreamWriter()

    while reader.has_bits:
        # Read variable bit lengths (e.g. 5 bits, 7 bits, 3 bits)
        chunk_size = 5
        sym = reader.read_bits(chunk_size)
        writer.append_bits(sym, chunk_size)

    recovered = writer.to_bytes()
    assert recovered[: len(data)] == data


def test_dssc_state_space_mapping():
    candidates = [f"item_{i:04d}" for i in range(100)]
    session_key = b"super_secret_session_key_123456"
    perm = derive_session_permutation(len(candidates), session_key, context_salt="test:salt")

    # 100 candidates -> floor(log2(100)) = 6 bits -> 64 states
    state_space = DSSCStateSpace(
        chunk_index=0,
        family_name="test_family",
        candidate_media_ids=candidates,
        permuted_indices=perm,
        bits_per_carrier=6,
    )

    assert state_space.capacity == 6
    assert state_space.state_count == 64

    # Every symbol in [0, 63] should map to a valid candidate and invert back uniquely
    seen_carriers = set()
    for sym in range(64):
        cid = state_space.symbol_to_media_id(sym)
        assert cid in candidates
        assert cid not in seen_carriers
        seen_carriers.add(cid)

        # Invert
        inv_sym = state_space.media_id_to_symbol(cid)
        assert inv_sym == sym


def _make_mock_index_and_mapper(n_per_family: int = 50):
    cluster_map: dict[str, int] = {}
    all_ids: list[str] = []
    metadata = {"image": [], "text": [], "audio": []}
    items = {}

    for fam in DEFAULT_SEMANTIC_FAMILIES:
        cluster_id = fam.cluster_ids[0]
        for i in range(n_per_family):
            mid = f"{fam.name}_{i}"
            cluster_map[mid] = cluster_id
            all_ids.append(mid)
            meta = {"id": mid, "content": f"{fam.name} content {i}", "family": fam.name}
            metadata["image"].append(meta)
            items[mid] = MediaItem(
                id=mid,
                modality="image",
                content=f"{fam.name} content {i}",
                score=1.0,
                normalized_score=1.0,
                metadata=meta,
            )

    mapper = MagicMock(spec=VCPPayloadMapper)
    mapper.symbol_for_media_id.side_effect = lambda mid: cluster_map.get(mid)

    index = MagicMock()
    index.metadata = metadata
    index.get_by_id.side_effect = lambda item_id: items.get(item_id)

    return index, mapper, all_ids, cluster_map


def test_dssc_end_to_end_exact_recovery():
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    session_key = b"shared_symmetric_session_key_32b"
    secret_message = "DSSC achieves 100% deterministic payload recovery with semantic state spaces!"

    # Encode
    res = encoder.encode(
        message=secret_message,
        session_key=session_key,
        ecc_parity_bytes=8,
    )

    assert len(res.carrier_ids) > 0
    assert res.bits_per_carrier_avg > 0

    # Decode
    dec_res = decoder.decode(
        carrier_ids=res.carrier_ids,
        session_key=session_key,
        ecc_parity_bytes=8,
    )

    assert dec_res.success is True
    assert dec_res.reconstructed_message == secret_message
    assert dec_res.verification_rate == 1.0


def test_dssc_reed_solomon_error_correction():
    mock_idx, mapper, all_ids, cluster_map = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    session_key = b"session_key_for_ecc_test_1234567"
    secret_message = "Resilient semantic transport under channel noise."

    res = encoder.encode(
        message=secret_message,
        session_key=session_key,
        ecc_parity_bytes=8,
    )

    # Corrupt 1 carrier in the sequence
    corrupted_ids = list(res.carrier_ids)
    # Find an ID from a different family
    orig_sym = mapper.symbol_for_media_id(corrupted_ids[1])
    diff_id = next(mid for mid, sym in cluster_map.items() if sym != orig_sym)
    corrupted_ids[1] = diff_id

    dec_res = decoder.decode(
        carrier_ids=corrupted_ids,
        session_key=session_key,
        ecc_parity_bytes=8,
    )

    assert dec_res.success is True
    assert dec_res.reconstructed_message == secret_message


def test_dssc_session_key_confidentiality():
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    key_alice = b"alice_and_bob_shared_key_secret!"
    key_eve = b"eve_unauthorized_attacker_key!!!"
    secret_message = "Confidential directive: execute protocol at midnight."

    res = encoder.encode(
        message=secret_message,
        session_key=key_alice,
        ecc_parity_bytes=8,
    )

    # Attacker tries to decode with wrong key
    dec_res_eve = decoder.decode(
        carrier_ids=res.carrier_ids,
        session_key=key_eve,
        ecc_parity_bytes=8,
    )

    assert dec_res_eve.success is False
    assert dec_res_eve.reconstructed_message is None


def test_encoder_uses_semantic_family_matching():
    """Encoder should assign family based on chunk semantics."""
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)

    message = "Let's have a meeting with the team about the project"
    res = encoder.encode(message=message, session_key=b"test_session_key_32bytes_long!")

    first_chunk_carrier = res.encoded_carriers[0]
    assert first_chunk_carrier.family == "people_interaction", \
        f"First chunk should use people_interaction family, got {first_chunk_carrier.family}"

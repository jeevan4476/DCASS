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
    """Encoder should deterministically assign family from valid semantic families."""
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)

    message = "Let's have a meeting with the team about the project"
    res = encoder.encode(message=message, session_key=b"test_session_key_32bytes_long!")

    first_chunk_carrier = res.encoded_carriers[0]
    valid_families = {fam.name for fam in DEFAULT_SEMANTIC_FAMILIES}
    assert first_chunk_carrier.family in valid_families, \
        f"First chunk should use a valid family, got {first_chunk_carrier.family}"


def test_dssc_reed_solomon_erasure_recovery():
    """DSSC recovers dropped carriers when erasure positions are passed to RS-ECC."""
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    session_key = b"session_key_for_erasure_test_123"
    secret_message = "DSSC erasure recovery recovers dropped packets deterministically."

    res = encoder.encode(
        message=secret_message,
        session_key=session_key,
        ecc_parity_bytes=12,
    )

    drop_positions = [1, 4, 7]
    received = [None if i in drop_positions else cid for i, cid in enumerate(res.carrier_ids)]
    dec_res = decoder.decode(
        carrier_ids=received,
        session_key=session_key,
        ecc_parity_bytes=12,
        erasure_positions=drop_positions,
    )

    assert dec_res.success is True
    assert dec_res.reconstructed_message == secret_message
    assert dec_res.verification_rate < 1.0
    assert len(dec_res.ecc_fixed_errors) > 0


def test_dssc_high_bpc_phantom_byte_erasure_boundary():
    """bpc > 8 phantom-byte boundary: the final carrier may append >= 8
    zero-padding bits, which must NOT materialize a phantom 0x00 byte beyond
    the true RS codeword length L (that phantom byte used to count as an extra
    erasure/error and break the 8/8 erasure boundary).
    """
    from src.engine.payload_framing import HEADER_LEN

    # 512 candidates per family -> bpc = floor(log2(512)) = 9 > 8.
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper(n_per_family=512)
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    session_key = b"phantom_byte_session_key_000000"
    # 13-byte message -> framed = 5 + 13 = 18 bytes -> codeword L = 18 + 8 = 26.
    # 8*L = 208 bits = 23*9 + 1, so the final (24th) carrier carries 1 real bit
    # plus 8 zero-padding bits -> BitStreamWriter.to_bytes() would emit L + 1
    # = 27 bytes with a phantom 0x00 at index 26.
    msg = "phantom byte!"
    assert len(msg.encode("utf-8")) == 13
    res = encoder.encode(message=msg, session_key=session_key, ecc_parity_bytes=8)
    L = HEADER_LEN + len(msg.encode("utf-8")) + 8
    assert L == 26 and (8 * L) % 9 == 1  # phantom-byte precondition

    # Clean decode: raw_payload_bytes must be exactly L, never L + 1.
    dec_clean = decoder.decode(
        carrier_ids=res.carrier_ids, session_key=session_key, ecc_parity_bytes=8
    )
    assert dec_clean.success is True
    assert dec_clean.reconstructed_message == msg
    assert len(dec_clean.raw_payload_bytes) == L

    # Drop the final carrier plus the 7 carriers before it: their combined byte
    # span is exactly R = 8 codeword bytes in the true L-byte view (bytes
    # 18..25), but 9 bytes in the buggy L+1 view (phantom byte 26 included) —
    # over the R = 8 erasure capacity, so only the fixed decoder recovers 100%.
    M = len(res.carrier_ids)
    assert M == 24
    drop = list(range(M - 8, M))
    received = [None if i in drop else cid for i, cid in enumerate(res.carrier_ids)]
    dec_res = decoder.decode(
        carrier_ids=received,
        session_key=session_key,
        ecc_parity_bytes=8,
        erasure_positions=drop,
    )
    assert dec_res.success is True
    assert dec_res.reconstructed_message == msg
    assert len(dec_res.raw_payload_bytes) == L


def test_dssc_fail_closed_with_zero_parity_and_wrong_key():
    """ecc_parity_bytes=0 must still fail closed on a wrong session key: the
    DSSC encoder always emits a CRC-16 frame, so garbage must never be
    accepted as a successful decode."""
    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)

    key_alice = b"alice_zero_parity_key_000000000"
    key_eve = b"eve_zero_parity_key_111111111"
    msg = "Zero-parity frames still fail closed on a wrong key."

    res = encoder.encode(message=msg, session_key=key_alice, ecc_parity_bytes=0)
    assert res.parity_bytes == 0

    # Right key: the CRC-16 frame validates -> success.
    dec_ok = decoder.decode(
        carrier_ids=res.carrier_ids, session_key=key_alice, ecc_parity_bytes=0
    )
    assert dec_ok.success is True
    assert dec_ok.reconstructed_message == msg

    # Wrong key: fail closed — never report a "reconstructed" message.
    dec_eve = decoder.decode(
        carrier_ids=res.carrier_ids, session_key=key_eve, ecc_parity_bytes=0
    )
    assert dec_eve.success is False
    assert dec_eve.reconstructed_message is None


def test_dssc_long_message_default_and_adaptive_parity_roundtrip():
    """Messages > 70 bytes round-trip with matching defaults (R = 8 on both
    sides for any length) and with opt-in adaptive parity
    (ecc_parity_bytes=None), which the decoder must infer end to end."""
    from src.engine.dssc_state_space import adaptive_ecc_parity
    from src.engine.payload_framing import HEADER_LEN

    mock_idx, mapper, _, _ = _make_mock_index_and_mapper()
    encoder = DSSCEncoder(index=mock_idx, vcp_mapper=mapper)
    decoder = DSSCDecoder(index=mock_idx, vcp_mapper=mapper)
    session_key = b"long_message_roundtrip_key_00"

    msg = (
        "This is a deliberately long secret message well beyond seventy bytes "
        "so the adaptive parity path is exercised end to end by the decoder."
    )
    assert len(msg.encode("utf-8")) > 70

    # Default parity must match on both sides (8), independent of length.
    res_def = encoder.encode(message=msg, session_key=session_key)
    assert res_def.parity_bytes == 8
    dec_def = decoder.decode(carrier_ids=res_def.carrier_ids, session_key=session_key)
    assert dec_def.success is True
    assert dec_def.reconstructed_message == msg
    assert dec_def.parity_bytes == 8

    # Opt-in adaptive parity: the encoder picks R from the framed length and
    # the decoder (ecc_parity_bytes=None) must infer the same R.
    res_ad = encoder.encode(message=msg, session_key=session_key, ecc_parity_bytes=None)
    expected_r = adaptive_ecc_parity(HEADER_LEN + len(msg.encode("utf-8")))
    assert res_ad.parity_bytes == expected_r
    dec_ad = decoder.decode(
        carrier_ids=res_ad.carrier_ids, session_key=session_key, ecc_parity_bytes=None
    )
    assert dec_ad.success is True
    assert dec_ad.reconstructed_message == msg
    assert dec_ad.parity_bytes == expected_r

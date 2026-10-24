from unittest.mock import MagicMock


def _make_vcp_mapper(cluster_map: dict[str, int]):
    """Create a mock VCPPayloadMapper with a fixed id -> cluster mapping."""
    mapper = MagicMock()
    mapper.symbol_for_media_id.side_effect = lambda mid: cluster_map.get(mid)
    return mapper


def _make_index(media_ids: list[str]):
    index = MagicMock()
    index.metadata = {
        "image": [{"id": mid} for mid in media_ids],
    }
    index.get_by_id.return_value = MagicMock(content="some content")
    return index


def test_encoder_uses_vcp_cluster_for_candidates():
    """
    Encoder must use VCP cluster membership, not modulo-hash, for candidate partition.
    Carriers in cluster 5 (nature_outdoor range 0-41) must be selected when the
    chunk matches 'nature_outdoor' family. Tech carriers in cluster 200 must not be selected.
    """
    from src.engine.dssc_encoder import DSSCEncoder

    nature_ids = [f"nature_item_{i}" for i in range(10)]
    tech_ids = [f"tech_item_{i}" for i in range(10)]
    media_ids = nature_ids + tech_ids
    cluster_map = {mid: 5 for mid in nature_ids}
    cluster_map.update({mid: 200 for mid in tech_ids})

    index = _make_index(media_ids)
    vcp_mapper = _make_vcp_mapper(cluster_map)

    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager
    family_manager = SemanticFamilyManager(families=[DEFAULT_SEMANTIC_FAMILIES[0]])
    enc = DSSCEncoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    result = enc.encode("forest river outdoor", session_key=b"testkey12345678!")
    # Nature items (cluster 5 ∈ 0-41) must appear in carrier ids
    assert len(result.carrier_ids) > 0
    assert all(cid in nature_ids for cid in result.carrier_ids)
    assert not any(cid in tech_ids for cid in result.carrier_ids)


def test_encoder_fallback_on_sparse_family():
    """
    When fewer than 8 carriers exist in the matching cluster family,
    encoder must fallback to all_ids[:256].
    """
    from src.engine.dssc_encoder import DSSCEncoder

    # Only 3 nature items, 15 tech items
    nature_ids = [f"nature_item_{i}" for i in range(3)]
    tech_ids = [f"tech_item_{i}" for i in range(15)]
    media_ids = nature_ids + tech_ids
    cluster_map = {mid: 5 for mid in nature_ids}
    cluster_map.update({mid: 200 for mid in tech_ids})

    index = _make_index(media_ids)
    vcp_mapper = _make_vcp_mapper(cluster_map)

    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager
    family_manager = SemanticFamilyManager(families=[DEFAULT_SEMANTIC_FAMILIES[0]])
    enc = DSSCEncoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    result = enc.encode("forest river outdoor", session_key=b"testkey12345678!")
    # With fallback to all_ids[:256], carriers are drawn from the full pool
    assert len(result.carrier_ids) > 0


def test_encoder_vcp_mapper_init_default():
    """
    DSSCEncoder.__init__ without vcp_mapper should default to VCPPayloadMapper(index).
    """
    from src.engine.dssc_encoder import DSSCEncoder
    from src.engine.vcp_payload import VCPPayloadMapper

    index = _make_index(["item_1"])
    enc = DSSCEncoder(index=index)
    assert isinstance(enc.vcp_mapper, VCPPayloadMapper)
    assert enc.vcp_mapper.index is index


def test_family_selection_not_message_derived():
    """
    Behavioral: carrier family selection must derive from the session key
    and the carrier index, never from message content.

    Two different messages encoded with the SAME session key must produce
    the same family stream (same key + same position => same HMAC family);
    the same message under a DIFFERENT key must not. A message-keyword
    selector would fail the first check, and a constant selector would fail
    the variation check.
    """
    from src.engine.dssc_encoder import DSSCEncoder

    media_ids = [f"item_{i:02d}" for i in range(24)]
    cluster_map = {mid: 5 for mid in media_ids}

    index = _make_index(media_ids)
    vcp_mapper = _make_vcp_mapper(cluster_map)

    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager
    # Both image-native families so the HMAC stream can really alternate.
    family_manager = SemanticFamilyManager(
        families=[DEFAULT_SEMANTIC_FAMILIES[0], DEFAULT_SEMANTIC_FAMILIES[1]]
    )
    enc = DSSCEncoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    key_a = b"session-key-aaaa"
    key_b = b"session-key-bbbb"
    msg_outdoor = "forest river outdoor camping trip with friends"
    msg_tech = "server database bitcoin mining technology update"

    def families(message, key):
        result = enc.encode(message, session_key=key)
        return [c.family for c in result.encoded_carriers]

    fam_outdoor_a = families(msg_outdoor, key_a)
    fam_tech_a = families(msg_tech, key_a)
    fam_outdoor_b = families(msg_outdoor, key_b)

    assert len(fam_outdoor_a) >= 10, "need enough carriers for a meaningful check"
    assert len(set(fam_outdoor_a)) > 1, (
        "HMAC family selection should exercise both admissible families; "
        "a constant family stream means selection is not key-derived"
    )

    k = min(len(fam_outdoor_a), len(fam_tech_a))
    assert fam_outdoor_a[:k] == fam_tech_a[:k], (
        "family stream changed with message content - selection must be "
        "HMAC-keyed on (session_key, carrier index), not message-derived"
    )

    k2 = min(len(fam_outdoor_a), len(fam_outdoor_b))
    assert fam_outdoor_a[:k2] != fam_outdoor_b[:k2], (
        "family stream did not change with the session key - selection "
        "must be key-derived"
    )


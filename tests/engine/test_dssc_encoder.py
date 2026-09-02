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
    """Encoder must not use message keywords to select carrier family."""
    import inspect
    from src.engine.dssc_encoder import DSSCEncoder
    source = inspect.getsource(DSSCEncoder.encode)
    assert "match_families(chunk.text)" not in source, (
        "DSSCEncoder.encode() must not call match_families(chunk.text) — "
        "this leaks message topic. Use HMAC-keyed family selection."
    )


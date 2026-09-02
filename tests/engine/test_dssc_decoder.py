from unittest.mock import MagicMock, patch


def _make_vcp_mapper(cluster_map: dict[str, int]):
    """Create a mock VCPPayloadMapper with a fixed id -> cluster mapping."""
    mapper = MagicMock()
    mapper.symbol_for_media_id.side_effect = lambda mid: cluster_map.get(mid)
    return mapper


def _make_index(media_ids: list[str], content_override: dict[str, str] = None):
    index = MagicMock()
    index.metadata = {
        "image": [{"id": mid} for mid in media_ids],
    }
    content_map = content_override or {}

    def get_item(mid):
        item = MagicMock()
        item.content = content_map.get(mid, "outdoor forest trees")
        item.metadata = {}
        return item

    index.get_by_id.side_effect = get_item
    return index


def test_decoder_uses_hmac_family_not_caption():
    """
    Decoder must resolve family deterministically via session_key HMAC,
    independent of caption text.
    """
    from src.engine.dssc_decoder import DSSCDecoder
    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager, derive_session_permutation

    nature_ids = [f"nature_item_{i}" for i in range(10)]
    tech_ids = [f"tech_item_{i}" for i in range(10)]
    media_ids = nature_ids + tech_ids
    cluster_map = {mid: 5 for mid in nature_ids}
    cluster_map.update({mid: 200 for mid in tech_ids})

    # Carrier has tech-related content but family manager has nature family
    content_override = {"nature_item_0": "computer technology software science code"}
    index = _make_index(media_ids, content_override=content_override)
    vcp_mapper = _make_vcp_mapper(cluster_map)

    family_manager = SemanticFamilyManager(families=[DEFAULT_SEMANTIC_FAMILIES[0]])
    dec = DSSCDecoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    with patch("src.engine.dssc_decoder.derive_session_permutation", wraps=derive_session_permutation) as mock_perm:
        result = dec.decode(
            carrier_ids=["nature_item_0"],
            session_key=b"testkey12345678!",
        )

        # Permutation context salt must use nature_outdoor family from HMAC/family_manager
        mock_perm.assert_called_once()
        _, kwargs = mock_perm.call_args
        context_salt = kwargs.get("context_salt")
        assert context_salt == "dssc:0:nature_outdoor"

    assert result is not None


def test_decoder_vcp_mapper_init_default():
    """
    DSSCDecoder.__init__ without vcp_mapper should default to VCPPayloadMapper(index).
    """
    from src.engine.dssc_decoder import DSSCDecoder
    from src.engine.vcp_payload import VCPPayloadMapper

    index = _make_index(["item_1"])
    dec = DSSCDecoder(index=index)
    assert isinstance(dec.vcp_mapper, VCPPayloadMapper)
    assert dec.vcp_mapper.index is index


def test_decoder_fallback_on_sparse_family():
    """
    When fewer than 8 carriers exist in the matching cluster family,
    decoder must fallback to all_ids[:256].
    """
    from src.engine.dssc_decoder import DSSCDecoder
    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager, derive_session_permutation

    # Only 3 nature items, 15 tech items
    nature_ids = [f"nature_item_{i}" for i in range(3)]
    tech_ids = [f"tech_item_{i}" for i in range(15)]
    media_ids = nature_ids + tech_ids
    cluster_map = {mid: 5 for mid in nature_ids}
    cluster_map.update({mid: 200 for mid in tech_ids})

    index = _make_index(media_ids)
    vcp_mapper = _make_vcp_mapper(cluster_map)

    family_manager = SemanticFamilyManager(families=[DEFAULT_SEMANTIC_FAMILIES[0]])
    dec = DSSCDecoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    with patch("src.engine.dssc_decoder.derive_session_permutation", wraps=derive_session_permutation) as mock_perm:
        dec.decode(
            carrier_ids=["nature_item_0"],
            session_key=b"testkey12345678!",
        )
        # Permutation should have been derived on fallback pool (18 items)
        mock_perm.assert_called_once()
        args, kwargs = mock_perm.call_args
        assert args[0] == len(media_ids)
        assert kwargs.get("context_salt") == "dssc:0:nature_outdoor"


def test_decoder_modality_filtering():
    """
    Decoder must filter candidates by requested modalities when specified.
    """
    from src.engine.dssc_decoder import DSSCDecoder
    from src.engine.dssc_state_space import DEFAULT_SEMANTIC_FAMILIES, SemanticFamilyManager, derive_session_permutation

    index = MagicMock()
    index.metadata = {
        "image": [{"id": f"img_{i}"} for i in range(10)],
        "text": [{"id": f"txt_{i}"} for i in range(10)],
    }
    index.get_by_id.return_value = MagicMock(content="sample")

    cluster_map = {f"img_{i}": 5 for i in range(10)}
    cluster_map.update({f"txt_{i}": 5 for i in range(10)})
    vcp_mapper = _make_vcp_mapper(cluster_map)

    family_manager = SemanticFamilyManager(families=[DEFAULT_SEMANTIC_FAMILIES[0]])
    dec = DSSCDecoder(index=index, vcp_mapper=vcp_mapper, family_manager=family_manager)

    with patch("src.engine.dssc_decoder.derive_session_permutation", wraps=derive_session_permutation) as mock_perm:
        dec.decode(
            carrier_ids=["img_0"],
            session_key=b"testkey12345678!",
            modalities=["image"],
        )
        mock_perm.assert_called_once()
        args, _ = mock_perm.call_args
        # Should only have the 10 image candidates
        assert args[0] == 10


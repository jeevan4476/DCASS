# tests/engine/test_encoder_decoder_roundtrip.py
"""Integration smoke test — no index loaded; checks wiring only."""
from unittest.mock import MagicMock

import pytest


def make_mock_index():
    index = MagicMock()
    index.search.return_value = []
    index.get_by_id.return_value = None
    index.load.return_value = {"image": True}
    index.status.return_value = {}
    index.normalizer = MagicMock()
    index.normalizer.normalize.return_value = 0.5
    index.indices = {}
    index.metadata = {}
    return index


def test_encoder_no_legacy_param():
    """encode() must not accept payload_mode='semantic_legacy'."""
    from src.engine.encoder import SemanticEncoder

    enc = SemanticEncoder(index=make_mock_index())
    enc._loaded = True
    with pytest.raises((TypeError, ValueError)):
        enc.encode("alpha bravo", payload_mode="semantic_legacy")


def test_decoder_no_raw_codeword_param():
    """decode() must not have raw_codeword parameter."""
    from src.engine.decoder import SemanticDecoder
    import inspect
    dec = SemanticDecoder(index=make_mock_index())
    sig = inspect.signature(dec.decode)
    assert "raw_codeword" not in sig.parameters

"""Tests for SemanticEngine unified facade."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight structural tests (no corpus load needed)
# ---------------------------------------------------------------------------

def test_import_and_instantiate():
    """SemanticEngine must import and instantiate without loading indices."""
    from src.engine.semantic_engine import SemanticEngine
    engine = SemanticEngine()
    assert engine is not None


def test_unknown_mode_raises_on_encode():
    """encode() with an unrecognised mode must raise ValueError immediately."""
    from src.engine.semantic_engine import SemanticEngine
    engine = SemanticEngine()
    engine._loaded = True  # Mock load state
    with pytest.raises(ValueError, match="Unknown mode"):
        engine.encode("hello", mode="invalid_mode")


def test_dssc_requires_session_key():
    """DSSC mode must raise ValueError when session_key is None."""
    from src.engine.semantic_engine import SemanticEngine
    engine = SemanticEngine()
    engine._loaded = True  # Mock load state
    with pytest.raises(ValueError, match="session_key"):
        engine.encode("hello", mode="dssc", session_key=None)


def test_unified_encoding_result_fields():
    """UnifiedEncodingResult must expose mode, media_ids, carrier_count."""
    from src.engine.semantic_engine import UnifiedEncodingResult
    r = UnifiedEncodingResult(
        mode="exact_vcp",
        media_ids=["a", "b"],
        carrier_count=2,
        bits_per_carrier=8.0,
        ecc_parity_bytes=8,
        payload_bytes=[1, 2],
        context_info={},
    )
    assert r.mode == "exact_vcp"
    assert r.carrier_count == 2


def test_unified_decoding_result_fields():
    """UnifiedDecodingResult must expose mode, reconstructed_message, success."""
    from src.engine.semantic_engine import UnifiedDecodingResult
    r = UnifiedDecodingResult(
        mode="dssc",
        reconstructed_message="hello",
        success=True,
        verification_rate=1.0,
        ecc_fixed_errors=[],
    )
    assert r.success is True
    assert r.reconstructed_message == "hello"


# ---------------------------------------------------------------------------
# Live Integration and Carrier Count Benchmark Tests
# ---------------------------------------------------------------------------

def _has_indices() -> bool:
    try:
        from src.corpus.index.unified_index import resolve_indices_base_path
        p = resolve_indices_base_path()
        return (p / "image.index").exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def live_engine():
    """A loaded SemanticEngine for integration tests."""
    if not _has_indices():
        pytest.skip("FAISS indices not found — run scripts/data/rebuild_all_indices_gpu.py first")
    from src.engine.semantic_engine import SemanticEngine
    engine = SemanticEngine()
    engine.load()
    return engine


@pytest.mark.integration
def test_exact_vcp_roundtrip(live_engine):
    """exact_vcp mode: encode then decode recovers exact message."""
    message = "Attack at dawn"
    enc = live_engine.encode(message, mode="exact_vcp", use_ecc=True)
    assert enc.mode == "exact_vcp"
    dec = live_engine.decode(enc.media_ids, mode="exact_vcp", use_ecc=True)
    assert dec.success is True
    assert dec.reconstructed_message == message


@pytest.mark.integration
def test_dssc_roundtrip_exact_recovery(live_engine):
    """DSSC mode: encode then decode recovers exact message via CRC frame."""
    import os
    session_key = os.urandom(32)
    message = "Attack at dawn"
    enc = live_engine.encode(message, mode="dssc", session_key=session_key, use_ecc=True)
    assert enc.mode == "dssc"
    dec = live_engine.decode(enc.media_ids, mode="dssc", session_key=session_key, use_ecc=True)
    assert dec.success is True
    assert dec.reconstructed_message == message


@pytest.mark.integration
@pytest.mark.parametrize("message", [
    "Hi",
    "Attack at dawn",
    "The quick brown fox jumps over the lazy dog",
    "A" * 64,  # exactly 64 bytes
])
def test_dssc_carrier_count_benchmark(live_engine, message):
    """DSSC carrier efficiency benchmark: significantly fewer carriers than exact_vcp."""
    import os
    session_key = os.urandom(32)
    enc_dssc = live_engine.encode(message, mode="dssc", session_key=session_key, use_ecc=True)
    enc_vcp = live_engine.encode(message, mode="exact_vcp", use_ecc=True)

    # DSSC must produce substantially fewer carriers than exact_vcp (1 byte per carrier)
    assert enc_dssc.carrier_count < enc_vcp.carrier_count
    assert enc_dssc.bits_per_carrier > 8.0  # Multi-bit symbols


@pytest.mark.integration
def test_dssc_wrong_key_fails(live_engine):
    """Decoding DSSC with a different session key must not return the original message."""
    import os
    session_key = os.urandom(32)
    wrong_key = os.urandom(32)
    message = "Attack at dawn"
    enc = live_engine.encode(message, mode="dssc", session_key=session_key, use_ecc=True)
    dec = live_engine.decode(enc.media_ids, mode="dssc", session_key=wrong_key, use_ecc=True)
    # Either decoding fails OR text is wrong (both are acceptable security outcomes)
    if dec.success:
        assert dec.reconstructed_message != message


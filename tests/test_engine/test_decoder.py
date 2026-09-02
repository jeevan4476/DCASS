# tests/test_engine/test_decoder.py
"""
Unit tests for SemanticDecoder.

Tests cover:
- Decoder initialization
- Loading indices
- Basic decoding
- Content extraction for different modalities
- Verification functionality
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch

from src.engine.decoder import (
    SemanticDecoder,
    DecodingResult,
    DecodedItem,
    decode_media_sequence
)
from src.corpus.index.unified_index import MediaItem


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_index():
    """Create a mock UnifiedSemanticIndex."""
    index = Mock()
    index.load.return_value = {"image": True, "text": True, "audio": True}
    index.status.return_value = {"loaded": True}

    # Mock get_by_id to return media items
    def mock_get_by_id(item_id):
        if item_id.startswith("image_"):
            return MediaItem(
                id=item_id,
                modality="image",
                content="path/to/image.jpg",
                score=1.0,
                normalized_score=1.0,
                metadata={"caption": "A beautiful sunset over the ocean"}
            )
        elif item_id.startswith("text_"):
            return MediaItem(
                id=item_id,
                modality="text",
                content="The quick brown fox jumps over the lazy dog.",
                score=1.0,
                normalized_score=1.0,
                metadata={"text": "The quick brown fox jumps over the lazy dog."}
            )
        elif item_id.startswith("audio_"):
            return MediaItem(
                id=item_id,
                modality="audio",
                content="path/to/audio.mp3",
                score=1.0,
                normalized_score=1.0,
                metadata={"text": "Hello, this is an audio transcription."}
            )
        elif item_id == "unknown_id":
            return None
        else:
            return MediaItem(
                id=item_id,
                modality="text",
                content="Default content",
                score=1.0,
                normalized_score=1.0,
                metadata={}
            )

    index.get_by_id.side_effect = mock_get_by_id
    return index


class FakeDecoderPayloadMapper:
    """Mock payload mapper for unit tests."""

    def __init__(self, mapping: dict[str, int] | None = None):
        self.mapping = mapping or {}

    def symbol_for_media_id(self, media_id: str):
        return self.mapping.get(media_id)

    def decode_symbols(self, media_ids: list[str]):
        symbols = bytearray()
        missing = []
        for media_id in media_ids:
            sym = self.mapping.get(media_id)
            if sym is None:
                missing.append(media_id)
                symbols.append(0)
            else:
                symbols.append(sym)
        return bytes(symbols), missing


@pytest.fixture
def mock_decoder(mock_index):
    """Create a decoder with mock index and fake payload mapper."""
    decoder = SemanticDecoder(index=mock_index)
    decoder._loaded = True
    decoder._payload_mapper = FakeDecoderPayloadMapper()
    return decoder


@pytest.fixture
def loaded_decoder():
    """
    Create a real decoder with loaded indices.

    This fixture loads actual indices, so tests using it
    are integration tests rather than unit tests.
    """
    decoder = SemanticDecoder()
    decoder.load()
    return decoder


# ============================================================
# Legacy Mode Removal Tests
# ============================================================


class TestLegacyModeRemoved:
    """Test that semantic_legacy mode and raw_codeword parameter are removed."""

    def test_legacy_mode_raises(self):
        """semantic_legacy is removed; passing it must raise immediately."""
        dec = SemanticDecoder()
        dec._loaded = True
        with pytest.raises((ValueError, TypeError)):
            dec.decode(["any_id"], payload_mode="semantic_legacy")

    def test_raw_codeword_param_removed(self):
        """raw_codeword side-channel param should no longer exist."""
        import inspect

        sig = inspect.signature(SemanticDecoder.decode)
        assert "raw_codeword" not in sig.parameters

    def test_default_payload_mode_is_exact_vcp(self):
        """decode() default signature must not use semantic_legacy."""
        import inspect

        sig = inspect.signature(SemanticDecoder.decode)
        params = sig.parameters
        if "payload_mode" in params:
            assert params["payload_mode"].default == "exact_vcp"
        else:
            assert "payload_mode" not in params


# ============================================================
# Initialization Tests
# ============================================================

class TestDecoderInit:
    """Test decoder initialization."""

    def test_default_init(self):
        """Test default initialization."""
        decoder = SemanticDecoder()

        assert not decoder.is_loaded()

    def test_with_prebuilt_index(self, mock_index):
        """Test initialization with pre-built index."""
        decoder = SemanticDecoder(index=mock_index)

        assert decoder.index is mock_index


# ============================================================
# Loading Tests
# ============================================================

class TestDecoderLoading:
    """Test decoder loading."""

    def test_load_creates_index(self, mock_index):
        """Test that load creates index if needed."""
        with patch('src.engine.decoder.UnifiedSemanticIndex') as MockIndex:
            MockIndex.return_value = mock_index

            decoder = SemanticDecoder()
            decoder.load()

            assert decoder.is_loaded()

    def test_load_returns_status(self, mock_index):
        """Test that load returns modality status."""
        decoder = SemanticDecoder(index=mock_index)
        status = decoder.load()

        assert "image" in status
        assert "text" in status
        assert "audio" in status

    def test_decode_before_load_raises(self):
        """Test that decoding before load raises error."""
        decoder = SemanticDecoder()

        with pytest.raises(RuntimeError, match="not loaded"):
            decoder.decode(["test_id"])


# ============================================================
# Basic Decoding Tests
# ============================================================

class TestDecoderBasicDecoding:
    """Test basic decoding functionality."""

    def test_decode_single_id(self, mock_decoder):
        """Test decoding a single media ID."""
        result = mock_decoder.decode(["text_00001"])

        assert isinstance(result, DecodingResult)
        assert len(result.decoded) == 1
        assert result.decoded[0].verified

    def test_decode_multiple_ids(self, mock_decoder):
        """Test decoding multiple media IDs."""
        result = mock_decoder.decode(["text_00001", "image_00001", "audio_00001"])

        assert len(result.decoded) == 3
        assert all(item.verified for item in result.decoded)

    def test_decode_returns_contents(self, mock_decoder):
        """Test that decode returns contents."""
        result = mock_decoder.decode(["text_00001"])

        assert len(result.contents) == 1
        assert isinstance(result.contents[0], str)

    def test_decode_to_text(self, mock_decoder):
        """Test decode_to_text convenience method."""
        text = mock_decoder.decode_to_text(["text_00001", "text_00002"])

        assert isinstance(text, str)
        assert "|" in text  # Separator between items


# ============================================================
# Exact VCP Decoding Tests
# ============================================================

class TestExactVCPDecoding:
    """Test exact VCP decoding paths (unkeyed framed, RS-ECC, keyed)."""

    def test_decode_framed_payload_unkeyed(self, mock_index):
        """Test decoding an unkeyed framed payload."""
        from src.engine.payload_framing import frame_payload

        secret_text = "Secret DCASS message"
        framed_bytes = frame_payload(secret_text)

        # Map media IDs to the framed bytes
        mapping = {}
        media_ids = []
        for i, b in enumerate(framed_bytes):
            mid = f"text_{i:05d}"
            media_ids.append(mid)
            mapping[mid] = b

        decoder = SemanticDecoder(index=mock_index)
        decoder._loaded = True
        decoder._payload_mapper = FakeDecoderPayloadMapper(mapping)

        result = decoder.decode(media_ids)
        assert result.ecc_success
        assert result.ecc_payload == secret_text
        assert result.reconstructed_meaning == secret_text
        assert result.payload_mode == "exact_vcp"

    def test_decode_rs_ecc_payload(self, mock_index):
        """Test decoding a payload protected by Reed-Solomon ECC with corruption."""
        from src.engine.ecc import RSErrorCorrection
        from src.engine.payload_framing import frame_payload

        secret_text = "ECC protected"
        framed_bytes = frame_payload(secret_text)
        rs = RSErrorCorrection(parity_bytes=8)
        codeword = rs.encode(framed_bytes)

        # Corrupt 2 bytes
        corrupted = bytearray(codeword)
        corrupted[0] ^= 0xFF
        corrupted[1] ^= 0xFF

        mapping = {}
        media_ids = []
        for i, b in enumerate(corrupted):
            mid = f"text_{i:05d}"
            media_ids.append(mid)
            mapping[mid] = b

        decoder = SemanticDecoder(index=mock_index)
        decoder._loaded = True
        decoder._payload_mapper = FakeDecoderPayloadMapper(mapping)

        result = decoder.decode(media_ids, use_ecc=True, ecc_parity_bytes=8)
        assert result.ecc_success
        assert len(result.ecc_errors_fixed) == 2
        assert result.reconstructed_meaning == secret_text

    def test_decode_keyed_with_context_manager(self, mock_index):
        """Test decoding with dynamic context keying."""
        from src.engine.ecc import RSErrorCorrection
        from src.engine.payload_framing import frame_payload

        secret_text = "Keyed secret"
        mock_ctx = Mock()
        mock_ctx.secret = b"ctx_secret_32b_length_exactly_32"

        epoch = Mock()
        epoch.epoch_id = "2026-09-02T08:00:00Z"
        mock_ctx.candidate_epochs.return_value = [epoch]

        framed = frame_payload(secret_text, secret=mock_ctx.secret)
        rs = RSErrorCorrection(parity_bytes=8)
        codeword = rs.encode(framed)

        # Permutation: cycle shifts bytes by 1
        perm = [(b + 1) % 256 for b in range(256)]
        inv_perm = {perm[i]: i for i in range(256)}
        mock_ctx.derive_inverse_permutation.return_value = inv_perm

        # Permute the framed bytes for transmission
        permuted_bytes = bytes(perm[b] for b in codeword)

        mapping = {}
        media_ids = []
        for i, b in enumerate(permuted_bytes):
            mid = f"text_{i:05d}"
            media_ids.append(mid)
            mapping[mid] = b

        decoder = SemanticDecoder(index=mock_index)
        decoder._loaded = True
        decoder._payload_mapper = FakeDecoderPayloadMapper(mapping)

        result = decoder.decode(media_ids, use_ecc=True, ecc_parity_bytes=8, context_manager=mock_ctx)
        assert result.ecc_success
        assert result.context_epoch_id == "2026-09-02T08:00:00Z"
        assert result.reconstructed_meaning == secret_text


# ============================================================
# Modality Content Extraction Tests
# ============================================================

class TestDecoderContentExtraction:
    """Test content extraction for different modalities."""

    def test_text_content(self, mock_decoder):
        """Test text content extraction."""
        result = mock_decoder.decode(["text_00001"])

        content = result.decoded[0].content
        assert "fox" in content or "dog" in content

    def test_image_content(self, mock_decoder):
        """Test image content extraction (caption)."""
        result = mock_decoder.decode(["image_00001"])

        content = result.decoded[0].content
        assert "sunset" in content or "ocean" in content

    def test_audio_content(self, mock_decoder):
        """Test audio content extraction (transcription)."""
        result = mock_decoder.decode(["audio_00001"])

        content = result.decoded[0].content
        assert "audio" in content or "Hello" in content


# ============================================================
# Verification Tests
# ============================================================

class TestDecoderVerification:
    """Test verification functionality."""

    def test_verified_item(self, mock_decoder):
        """Test that found items are verified."""
        result = mock_decoder.decode(["text_00001"])

        assert result.decoded[0].verified
        assert result.all_verified
        assert result.verification_rate == 1.0

    def test_unverified_item(self, mock_decoder):
        """Test that missing items are not verified."""
        result = mock_decoder.decode(["unknown_id"])

        assert not result.decoded[0].verified
        assert not result.all_verified
        assert result.verification_rate == 0.0

    def test_partial_verification(self, mock_decoder):
        """Test partial verification rate."""
        result = mock_decoder.decode(["text_00001", "unknown_id"])

        assert result.verification_rate == 0.5
        assert not result.all_verified

    def test_verify_sequence(self, mock_decoder):
        """Test verify_sequence method."""
        all_verified, rate = mock_decoder.verify_sequence(["text_00001", "image_00001"])

        assert all_verified
        assert rate == 1.0

    def test_verify_sequence_with_missing(self, mock_decoder):
        """Test verify_sequence with missing items."""
        all_verified, rate = mock_decoder.verify_sequence(["text_00001", "unknown_id"])

        assert not all_verified
        assert rate == 0.5


# ============================================================
# Edge Cases
# ============================================================

class TestDecoderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_list(self, mock_decoder):
        """Test decoding empty list."""
        result = mock_decoder.decode([])

        assert len(result.decoded) == 0
        assert result.verification_rate == 0.0

    def test_unverified_content_format(self, mock_decoder):
        """Test content format for unverified items."""
        result = mock_decoder.decode(["unknown_id"])

        assert "UNVERIFIED" in result.decoded[0].content
        assert "unknown_id" in result.decoded[0].content

    def test_reconstructed_meaning(self, mock_decoder):
        """Test reconstructed meaning generation."""
        result = mock_decoder.decode(["text_00001", "text_00002"])

        meaning = result.reconstructed_meaning
        assert isinstance(meaning, str)
        assert "|" in meaning


# ============================================================
# DecodingResult Tests
# ============================================================

class TestDecodingResult:
    """Test DecodingResult dataclass."""

    def test_summary(self, mock_decoder):
        """Test summary generation."""
        result = mock_decoder.decode(["text_00001"])
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Media IDs" in summary
        assert "Verification" in summary

    def test_repr(self, mock_decoder):
        """Test repr generation."""
        result = mock_decoder.decode(["text_00001"])
        repr_str = repr(result)

        assert "DecodingResult" in repr_str
        assert "items=" in repr_str
        assert "verified=" in repr_str

    def test_contents_excludes_empty(self, mock_decoder):
        """Test that contents excludes empty items."""
        # Mock an item with empty content
        mock_decoder.index.get_by_id.side_effect = lambda x: MediaItem(
            id=x,
            modality="text",
            content="",  # Empty content
            score=1.0,
            normalized_score=1.0,
            metadata={"text": ""}
        )

        result = mock_decoder.decode(["text_00001"])

        # Empty content should be excluded
        assert "" not in result.contents or len(result.contents) == 0


class TestDecodedItem:
    """Test DecodedItem dataclass."""

    def test_repr_verified(self):
        """Test DecodedItem repr for verified item."""
        item = DecodedItem(
            media_id="test_001",
            modality="text",
            content="Test content",
            verified=True
        )
        repr_str = repr(item)

        assert "DecodedItem" in repr_str
        assert "VERIFIED" in repr_str

    def test_repr_unverified(self):
        """Test DecodedItem repr for unverified item."""
        item = DecodedItem(
            media_id="test_001",
            modality="text",
            content="Unknown",
            verified=False
        )
        repr_str = repr(item)

        assert "UNVERIFIED" in repr_str


# ============================================================
# Status Tests
# ============================================================

class TestDecoderStatus:
    """Test status reporting."""

    def test_status_not_loaded(self):
        """Test status when not loaded."""
        decoder = SemanticDecoder()
        status = decoder.status()

        assert status["loaded"] is False

    def test_status_loaded(self, mock_decoder):
        """Test status when loaded."""
        status = mock_decoder.status()

        assert status["loaded"] is True

    def test_repr(self, mock_decoder):
        """Test string representation."""
        repr_str = repr(mock_decoder)

        assert "SemanticDecoder" in repr_str
        assert "loaded" in repr_str


# ============================================================
# Integration Tests (require real indices)
# ============================================================

@pytest.mark.integration
class TestDecoderIntegration:
    """
    Integration tests that use real indices.

    These tests are slower and require the indices to be built.
    """

    @pytest.fixture(autouse=True)
    def check_indices(self):
        """Skip if indices don't exist or codebook fingerprint is stale."""
        import json
        import hashlib
        import numpy as np
        import faiss
        from pathlib import Path

        base = Path(__file__).parent.parent.parent / "storage" / "data" / "indices"
        index_path = base / "text.index"
        if not index_path.exists():
            pytest.skip("Indices not built")

        # Skip if the VCP codebook fingerprint doesn't match the live image index.
        # Exact-VCP mode now always runs the binding check — when the sidecar is
        # stale the test would fail with a RuntimeError, not a test assertion failure.
        sidecar = base / "voronoi_codebook.meta.json"
        img_index = base / "image.index"
        if sidecar.exists() and img_index.exists():
            try:
                meta = json.loads(sidecar.read_text())
                exp = meta.get("index_fingerprints", {}).get("image", {}).get("fingerprint")
                if exp:
                    idx = faiss.read_index(str(img_index))
                    ntotal = int(idx.ntotal)
                    v0 = np.asarray(idx.reconstruct(0), dtype=np.float32).tobytes()
                    vn = np.asarray(idx.reconstruct(ntotal - 1), dtype=np.float32).tobytes() if ntotal > 1 else b""
                    live_fp = hashlib.sha256(v0 + vn + str(ntotal).encode()).hexdigest()[:16]
                    if live_fp != exp:
                        pytest.skip("VCP codebook fingerprint stale — re-bless before running integration tests")
            except Exception:
                pass  # If the check itself fails, let the test run and fail naturally

    def test_real_decode(self, loaded_decoder):
        """Test decoding with real indices."""
        # Use known IDs from the corpus
        # This assumes the indices have been built with flickr8k data
        result = loaded_decoder.decode(["flickr8k_00001"])

        # Item should either be verified or not found
        assert len(result.decoded) == 1


# ============================================================
# Convenience Function Tests
# ============================================================

@pytest.mark.integration
class TestDecodeMediaSequenceFunction:
    """Test the decode_media_sequence convenience function."""

    @pytest.fixture(autouse=True)
    def check_indices(self):
        """Skip if indices don't exist or codebook fingerprint is stale."""
        import json
        import hashlib
        import numpy as np
        import faiss
        from pathlib import Path

        base = Path(__file__).parent.parent.parent / "storage" / "data" / "indices"
        index_path = base / "text.index"
        if not index_path.exists():
            pytest.skip("Indices not built")

        sidecar = base / "voronoi_codebook.meta.json"
        img_index = base / "image.index"
        if sidecar.exists() and img_index.exists():
            try:
                meta = json.loads(sidecar.read_text())
                exp = meta.get("index_fingerprints", {}).get("image", {}).get("fingerprint")
                if exp:
                    idx = faiss.read_index(str(img_index))
                    ntotal = int(idx.ntotal)
                    v0 = np.asarray(idx.reconstruct(0), dtype=np.float32).tobytes()
                    vn = np.asarray(idx.reconstruct(ntotal - 1), dtype=np.float32).tobytes() if ntotal > 1 else b""
                    live_fp = hashlib.sha256(v0 + vn + str(ntotal).encode()).hexdigest()[:16]
                    if live_fp != exp:
                        pytest.skip("VCP codebook fingerprint stale — re-bless before running integration tests")
            except Exception:
                pass  # If the check itself fails, let the test run and fail naturally

    def test_decode_media_sequence_basic(self):
        """Test basic decode_media_sequence usage."""
        result = decode_media_sequence(["flickr8k_00001"])

        assert isinstance(result, DecodingResult)
        assert len(result.decoded) == 1

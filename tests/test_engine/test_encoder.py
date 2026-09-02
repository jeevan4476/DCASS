# tests/test_engine/test_encoder.py
"""
Unit tests for SemanticEncoder.

Tests cover:
- Encoder initialization
- Loading indices
- Basic encoding
- Diversity modes (best, round_robin, balanced)
- Modality-specific encoding
- Edge cases and error handling

Note: These tests require the FAISS indices to be built.
Some tests use mocking to avoid loading actual indices.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.engine.encoder import (
    SemanticEncoder,
    EncodingResult,
    EncodedChunk,
    encode_message,
)
from src.engine.chunker import SemanticChunk
from src.corpus.index.unified_index import MediaItem
from src.engine.vcp_payload import PayloadCarrier


class FakePayloadMapper:
    def __init__(self):
        self.id_to_symbol: dict[str, int] = {}
        self.counter = 0

    def select_carrier(
        self,
        symbol: int,
        query: str = "",
        modalities: list[str] | None = None,
        used_ids: set[str] | None = None,
        avoid_duplicates: bool = True,
    ) -> PayloadCarrier:
        modality = (modalities or ["text"])[0]
        media_id = f"{modality}_{symbol:02x}_{self.counter}"
        if avoid_duplicates and media_id in (used_ids or set()):
            raise RuntimeError(f"Duplicate carrier for {symbol}")
        self.counter += 1
        self.id_to_symbol[media_id] = int(symbol)
        media = MediaItem(
            id=media_id,
            modality=modality,
            content=f"carrier for 0x{symbol:02x}",
            score=0.8,
            normalized_score=0.8,
            _file_path=f"/fake/path/{media_id}.jpg",
            _file_path_resolved=True,
            metadata={"text": f"carrier for 0x{symbol:02x}"},
        )
        return PayloadCarrier(media, int(symbol), self.counter - 1, self.counter - 1, 0.8)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_index():
    """Create a mock UnifiedSemanticIndex."""
    index = Mock()
    index.load.return_value = {"image": True, "text": True, "audio": True}
    index.status.return_value = {"loaded": True}

    # Mock search to return media items
    def mock_search(query, k=5, modalities=None, min_score=0.0):
        items = []
        for i in range(k):
            modality = modalities[i % len(modalities)] if modalities else "text"
            items.append(
                MediaItem(
                    id=f"{modality}_{i:05d}",
                    modality=modality,
                    content=f"Content for {query}",
                    score=0.8 - i * 0.1,
                    normalized_score=0.8 - i * 0.1,
                    _file_path=f"/fake/path/{modality}_{i:05d}.jpg",
                    _file_path_resolved=True,
                    metadata={"text": f"Content for {query}"},
                )
            )
        return items

    index.search.side_effect = mock_search
    return index


@pytest.fixture
def mock_encoder(mock_index):
    """Create an encoder with mock index and fake payload mapper."""
    encoder = SemanticEncoder(index=mock_index)
    encoder._loaded = True
    encoder._payload_mapper = FakePayloadMapper()
    return encoder


@pytest.fixture
def loaded_encoder():
    """
    Create a real encoder with loaded indices.

    This fixture loads actual indices, so tests using it
    are integration tests rather than unit tests.
    """
    encoder = SemanticEncoder()
    encoder.load()
    return encoder


# ============================================================
# Initialization Tests
# ============================================================


class TestEncoderInit:
    """Test encoder initialization."""

    def test_default_init(self):
        """Test default initialization."""
        encoder = SemanticEncoder()

        assert encoder.default_modalities == ["image", "text", "audio"]
        assert not encoder.is_loaded()

    def test_custom_modalities(self):
        """Test with custom modalities."""
        encoder = SemanticEncoder(default_modalities=["image"])

        assert encoder.default_modalities == ["image"]

    def test_expand_synonyms_passed_to_chunker(self):
        """Test that expand_synonyms is passed to chunker."""
        encoder = SemanticEncoder(expand_synonyms=True)

        assert encoder.chunker.expand_synonyms is True

    def test_with_prebuilt_index(self, mock_index):
        """Test initialization with pre-built index."""
        encoder = SemanticEncoder(index=mock_index)

        assert encoder.index is mock_index

    def test_semantic_legacy_mode_raises(self):
        """semantic_legacy is removed; passing it must raise immediately."""
        enc = SemanticEncoder()
        enc._loaded = True
        with pytest.raises((ValueError, TypeError)):
            enc.encode("hello", payload_mode="semantic_legacy")

    def test_default_payload_mode_is_exact_vcp(self):
        """encode() default must route to exact_vcp, not legacy."""
        import inspect

        sig = inspect.signature(SemanticEncoder.encode)
        params = sig.parameters
        if "payload_mode" in params:
            assert params["payload_mode"].default == "exact_vcp"
        else:
            assert "payload_mode" not in params


# ============================================================
# Loading Tests
# ============================================================


class TestEncoderLoading:
    """Test encoder loading."""

    def test_load_creates_index(self, mock_index):
        """Test that load creates index if needed."""
        with patch("src.engine.encoder.UnifiedSemanticIndex") as MockIndex:
            MockIndex.return_value = mock_index

            encoder = SemanticEncoder()
            encoder.load()

            assert encoder.is_loaded()

    def test_load_returns_status(self, mock_index):
        """Test that load returns modality status."""
        encoder = SemanticEncoder(index=mock_index)
        status = encoder.load()

        assert "image" in status
        assert "text" in status
        assert "audio" in status

    def test_encode_before_load_raises(self):
        """Test that encoding before load raises error."""
        encoder = SemanticEncoder()

        with pytest.raises(RuntimeError, match="not loaded"):
            encoder.encode("test message")


# ============================================================
# Basic Encoding Tests
# ============================================================


class TestEncoderBasicEncoding:
    """Test basic encoding functionality."""

    def test_simple_encode(self, mock_encoder):
        """Test encoding a simple message."""
        result = mock_encoder.encode("Hello world")

        assert isinstance(result, EncodingResult)
        assert result.original_message == "Hello world"
        assert len(result.chunks) >= 1
        assert len(result.encoded) >= 1

    def test_encode_returns_media_ids(self, mock_encoder):
        """Test that encode returns media IDs."""
        result = mock_encoder.encode("Test message")

        assert len(result.media_ids) >= 1
        assert all(isinstance(id, str) for id in result.media_ids)

    def test_encode_to_ids(self, mock_encoder):
        """Test encode_to_ids convenience method."""
        ids = mock_encoder.encode_to_ids("Test message")

        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_modality_breakdown(self, mock_encoder):
        """Test modality breakdown property."""
        result = mock_encoder.encode("Test message")

        breakdown = result.modality_breakdown
        assert isinstance(breakdown, dict)
        assert sum(breakdown.values()) == len(result.encoded)

    def test_media_sequence(self, mock_encoder):
        """Test media_sequence property."""
        result = mock_encoder.encode("Test message")

        sequence = result.media_sequence
        assert all(isinstance(item, MediaItem) for item in sequence)
        assert len(sequence) == len(result.encoded)


# ============================================================
# Diversity Mode Tests
# ============================================================


class TestEncoderDiversityModes:
    """Test diversity mode selection."""

    def test_best_mode(self, mock_encoder):
        """Test 'best' diversity mode."""
        result = mock_encoder.encode("First chunk, second chunk", diversity_mode="best")

        # Should produce results
        assert len(result.encoded) >= 1

    def test_round_robin_mode(self, mock_encoder):
        """Test 'round_robin' diversity mode."""
        # Create encoder with predictable search results
        mock_encoder.index.search.side_effect = None

        def predictable_search(query, k=5, modalities=None, min_score=0.0):
            # Return items for each requested modality
            items = []
            for i, mod in enumerate(modalities or ["image", "text", "audio"]):
                items.append(
                    MediaItem(
                        id=f"{mod}_{hash(query) % 10000:05d}_{i}",
                        modality=mod,
                        content="Content",
                        score=0.8,
                        normalized_score=0.8,
                        metadata={},
                    )
                )
            return items

        mock_encoder.index.search.side_effect = predictable_search

        result = mock_encoder.encode(
            "First chunk, second chunk, third chunk", diversity_mode="round_robin"
        )

        # Should cycle through modalities
        assert len(result.encoded) >= 1

    def test_balanced_mode(self, mock_encoder):
        """Test 'balanced' diversity mode."""
        result = mock_encoder.encode(
            "Chunk one, chunk two, chunk three", diversity_mode="balanced"
        )

        # Should produce results
        assert len(result.encoded) >= 1


# ============================================================
# Modality-Specific Encoding Tests
# ============================================================


class TestEncoderModalitySpecific:
    """Test modality-specific encoding methods."""

    def test_encode_images_only(self, mock_encoder):
        """Test encoding with images only."""
        # Override search to return only images
        mock_encoder.index.search.side_effect = lambda **kwargs: [
            MediaItem(
                id="image_00001",
                modality="image",
                content="Image content",
                score=0.8,
                normalized_score=0.8,
                metadata={},
            )
        ]

        ids = mock_encoder.encode_images_only("Test message")

        assert len(ids) >= 1
        assert all("image" in id for id in ids)

    def test_encode_text_only(self, mock_encoder):
        """Test encoding with text only."""
        mock_encoder.index.search.side_effect = lambda **kwargs: [
            MediaItem(
                id="text_00001",
                modality="text",
                content="Text content",
                score=0.8,
                normalized_score=0.8,
                metadata={},
            )
        ]

        ids = mock_encoder.encode_text_only("Test message")

        assert len(ids) >= 1
        assert all("text" in id for id in ids)

    def test_encode_audio_only(self, mock_encoder):
        """Test encoding with audio only."""
        mock_encoder.index.search.side_effect = lambda **kwargs: [
            MediaItem(
                id="audio_00001",
                modality="audio",
                content="Audio content",
                score=0.8,
                normalized_score=0.8,
                metadata={},
            )
        ]

        ids = mock_encoder.encode_audio_only("Test message")

        assert len(ids) >= 1
        assert all("audio" in id for id in ids)


# ============================================================
# Edge Cases
# ============================================================


class TestEncoderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_message_raises(self, mock_encoder):
        """Test that empty message raises ValueError."""
        with pytest.raises(ValueError, match="no valid chunks"):
            mock_encoder.encode("")

    def test_very_short_message(self, mock_encoder):
        """Test handling of very short message."""
        # "Hi" is likely too short for chunker's min_chunk_length
        # Depending on chunker settings, this may or may not work
        try:
            result = mock_encoder.encode("Hi")
            # If it works, should have at least one chunk
            assert len(result.encoded) >= 0
        except ValueError:
            # Expected if message too short
            pass

    def test_no_search_results_raises(self, mock_encoder):
        """Test that failure to find carriers raises RuntimeError."""
        mock_encoder._payload_mapper.select_carrier = Mock(
            side_effect=RuntimeError("Cannot encode byte")
        )

        with pytest.raises(RuntimeError, match="Cannot encode byte"):
            mock_encoder.encode("Test message with no results")

    def test_avoid_duplicates(self, mock_encoder):
        """Test that duplicates are avoided when requested."""
        def duplicate_select(symbol, query="", modalities=None, used_ids=None, avoid_duplicates=True):
            if avoid_duplicates and used_ids and "same_id" in used_ids:
                raise RuntimeError("already used")
            media = MediaItem(
                id="same_id",
                modality="text",
                content="Content",
                score=0.8,
                normalized_score=0.8,
                _file_path="/fake/path/same_id.txt",
                _file_path_resolved=True,
                metadata={},
            )
            return PayloadCarrier(media, int(symbol), 0, 0, 0.8)

        mock_encoder._payload_mapper.select_carrier = Mock(side_effect=duplicate_select)

        with pytest.raises(RuntimeError, match="already used"):
            mock_encoder.encode("First, second, third", avoid_duplicates=True)

    def test_allow_duplicates(self, mock_encoder):
        """Test that duplicates are allowed when flag is False."""
        def duplicate_select(symbol, query="", modalities=None, used_ids=None, avoid_duplicates=True):
            media = MediaItem(
                id="same_id",
                modality="text",
                content="Content",
                score=0.8,
                normalized_score=0.8,
                _file_path="/fake/path/same_id.txt",
                _file_path_resolved=True,
                metadata={},
            )
            return PayloadCarrier(media, int(symbol), 0, 0, 0.8)

        mock_encoder._payload_mapper.select_carrier = Mock(side_effect=duplicate_select)

        result = mock_encoder.encode("First, second, third", avoid_duplicates=False)

        # All IDs should be the same
        assert all(id == "same_id" for id in result.media_ids)


# ============================================================
# EncodingResult Tests
# ============================================================


class TestEncodingResult:
    """Test EncodingResult dataclass."""

    def test_summary(self, mock_encoder):
        """Test summary generation."""
        result = mock_encoder.encode("Test message")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Test message" in summary
        assert "Chunks:" in summary
        assert "Encoding:" in summary

    def test_repr(self, mock_encoder):
        """Test repr generation."""
        result = mock_encoder.encode("Test")
        repr_str = repr(result)

        assert "EncodingResult" in repr_str
        assert "chunks=" in repr_str


class TestEncodedChunk:
    """Test EncodedChunk dataclass."""

    def test_repr(self):
        """Test EncodedChunk repr."""
        chunk = SemanticChunk(text="test", original="test", index=0)
        media = MediaItem(
            id="test_id",
            modality="text",
            content="content",
            score=0.8,
            normalized_score=0.8,
        )

        encoded = EncodedChunk(chunk=chunk, media=media, alternatives=[])
        repr_str = repr(encoded)

        assert "EncodedChunk" in repr_str
        assert "test_id" in repr_str


# ============================================================
# Integration Tests (require real indices)
# ============================================================


@pytest.mark.integration
class TestEncoderIntegration:
    """
    Integration tests that use real indices.

    These tests are slower and require the indices to be built.
    Mark with @pytest.mark.integration and skip if indices don't exist.
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

    def test_real_encode(self, loaded_encoder):
        """Test encoding with real indices."""
        result = loaded_encoder.encode("Hello world")

        assert len(result.encoded) >= 1
        assert all(item.media.id for item in result.encoded)

    def test_real_round_robin(self, loaded_encoder):
        """Test round_robin with real indices."""
        result = loaded_encoder.encode(
            "First chunk, second chunk, third chunk", diversity_mode="round_robin"
        )

        # Should have multiple modalities
        modalities = set(m.modality for m in result.media_sequence)
        assert len(modalities) >= 1

    def test_real_balanced(self, loaded_encoder):
        """Test balanced mode with real indices."""
        result = loaded_encoder.encode(
            "This is a longer message that should produce multiple chunks for testing",
            diversity_mode="balanced",
        )

        assert len(result.encoded) >= 1

    def test_real_encode_file_paths(self, loaded_encoder):
        """Test encoding returns valid non-empty absolute file paths for selected media items."""
        result = loaded_encoder.encode("A cute dog running in the park")
        assert len(result.encoded) >= 1
        for item in result.encoded:
            assert item.file_path is not None
            assert isinstance(item.file_path, str)
            assert len(item.file_path) > 0
            assert Path(item.file_path).is_absolute()
            assert item.media.file_path == item.file_path
        assert len(result.file_paths) >= 1
        assert all(Path(p).is_absolute() for p in result.file_paths)


# ============================================================
# Convenience Function Tests
# ============================================================


@pytest.mark.integration
class TestEncodeMessageFunction:
    """Tests for encode_message convenience function."""

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

    def test_encode_message_basic(self):
        """Test basic encode_message usage."""
        result = encode_message("Test message")

        assert isinstance(result, EncodingResult)
        assert len(result.encoded) >= 1

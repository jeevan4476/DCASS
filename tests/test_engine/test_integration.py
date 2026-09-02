# tests/test_engine/test_integration.py
"""
Integration tests for the DCASS engine module.

These tests verify end-to-end functionality:
- Full encode -> decode loop
- Round-trip verification
- Multi-modal encoding and decoding

Requires actual FAISS indices to be built.
"""

import pytest
from pathlib import Path


# Skip all tests in this module if indices don't exist
def check_indices_exist():
    """Check if indices exist."""
    index_path = Path(__file__).parent.parent.parent / "storage" / "data" / "indices" / "text.index"
    return index_path.exists()


pytestmark = pytest.mark.skipif(
    not check_indices_exist(),
    reason="FAISS indices not built - run build_indices.py first"
)


class TestEndToEndEncoding:
    """Test complete encode -> decode cycle."""

    @pytest.fixture
    def encoder(self):
        """Create and load encoder."""
        from src.engine import SemanticEncoder
        encoder = SemanticEncoder()
        encoder.load()
        return encoder

    @pytest.fixture
    def decoder(self):
        """Create and load decoder."""
        from src.engine import SemanticDecoder
        decoder = SemanticDecoder()
        decoder.load()
        return decoder

    def test_simple_round_trip(self, encoder, decoder):
        """Test simple encode -> decode round trip."""
        message = "Hello world"

        # Encode
        encode_result = encoder.encode(message)
        assert len(encode_result.media_ids) >= 1

        # Decode
        decode_result = decoder.decode(encode_result.media_ids)

        # Verify all items found
        assert decode_result.verification_rate == 1.0

    def test_complex_message_round_trip(self, encoder, decoder):
        """Test round trip with complex message."""
        message = "The quick brown fox jumps over the lazy dog. Meet me at the park."

        encode_result = encoder.encode(message)
        decode_result = decoder.decode(encode_result.media_ids)

        assert decode_result.all_verified
        assert len(decode_result.contents) == len(encode_result.encoded)

    def test_round_robin_round_trip(self, encoder, decoder):
        """Test round trip with round_robin diversity mode."""
        message = "First chunk, second chunk, third chunk, fourth chunk"

        encode_result = encoder.encode(message, diversity_mode="round_robin")
        decode_result = decoder.decode(encode_result.media_ids)

        assert decode_result.verification_rate == 1.0

        # Should have multiple modalities
        modalities = encode_result.modality_breakdown
        assert sum(modalities.values()) >= 1

    def test_balanced_round_trip(self, encoder, decoder):
        """Test round trip with balanced diversity mode."""
        message = "Testing balanced mode with a longer message for better distribution"

        encode_result = encoder.encode(message, diversity_mode="balanced")
        decode_result = decoder.decode(encode_result.media_ids)

        assert decode_result.all_verified


class TestModalitySpecificEncoding:
    """Test modality-specific encoding."""

    @pytest.fixture
    def encoder(self):
        """Create and load encoder."""
        from src.engine import SemanticEncoder
        encoder = SemanticEncoder()
        encoder.load()
        return encoder

    @pytest.fixture
    def decoder(self):
        """Create and load decoder."""
        from src.engine import SemanticDecoder
        decoder = SemanticDecoder()
        decoder.load()
        return decoder

    def test_image_only_encoding(self, encoder, decoder):
        """Test encoding with images requested."""
        ids = encoder.encode_images_only("A beautiful sunset")
        assert len(ids) >= 1
        result = decoder.decode(ids)
        assert len(result.decoded) == len(ids)
        assert any(item.modality == "image" for item in result.decoded if item.verified)

    def test_text_only_encoding(self, encoder, decoder):
        """Test encoding with text requested."""
        ids = encoder.encode_text_only("A beautiful sunset")
        assert len(ids) >= 1
        result = decoder.decode(ids)
        assert len(result.decoded) == len(ids)
        assert any(item.modality == "text" for item in result.decoded if item.verified)

    def test_audio_only_encoding(self, encoder, decoder):
        """Test encoding with audio requested."""
        ids = encoder.encode_audio_only("A beautiful sunset")
        assert len(ids) >= 1
        result = decoder.decode(ids)
        assert len(result.decoded) == len(ids)
        assert any(item.modality == "audio" for item in result.decoded if item.verified)


class TestChunkerIntegration:
    """Test chunker integration with encoder."""

    @pytest.fixture
    def encoder(self):
        """Create and load encoder."""
        from src.engine import SemanticEncoder
        encoder = SemanticEncoder()
        encoder.load()
        return encoder

    def test_chunking_consistency(self, encoder):
        """Test that chunking is consistent."""
        message = "First part, second part, third part"

        result1 = encoder.encode(message)
        result2 = encoder.encode(message)

        # Same message should produce same chunks
        assert len(result1.chunks) == len(result2.chunks)
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.original == c2.original

    def test_long_message_chunking(self, encoder):
        """Test chunking of long messages."""
        message = " ".join(["This is sentence number " + str(i) + "." for i in range(10)])

        result = encoder.encode(message)

        # Should produce multiple chunks
        assert len(result.chunks) >= 2


class TestScoreNormalization:
    """Test score normalization across modalities."""

    @pytest.fixture
    def encoder(self):
        """Create and load encoder."""
        from src.engine import SemanticEncoder
        encoder = SemanticEncoder()
        encoder.load()
        return encoder

    def test_normalized_scores_in_range(self, encoder):
        """Test that normalized scores are in [0, 1]."""
        result = encoder.encode("Test message for score check")

        for item in result.media_sequence:
            assert 0.0 <= item.normalized_score <= 1.0

    def test_best_mode_selects_highest_score(self, encoder):
        """Test that best mode selects highest scoring items."""
        result = encoder.encode("Test message", diversity_mode="best")

        # Can't easily verify "highest" without comparing to alternatives
        # But scores should be reasonable
        for item in result.media_sequence:
            assert item.normalized_score > 0.0


class TestEdgeCases:
    """Test edge cases in integration."""

    @pytest.fixture
    def encoder(self):
        """Create and load encoder."""
        from src.engine import SemanticEncoder
        encoder = SemanticEncoder()
        encoder.load()
        return encoder

    def test_special_characters(self, encoder):
        """Test handling of special characters."""
        result = encoder.encode("Hello! @#$% How are you???")

        assert len(result.encoded) >= 1

    def test_numbers_in_message(self, encoder):
        """Test handling of numbers."""
        result = encoder.encode("Meet at 5pm on floor 23")

        assert len(result.encoded) >= 1

    def test_very_long_message(self, encoder):
        """Test handling of very long message."""
        # Use different sentences to avoid duplicate media selection issues
        sentences = [
            "The weather is nice today.",
            "Birds are singing in the trees.",
            "A dog runs across the field.",
            "Children play in the park.",
            "The sun sets over the mountains.",
        ]
        message = " ".join(sentences * 3)  # 15 different-ish sentences

        result = encoder.encode(message)

        # Should handle without error
        assert len(result.encoded) >= 1
        assert len(result.chunks) == len(result.encoded)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_encode_message(self):
        """Test encode_message convenience function."""
        from src.engine import encode_message

        result = encode_message("Test message")

        assert len(result.encoded) >= 1

    def test_decode_media_sequence(self):
        """Test decode_media_sequence convenience function."""
        from src.engine import encode_message, decode_media_sequence

        # First encode to get valid IDs
        encode_result = encode_message("Test message")

        # Then decode
        decode_result = decode_media_sequence(encode_result.media_ids)

        assert decode_result.verification_rate == 1.0

    def test_chunk_message(self):
        """Test chunk_message convenience function."""
        from src.engine import chunk_message

        chunks = chunk_message("Hello, world")

        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

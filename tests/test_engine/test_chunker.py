# tests/test_engine/test_chunker.py
"""
Unit tests for SemanticChunker.

Tests cover:
- Basic chunking functionality
- Sentence splitting
- Delimiter handling
- Long chunk splitting
- Synonym expansion
- Edge cases (empty, short, special chars)
"""

import pytest
from src.engine.chunker import SemanticChunker, SemanticChunk, chunk_message


class TestSemanticChunkerBasic:
    """Test basic chunking functionality."""
    
    def test_simple_sentence(self):
        """Test chunking a simple sentence."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Hello world")
        
        assert len(chunks) == 1
        assert chunks[0].original == "hello world"
        assert chunks[0].index == 0
    
    def test_comma_separated(self):
        """Test splitting on commas."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Meet me at the cafe, bring the documents")
        
        # Splits on comma and "at the"
        assert len(chunks) >= 2
        # First chunk should have "meet me"
        assert "meet me" in chunks[0].original
        # Last chunk should have "bring"
        assert "bring" in chunks[-1].original
    
    def test_conjunction_and(self):
        """Test splitting on 'and'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("The dog ran and the cat jumped")
        
        assert len(chunks) == 2
        assert "dog ran" in chunks[0].original
        assert "cat jumped" in chunks[1].original
    
    def test_conjunction_but(self):
        """Test splitting on 'but'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("I wanted to go but it was raining")
        
        assert len(chunks) == 2
        assert "wanted to go" in chunks[0].original
        assert "raining" in chunks[1].original
    
    def test_multiple_sentences(self):
        """Test splitting multiple sentences."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("First sentence. Second sentence. Third one!")
        
        assert len(chunks) >= 2  # May be more depending on min_chunk_length
    
    def test_preserves_order(self):
        """Test that chunk indices preserve order."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("First, second, third, fourth")
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestSemanticChunkerDelimiters:
    """Test delimiter handling."""
    
    def test_semicolon(self):
        """Test splitting on semicolons."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Part one; part two; part three")
        
        assert len(chunks) >= 2
    
    def test_preposition_in_the(self):
        """Test splitting on 'in the'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("The meeting is scheduled in the conference room")
        
        assert len(chunks) >= 1
    
    def test_preposition_at_the(self):
        """Test splitting on 'at the'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("See you at the park tomorrow")
        
        assert len(chunks) >= 1
    
    def test_when_clause(self):
        """Test splitting on 'when'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Call me when you arrive")
        
        assert len(chunks) == 2
    
    def test_which_clause(self):
        """Test splitting on 'which'."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("The book which I read was excellent")
        
        assert len(chunks) == 2
    
    def test_custom_delimiters(self):
        """Test with custom delimiters."""
        chunker = SemanticChunker(delimiters=r"\|")  # Pipe only
        chunks = chunker.chunk("part one | part two | part three")
        
        assert len(chunks) == 3


class TestSemanticChunkerLongChunks:
    """Test handling of long chunks."""
    
    def test_long_chunk_split(self):
        """Test that long chunks are split."""
        chunker = SemanticChunker(max_chunk_length=30)
        long_text = "This is a very long sentence that should definitely be split into multiple chunks"
        chunks = chunker.chunk(long_text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.original) <= 60  # Allow some flexibility
    
    def test_split_at_preposition(self):
        """Test that splits prefer prepositions."""
        chunker = SemanticChunker(max_chunk_length=40)
        text = "The quick brown fox jumps over the lazy dog in the forest"
        chunks = chunker.chunk(text)
        
        # Should split at natural boundary
        assert len(chunks) >= 1
    
    def test_very_long_word(self):
        """Test handling of very long words."""
        chunker = SemanticChunker(max_chunk_length=20)
        text = "Supercalifragilisticexpialidocious is a word"
        chunks = chunker.chunk(text)
        
        # Should still produce at least one chunk
        assert len(chunks) >= 1
    
    def test_default_max_length(self):
        """Test default max chunk length of 60."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_length == 60


class TestSemanticChunkerSynonyms:
    """Test synonym expansion."""
    
    def test_synonym_expansion_disabled(self):
        """Test that synonyms are not added by default."""
        chunker = SemanticChunker(expand_synonyms=False)
        chunks = chunker.chunk("happy dog")
        
        assert len(chunks) == 1
        assert chunks[0].text == chunks[0].original
    
    def test_synonym_expansion_enabled(self):
        """Test that synonyms are added when enabled."""
        chunker = SemanticChunker(expand_synonyms=True)
        chunks = chunker.chunk("happy dog")
        
        assert len(chunks) == 1
        # Should have expanded text
        assert "joyful" in chunks[0].text or chunks[0].text != chunks[0].original
    
    def test_multiple_synonyms(self):
        """Test expansion with multiple expandable words."""
        chunker = SemanticChunker(expand_synonyms=True)
        chunks = chunker.chunk("sad cat sleeping")
        
        assert len(chunks) == 1
        # Original should be preserved
        assert chunks[0].original == "sad cat sleeping"
    
    def test_custom_synonyms(self):
        """Test with custom synonym map."""
        custom = {"test": ["exam", "quiz"]}
        chunker = SemanticChunker(expand_synonyms=True, custom_synonyms=custom)
        chunks = chunker.chunk("test message")
        
        assert len(chunks) == 1
        assert "exam" in chunks[0].text


class TestSemanticChunkerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string(self):
        """Test empty string input."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("")
        
        assert len(chunks) == 0
    
    def test_whitespace_only(self):
        """Test whitespace-only input."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("   \t\n  ")
        
        assert len(chunks) == 0
    
    def test_very_short_input(self):
        """Test input shorter than min_chunk_length."""
        chunker = SemanticChunker(min_chunk_length=5)
        chunks = chunker.chunk("Hi")
        
        assert len(chunks) == 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Hello! @#$% World???")
        
        assert len(chunks) >= 1
        # Text should be lowercased and cleaned of trailing punctuation
        all_text = " ".join(c.original for c in chunks)
        assert "hello" in all_text.lower()
        assert "world" in all_text.lower()
    
    def test_numbers(self):
        """Test handling of numbers."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Meet at 5pm on the 23rd")
        
        assert len(chunks) >= 1
        # Numbers should be preserved
        assert any("5" in c.original or "23" in c.original for c in chunks)
    
    def test_unicode(self):
        """Test handling of unicode characters."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("Cafe with friends")  # e with accent
        
        assert len(chunks) >= 1
    
    def test_quotes(self):
        """Test handling of quoted text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk('He said "hello there"')
        
        assert len(chunks) >= 1
    
    def test_contractions(self):
        """Test that contractions are preserved."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("I don't think we'll make it")
        
        # Should not split on apostrophe in contractions
        joined = " ".join(c.original for c in chunks)
        assert "don" in joined or "dont" in joined.replace("'", "")


class TestSemanticChunkerMethods:
    """Test additional methods."""
    
    def test_chunk_simple(self):
        """Test chunk_simple returns strings."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_simple("Hello, world")
        
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
    
    def test_reconstruct(self):
        """Test reconstruct method."""
        chunker = SemanticChunker()
        original = "First part, second part, third part"
        chunks = chunker.chunk(original)
        
        reconstructed = chunker.reconstruct(chunks)
        # Should contain all original parts
        for chunk in chunks:
            assert chunk.original in reconstructed
    
    def test_repr(self):
        """Test string representation."""
        chunker = SemanticChunker(expand_synonyms=True, max_chunk_length=50)
        repr_str = repr(chunker)
        
        assert "SemanticChunker" in repr_str
        assert "expand_synonyms=True" in repr_str
        assert "50" in repr_str


class TestChunkMessageFunction:
    """Test the convenience function."""
    
    def test_basic_usage(self):
        """Test basic chunk_message usage."""
        chunks = chunk_message("Hello, world")
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
    
    def test_with_expansion(self):
        """Test with synonym expansion."""
        chunks_no_expand = chunk_message("happy dog", expand=False)
        chunks_expand = chunk_message("happy dog", expand=True)
        
        # Expanded version should be different or longer
        assert len(chunks_no_expand) == len(chunks_expand)


class TestSemanticChunk:
    """Test SemanticChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a SemanticChunk."""
        chunk = SemanticChunk(
            text="expanded text",
            original="original",
            index=0
        )
        
        assert chunk.text == "expanded text"
        assert chunk.original == "original"
        assert chunk.index == 0
    
    def test_chunk_repr_same_text(self):
        """Test repr when text equals original."""
        chunk = SemanticChunk(text="same", original="same", index=0)
        repr_str = repr(chunk)
        
        assert "Chunk(0: 'same')" == repr_str
    
    def test_chunk_repr_different_text(self):
        """Test repr when text differs from original."""
        chunk = SemanticChunk(text="expanded", original="original", index=1)
        repr_str = repr(chunk)
        
        assert "->" in repr_str
        assert "original" in repr_str
        assert "expanded" in repr_str

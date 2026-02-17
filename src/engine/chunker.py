# src/engine/chunker.py
"""
SemanticChunker - Intelligent message chunking for steganographic encoding.

Breaks down messages into semantic units that can be individually encoded
into media items, with optional synonym expansion for better corpus matches.

Architecture:
    Input Message -> Sentence Split -> Phrase Split -> Expand synonyms -> Semantic Chunks

Improvements (v2):
- Sentence boundary detection
- Maximum chunk length with smart splitting
- Better delimiter patterns including quotes and clauses
- Phrase extraction for complex sentences
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticChunk:
    """Represents a semantic unit extracted from a message."""
    text: str           # The chunk text (possibly expanded with synonyms)
    original: str       # Original text before expansion
    index: int          # Position in the original message
    
    def __repr__(self) -> str:
        if self.text != self.original:
            return f"Chunk({self.index}: '{self.original}' -> '{self.text}')"
        return f"Chunk({self.index}: '{self.text}')"


class SemanticChunker:
    """
    Intelligent chunker for semantic steganography.
    
    Splits messages into semantic units optimized for corpus matching:
    1. First splits on sentence boundaries
    2. Then splits on natural delimiters (commas, conjunctions, etc.)
    3. Enforces maximum chunk length for long phrases
    4. Optionally expands with synonyms for better matches
    5. Cleans and normalizes each chunk
    
    Usage:
        chunker = SemanticChunker()
        chunks = chunker.chunk("Meet me at the cafe, bring the documents")
        # -> [Chunk("meet me at the cafe"), Chunk("bring the documents")]
        
        # With synonym expansion
        chunker = SemanticChunker(expand_synonyms=True)
        chunks = chunker.chunk("happy dog running")
        # -> [Chunk("happy joyful dog running")]
        
        # Long complex sentences are split intelligently
        chunker = SemanticChunker()
        chunks = chunker.chunk("There have been several claims for the longest sentence in English")
        # -> [Chunk("several claims"), Chunk("longest sentence in english")]
    
    Attributes:
        expand_synonyms: Whether to add synonyms to chunks
        min_chunk_length: Minimum characters for a valid chunk
        max_chunk_length: Maximum characters before forcing a split
        delimiters: Regex pattern for splitting
    """
    
    # Common synonyms for expansion (simple dictionary approach)
    # In production, could use WordNet or word2vec
    SYNONYM_MAP = {
        # Emotions
        "happy": ["joyful", "cheerful", "pleased"],
        "sad": ["unhappy", "sorrowful", "melancholy"],
        "angry": ["furious", "enraged", "irritated"],
        "scared": ["frightened", "afraid", "terrified"],
        
        # Actions
        "run": ["running", "sprint", "dash", "jog"],
        "walk": ["walking", "stroll", "wander"],
        "eat": ["eating", "consume", "dine"],
        "sleep": ["sleeping", "rest", "slumber"],
        
        # Objects
        "car": ["vehicle", "automobile"],
        "house": ["home", "building", "residence"],
        "dog": ["canine", "puppy", "hound"],
        "cat": ["feline", "kitten"],
        
        # Nature
        "tree": ["trees", "forest", "woods"],
        "water": ["ocean", "sea", "river", "lake"],
        "mountain": ["mountains", "hill", "peak"],
        "sky": ["clouds", "heaven", "atmosphere"],
        
        # Time
        "morning": ["dawn", "sunrise", "daybreak"],
        "evening": ["dusk", "sunset", "twilight"],
        "night": ["nighttime", "darkness", "midnight"],
        
        # Weather
        "rain": ["raining", "rainy", "storm"],
        "sun": ["sunny", "sunshine", "sunlight"],
        "snow": ["snowy", "winter", "frost"],
        
        # Colors (common in image search)
        "red": ["crimson", "scarlet"],
        "blue": ["azure", "navy"],
        "green": ["emerald", "verdant"],
        
        # Abstract concepts (new)
        "meeting": ["gathering", "conference", "assembly"],
        "secret": ["hidden", "confidential", "private"],
        "important": ["significant", "crucial", "vital"],
        "message": ["communication", "information", "note"],
    }
    
    # Sentence boundary pattern - splits on . ! ? followed by space
    SENTENCE_PATTERN = r'(?<=[.!?])\s+'
    
    # Default delimiters for splitting phrases
    # Includes: comma, semicolon, conjunctions, relative clauses, quotes
    # Note: Quotes use lookbehind/lookahead to avoid splitting contractions
    DEFAULT_DELIMITERS = (
        r",\s*"              # comma
        r"|\s+and\s+"        # and
        r"|\s+but\s+"        # but
        r"|\s+then\s+"       # then
        r"|\s+while\s+"      # while
        r"|\s+or\s+"         # or
        r"|\s+that\s+"       # that
        r"|\s+which\s+"      # which
        r"|\s+where\s+"      # where
        r"|\s+when\s+"       # when
        r"|;\s*"             # semicolon
        r"|(?<!\w)'(?!\w)"   # single quote (not in contractions)
        r'|"\s*'             # double quote
        r"|\s+for\s+the\s+"  # "for the" often separates concepts
        r"|\s+in\s+the\s+"   # "in the"
        r"|\s+of\s+the\s+"   # "of the"
        r"|\s+at\s+the\s+"   # "at the" (new)
        r"|\s+to\s+the\s+"   # "to the" (new)
        r"|\s+on\s+the\s+"   # "on the" (new)
        r"|\s+with\s+the\s+" # "with the" (new)
        r"|\s+with\s+a\s+"   # "with a" (new)
        r"|\s+from\s+the\s+" # "from the" (new)
        r"|\s+about\s+"      # about
        r"|\s+around\s+"     # around
        r"|\s+over\s+the\s+" # "over the" (new)
        r"|\s+under\s+the\s+"# "under the" (new)
        r"|\s+into\s+the\s+" # "into the" (new)
        r"|\s+through\s+"    # through (new)
        r"|\s+during\s+"     # during (new)
        r"|\s+after\s+"      # after (new)
        r"|\s+before\s+"     # before (new)
    )
    
    def __init__(
        self,
        expand_synonyms: bool = False,
        min_chunk_length: int = 3,
        max_chunk_length: int = 60,
        delimiters: str = None,
        custom_synonyms: dict[str, list[str]] = None,
        split_sentences: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            expand_synonyms: Whether to expand chunks with synonyms
            min_chunk_length: Minimum characters for valid chunk
            max_chunk_length: Maximum characters before forcing split
            delimiters: Custom regex pattern for splitting
            custom_synonyms: Additional synonym mappings
            split_sentences: Whether to split on sentence boundaries first
        """
        self.expand_synonyms = expand_synonyms
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.delimiters = delimiters or self.DEFAULT_DELIMITERS
        self.split_sentences = split_sentences
        
        # Merge custom synonyms with defaults
        self.synonyms = self.SYNONYM_MAP.copy()
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove leading/trailing punctuation (keep internal)
        text = text.strip(".,!?;:'\"()-")
        return text
    
    def _expand_with_synonyms(self, text: str) -> str:
        """
        Expand text by appending synonyms of key words.
        
        This helps find better matches in the corpus by including
        alternative phrasings.
        """
        words = text.split()
        expansions = []
        
        for word in words:
            clean_word = word.strip(".,!?;:'\"").lower()
            if clean_word in self.synonyms:
                # Add first synonym only (to avoid too long queries)
                synonym = self.synonyms[clean_word][0]
                if synonym not in text:
                    expansions.append(synonym)
        
        if expansions:
            return f"{text} {' '.join(expansions)}"
        return text
    
    def _split_long_chunk(self, text: str) -> list[str]:
        """
        Split a chunk that exceeds max_chunk_length.
        
        Tries to split at word boundaries intelligently:
        1. First try splitting at prepositions (in, on, at, for, of, etc.)
        2. Then try splitting at articles (the, a, an)
        3. Finally split at nearest word boundary to midpoint
        """
        if len(text) <= self.max_chunk_length:
            return [text]
        
        results = []
        remaining = text
        
        while len(remaining) > self.max_chunk_length:
            # Find a good split point
            split_point = self._find_split_point(remaining)
            
            if split_point <= 0:
                # No good split found, force split at max_chunk_length
                split_point = self.max_chunk_length
                # Back up to last space
                while split_point > 0 and remaining[split_point] != ' ':
                    split_point -= 1
                if split_point <= 0:
                    split_point = self.max_chunk_length
            
            chunk = remaining[:split_point].strip()
            if len(chunk) >= self.min_chunk_length:
                results.append(chunk)
            remaining = remaining[split_point:].strip()
        
        if len(remaining) >= self.min_chunk_length:
            results.append(remaining)
        
        return results
    
    def _find_split_point(self, text: str) -> int:
        """
        Find a good point to split a long text.
        
        Looks for prepositions and articles as natural break points.
        Returns position of the split word, or -1 if none found.
        """
        # Words that make good split points (split BEFORE these)
        split_words = [
            ' in ', ' on ', ' at ', ' for ', ' of ', ' to ', ' from ',
            ' with ', ' about ', ' around ', ' through ', ' between ',
            ' the ', ' a ', ' an ', ' this ', ' that ', ' these ', ' those ',
        ]
        
        # Look for split points in the preferred range (40-80% of max length)
        min_pos = int(self.max_chunk_length * 0.4)
        max_pos = min(len(text), int(self.max_chunk_length * 0.9))
        
        best_pos = -1
        
        for word in split_words:
            pos = text.find(word, min_pos, max_pos)
            if pos > 0:
                # Found a good split point
                if best_pos < 0 or pos < best_pos:
                    best_pos = pos
        
        return best_pos
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(self.SENTENCE_PATTERN, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, message: str) -> list[SemanticChunk]:
        """
        Split message into semantic chunks.
        
        Args:
            message: Input message to chunk
            
        Returns:
            List of SemanticChunk objects
        """
        all_parts = []
        
        # Step 1: Split into sentences first (if enabled)
        if self.split_sentences:
            sentences = self._split_sentences(message)
        else:
            sentences = [message]
        
        # Step 2: Split each sentence by delimiters
        for sentence in sentences:
            parts = re.split(self.delimiters, sentence, flags=re.IGNORECASE)
            all_parts.extend(parts)
        
        # Step 3: Process each part
        chunks = []
        for part in all_parts:
            # Clean the chunk
            original = self._clean_text(part)
            
            # Skip if too short
            if len(original) < self.min_chunk_length:
                continue
            
            # Step 4: Split if too long
            if len(original) > self.max_chunk_length:
                sub_parts = self._split_long_chunk(original)
            else:
                sub_parts = [original]
            
            # Step 5: Create chunks with optional synonym expansion
            for sub_part in sub_parts:
                if len(sub_part) < self.min_chunk_length:
                    continue
                
                if self.expand_synonyms:
                    expanded = self._expand_with_synonyms(sub_part)
                else:
                    expanded = sub_part
                
                chunk = SemanticChunk(
                    text=expanded,
                    original=sub_part,
                    index=len(chunks)
                )
                chunks.append(chunk)
        
        # Fallback: If no chunks were created, treat entire message as one chunk
        if not chunks and message.strip():
            original = self._clean_text(message)
            # Still try to split if too long
            if len(original) > self.max_chunk_length:
                sub_parts = self._split_long_chunk(original)
            else:
                sub_parts = [original]
            
            for sub_part in sub_parts:
                if len(sub_part) >= self.min_chunk_length:
                    if self.expand_synonyms:
                        expanded = self._expand_with_synonyms(sub_part)
                    else:
                        expanded = sub_part
                    chunks.append(SemanticChunk(
                        text=expanded,
                        original=sub_part,
                        index=len(chunks)
                    ))
        
        return chunks
    
    def chunk_simple(self, message: str) -> list[str]:
        """
        Simple chunking that returns just text strings.
        
        Args:
            message: Input message
            
        Returns:
            List of chunk text strings
        """
        return [c.text for c in self.chunk(message)]
    
    def reconstruct(self, chunks: list[SemanticChunk]) -> str:
        """
        Reconstruct message from chunks (using original text).
        
        Args:
            chunks: List of chunks to combine
            
        Returns:
            Reconstructed message
        """
        # Sort by index to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.index)
        return ", ".join(c.original for c in sorted_chunks)
    
    def __repr__(self) -> str:
        return f"SemanticChunker(expand_synonyms={self.expand_synonyms}, max_len={self.max_chunk_length})"


# Convenience function
def chunk_message(message: str, expand: bool = False) -> list[str]:
    """
    Quick chunking of a message.
    
    Args:
        message: Text to chunk
        expand: Whether to expand with synonyms
        
    Returns:
        List of chunk strings
    """
    chunker = SemanticChunker(expand_synonyms=expand)
    return chunker.chunk_simple(message)

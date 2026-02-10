"""
Semantic Chunker

Splits text into meaningful semantic chunks for encoding.
"""

import re
from typing import List, Optional


class SemanticChunker:
    """
    Splits text into semantic chunks for encoding.
    
    The chunker breaks text into meaningful units that can be
    independently encoded as media references.
    
    Strategies:
        - "sentence": Split on sentence boundaries
        - "clause": Split on clauses (commas, 'and', 'or', etc.)
        - "phrase": Split on phrases (shorter units)
    
    Example:
        >>> chunker = SemanticChunker(strategy="clause")
        >>> chunks = chunker.chunk("A dog running through water and a cat sleeping")
        >>> print(chunks)
        ['a dog running through water', 'a cat sleeping']
    """
    
    # Clause delimiters
    CLAUSE_DELIMITERS = r',|\band\b|\bor\b|\bbut\b|\bthen\b|\bwhile\b|\bwhen\b'
    
    def __init__(
        self,
        strategy: str = "clause",
        min_chunk_length: int = 3,
        max_chunk_length: int = 100
    ):
        """
        Initialize the chunker.
        
        Args:
            strategy: Chunking strategy ('sentence', 'clause', 'phrase')
            min_chunk_length: Minimum chunk length in characters
            max_chunk_length: Maximum chunk length in characters
        """
        self.strategy = strategy
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Normalize text
        text = self._normalize(text)
        
        # Apply chunking strategy
        if self.strategy == "sentence":
            chunks = self._chunk_sentences(text)
        elif self.strategy == "clause":
            chunks = self._chunk_clauses(text)
        elif self.strategy == "phrase":
            chunks = self._chunk_phrases(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Filter and clean chunks
        chunks = self._filter_chunks(chunks)
        
        return chunks
    
    def _normalize(self, text: str) -> str:
        """Normalize text for chunking."""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation but keep sentence-ending ones
        text = re.sub(r'["\'\(\)\[\]\{\}]', '', text)
        
        return text.strip()
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_clauses(self, text: str) -> List[str]:
        """Split on clause boundaries (commas, 'and', 'or', etc.)."""
        chunks = re.split(self.CLAUSE_DELIMITERS, text, flags=re.IGNORECASE)
        return [c.strip() for c in chunks if c.strip()]
    
    def _chunk_phrases(self, text: str) -> List[str]:
        """Split on phrase boundaries (smaller units)."""
        # Split on commas, prepositions, etc.
        delimiters = r',|\band\b|\bor\b|\bof\b|\bin\b|\bwith\b|\bto\b|\bfor\b'
        chunks = re.split(delimiters, text, flags=re.IGNORECASE)
        return [c.strip() for c in chunks if c.strip()]
    
    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """Filter chunks based on length constraints."""
        filtered = []
        
        for chunk in chunks:
            # Skip too short
            if len(chunk) < self.min_chunk_length:
                continue
            
            # Truncate too long
            if len(chunk) > self.max_chunk_length:
                chunk = chunk[:self.max_chunk_length]
            
            # Remove leading/trailing punctuation
            chunk = chunk.strip(' ,.-:;')
            
            if chunk:
                filtered.append(chunk)
        
        return filtered
    
    def __repr__(self) -> str:
        return f"SemanticChunker(strategy={self.strategy})"

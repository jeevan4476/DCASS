"""
Semantic Encoder

The main encoder class for DCASS. Encodes secret messages into
sequences of unmodified media references using semantic similarity.

KEY FEATURE: Mixed-Modality Encoding
When modality="auto", each message chunk is matched against ALL
modalities (images AND texts), and the best match is selected
regardless of type. This produces a heterogeneous sequence like:
[image, text, image, image, text]

This makes detection much harder because:
1. No media files are modified (zero-modification steganography)
2. The output is a MIX of different media types
3. There's no predictable pattern in modality selection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import json
import hashlib

from src.corpus.index.unified_index import UnifiedSemanticIndex, SearchResult
from src.engine.chunker import SemanticChunker


# Type alias for valid modalities
Modality = Literal["text", "image", "audio", "auto"]


@dataclass
class EncodedMessage:
    """
    Represents an encoded message as a sequence of media references.
    
    The sequence may contain a MIX of different modalities when
    encoded with modality="auto".
    
    Attributes:
        original_message: The original plaintext message
        chunks: The semantic chunks the message was split into
        sequence: The sequence of SearchResults (media references)
        modality_used: Which modality mode was used ("auto" for mixed)
        metadata: Additional encoding metadata
    """
    original_message: str
    chunks: List[str]
    sequence: List[SearchResult]
    modality_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def media_ids(self) -> List[str]:
        """Get list of media IDs in sequence order."""
        return [r.id for r in self.sequence]
    
    @property
    def media_paths(self) -> List[str]:
        """Get list of media file paths/content in sequence order."""
        return [r.content for r in self.sequence]
    
    @property
    def modality_sequence(self) -> List[str]:
        """Get the sequence of modalities used."""
        return [r.modality for r in self.sequence]
    
    @property
    def modality_distribution(self) -> Dict[str, int]:
        """Get count of each modality in the sequence."""
        dist: Dict[str, int] = {}
        for r in self.sequence:
            dist[r.modality] = dist.get(r.modality, 0) + 1
        return dist
    
    @property
    def is_mixed_modality(self) -> bool:
        """Check if sequence contains multiple modalities."""
        return len(set(self.modality_sequence)) > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_message": self.original_message,
            "chunks": self.chunks,
            "sequence": [
                {
                    "id": r.id,
                    "score": r.score,
                    "modality": r.modality,
                    "content": r.content,
                    "metadata": r.metadata
                }
                for r in self.sequence
            ],
            "modality_used": self.modality_used,
            "modality_distribution": self.modality_distribution,
            "is_mixed": self.is_mixed_modality,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: Path) -> None:
        """Save encoded message to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodedMessage":
        """Create from dictionary."""
        sequence = [
            SearchResult(
                id=r["id"],
                score=r["score"],
                modality=r["modality"],
                content=r["content"],
                metadata=r.get("metadata", {})
            )
            for r in data["sequence"]
        ]
        return cls(
            original_message=data["original_message"],
            chunks=data["chunks"],
            sequence=sequence,
            modality_used=data.get("modality_used", "auto"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def load(cls, path: Path) -> "EncodedMessage":
        """Load encoded message from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        dist = self.modality_distribution
        dist_str = ", ".join(f"{k}={v}" for k, v in dist.items())
        return f"EncodedMessage(chunks={len(self.chunks)}, [{dist_str}])"


class SemanticEncoder:
    """
    Encodes messages into sequences of media references.
    
    MIXED-MODALITY ENCODING (default):
    When modality="auto", each chunk searches ALL indices and picks
    the best match regardless of type. This produces heterogeneous
    sequences that are harder to detect.
    
    Example:
        >>> encoder = SemanticEncoder()
        >>> encoder.load()
        >>> encoded = encoder.encode("The secret meeting is at dawn in the park")
        >>> print(encoded.modality_sequence)
        ['image', 'text', 'image']  # Mixed!
        >>> print(encoded.media_paths)
        ['whisper.jpg', 'The sun rises over...', 'park_bench.jpg']
    
    Attributes:
        index: The UnifiedSemanticIndex for searching
        chunker: The SemanticChunker for splitting messages
        default_modality: Default modality ("auto" for mixed)
    """
    
    def __init__(
        self,
        index: Optional[UnifiedSemanticIndex] = None,
        chunker: Optional[SemanticChunker] = None,
        default_modality: Modality = "auto",
        chunk_strategy: str = "clause",
        normalize_scores: bool = True,
        diversity_ratio: float = 0.0,
    ):
        """
        Initialize the encoder.
        
        Args:
            index: UnifiedSemanticIndex instance. If None, creates new one.
            chunker: SemanticChunker instance. If None, creates new one.
            default_modality: Default modality ("auto" for mixed-modality)
            chunk_strategy: Chunking strategy ('sentence', 'clause', 'phrase')
            normalize_scores: Whether to normalize scores across modalities
            diversity_ratio: Force minimum ratio of each modality (0.0-1.0)
        """
        self.index = index or UnifiedSemanticIndex(
            normalize_scores=normalize_scores,
            diversity_ratio=diversity_ratio,
        )
        self.chunker = chunker or SemanticChunker(strategy=chunk_strategy)
        self.default_modality: Modality = default_modality
        self._loaded = False
    
    def load(self, modalities: Optional[List[str]] = None) -> None:
        """
        Load indices from disk.
        
        Args:
            modalities: List of modalities to load. If None, loads all.
        """
        self.index.load(modalities)
        self._loaded = True
    
    def encode(
        self,
        message: str,
        modality: Optional[Modality] = None,
        k_candidates: int = 1,
        diversity_penalty: float = 0.0
    ) -> EncodedMessage:
        """
        Encode a message into a sequence of media references.
        
        When modality="auto" (default):
        - Each chunk searches ALL loaded indices
        - Best match is selected regardless of modality
        - Result is a MIX of images and texts
        
        Args:
            message: The secret message to encode
            modality: Which modality to use ("auto" for mixed)
            k_candidates: Number of candidates to consider per chunk
            diversity_penalty: Penalty for selecting same media twice (0.0-1.0)
            
        Returns:
            EncodedMessage containing the encoded sequence
            
        Raises:
            RuntimeError: If index not loaded
        """
        if not self._loaded:
            raise RuntimeError("Index not loaded. Call load() first.")
        
        modality = modality or self.default_modality
        
        # Step 1: Chunk the message
        chunks = self.chunker.chunk(message)
        
        if not chunks:
            # If chunking produces nothing, treat whole message as one chunk
            chunks = [message.strip().lower()]
        
        # Step 2: Encode each chunk
        sequence: List[SearchResult] = []
        used_ids: set = set()
        
        for chunk in chunks:
            # Get candidates - when modality="auto", this searches ALL indices
            # Cast modality to the Literal type for the search function
            search_modality: Literal["text", "image", "audio", "auto"] = modality  # type: ignore
            candidates = self.index.search(
                chunk,
                modality=search_modality,
                k=k_candidates + len(used_ids)  # Get extra to handle diversity
            )
            
            # Apply diversity penalty if needed
            if diversity_penalty > 0 and used_ids:
                candidates = self._apply_diversity(candidates, used_ids, diversity_penalty)
            
            # Select best candidate
            if candidates:
                selected = candidates[0]
                sequence.append(selected)
                used_ids.add(selected.id)
        
        # Step 3: Build metadata
        metadata = {
            "chunk_strategy": self.chunker.strategy,
            "k_candidates": k_candidates,
            "diversity_penalty": diversity_penalty,
            "message_hash": hashlib.sha256(message.encode()).hexdigest()[:16]
        }
        
        return EncodedMessage(
            original_message=message,
            chunks=chunks,
            sequence=sequence,
            modality_used=modality,
            metadata=metadata
        )
    
    def _apply_diversity(
        self,
        candidates: List[SearchResult],
        used_ids: set,
        penalty: float
    ) -> List[SearchResult]:
        """
        Apply diversity penalty to already-used media.
        
        Args:
            candidates: List of candidates
            used_ids: Set of already-used media IDs
            penalty: Penalty factor (0.0-1.0)
            
        Returns:
            Re-ranked candidates list
        """
        adjusted = []
        for c in candidates:
            if c.id in used_ids:
                # Reduce score for already-used items
                adjusted_score = c.score * (1 - penalty)
                adjusted.append(SearchResult(
                    id=c.id,
                    score=adjusted_score,
                    modality=c.modality,
                    content=c.content,
                    metadata=c.metadata
                ))
            else:
                adjusted.append(c)
        
        # Re-sort by adjusted score
        adjusted.sort(key=lambda x: x.score, reverse=True)
        return adjusted
    
    def encode_batch(
        self,
        messages: List[str],
        modality: Optional[Modality] = None,
        **kwargs
    ) -> List[EncodedMessage]:
        """
        Encode multiple messages.
        
        Args:
            messages: List of messages to encode
            modality: Which modality to use
            **kwargs: Additional arguments passed to encode()
            
        Returns:
            List of EncodedMessage objects
        """
        return [self.encode(msg, modality=modality, **kwargs) for msg in messages]
    
    def get_statistics(self, encoded: EncodedMessage) -> Dict[str, Any]:
        """
        Get encoding statistics.
        
        Args:
            encoded: An EncodedMessage object
            
        Returns:
            Dictionary with encoding statistics
        """
        scores = [r.score for r in encoded.sequence]
        
        return {
            "message_length": len(encoded.original_message),
            "num_chunks": len(encoded.chunks),
            "num_media": len(encoded.sequence),
            "avg_chunk_length": sum(len(c) for c in encoded.chunks) / len(encoded.chunks) if encoded.chunks else 0,
            "avg_similarity": sum(scores) / len(scores) if scores else 0,
            "min_similarity": min(scores) if scores else 0,
            "max_similarity": max(scores) if scores else 0,
            "unique_media": len(set(r.id for r in encoded.sequence)),
            "modality_used": encoded.modality_used,
            "modality_distribution": encoded.modality_distribution,
            "is_mixed_modality": encoded.is_mixed_modality
        }
    
    def __repr__(self) -> str:
        loaded_str = "loaded" if self._loaded else "not loaded"
        return f"SemanticEncoder(modality={self.default_modality}, {loaded_str})"


# Convenience function for quick encoding
def encode_message(
    message: str,
    modality: Modality = "auto",
    chunk_strategy: str = "clause"
) -> EncodedMessage:
    """
    Convenience function to encode a message.
    
    Creates an encoder, loads the index, and encodes the message.
    Uses mixed-modality by default (modality="auto").
    
    Args:
        message: The message to encode
        modality: Which modality to use ("auto" for mixed)
        chunk_strategy: Chunking strategy
        
    Returns:
        EncodedMessage object
    """
    encoder = SemanticEncoder(
        default_modality=modality,
        chunk_strategy=chunk_strategy
    )
    encoder.load()  # Load all modalities
    return encoder.encode(message)

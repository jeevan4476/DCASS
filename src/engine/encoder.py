# src/engine/encoder.py
"""
SemanticEncoder - Core encoding engine for DCASS.

Encodes secret messages into sequences of unmodified media items by:
1. Chunking the message into semantic units
2. Searching the unified corpus for matching media
3. Selecting the best media item for each chunk
4. Returning the encoded sequence

Architecture:
    Message -> Chunker -> For each chunk: Search Index -> Select Best -> Media Sequence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import re

from src.corpus.index.unified_index import UnifiedSemanticIndex, MediaItem, Modality
from src.engine.chunker import SemanticChunker, SemanticChunk

# Diversity mode type
DiversityMode = Literal["best", "round_robin", "balanced"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by", "for",
    "from", "if", "in", "inside", "into", "is", "it", "let", "me", "near",
    "of", "on", "or", "outside", "the", "their", "them", "then", "there",
    "they", "to", "up", "us", "we", "when", "with", "you", "your",
}


def _keywords(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z']+", text.lower()) if token not in STOPWORDS]


def _candidate_text_score(chunk_text: str, media: MediaItem) -> float:
    """
    Add a lightweight lexical bias toward candidates whose decoded text
    preserves chunk keywords rather than only being embedding-near.
    """
    chunk_keywords = _keywords(chunk_text)
    if not chunk_keywords:
        return media.normalized_score

    content = (media.content or "").lower()
    overlap = sum(1 for token in chunk_keywords if token in content)
    overlap_ratio = overlap / len(chunk_keywords)

    exact_phrase_bonus = 0.08 if chunk_text.lower() in content else 0.0
    modality_bonus = 0.04 if media.modality == "text" else 0.0
    semantic_bonus = overlap_ratio * 0.35

    return media.normalized_score + semantic_bonus + exact_phrase_bonus + modality_bonus


@dataclass
class EncodedChunk:
    """Represents an encoded semantic chunk."""
    chunk: SemanticChunk         # Original semantic chunk
    media: MediaItem             # Selected media item
    alternatives: list[MediaItem] = field(default_factory=list)  # Other options
    
    def __repr__(self) -> str:
        return f"EncodedChunk('{self.chunk.original}' -> {self.media.modality}:{self.media.id})"


@dataclass  
class EncodingResult:
    """Complete result of encoding a message."""
    original_message: str
    chunks: list[SemanticChunk]
    encoded: list[EncodedChunk]
    
    @property
    def media_sequence(self) -> list[MediaItem]:
        """Get the sequence of selected media items."""
        return [e.media for e in self.encoded]
    
    @property
    def media_ids(self) -> list[str]:
        """Get just the media IDs for transmission."""
        return [e.media.id for e in self.encoded]
    
    @property
    def modality_breakdown(self) -> dict[str, int]:
        """Count of media items by modality."""
        counts = {}
        for e in self.encoded:
            m = e.media.modality
            counts[m] = counts.get(m, 0) + 1
        return counts
    
    def summary(self) -> str:
        """Human-readable summary of encoding."""
        lines = [
            f"Message: \"{self.original_message}\"",
            f"Chunks: {len(self.chunks)}",
            f"Modalities: {self.modality_breakdown}",
            "",
            "Encoding:",
        ]
        for i, enc in enumerate(self.encoded, 1):
            lines.append(
                f"  {i}. \"{enc.chunk.original}\" -> "
                f"{enc.media.modality}:{enc.media.id} "
                f"(score: {enc.media.normalized_score:.3f})"
            )
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"EncodingResult(chunks={len(self.chunks)}, media={self.media_ids})"


class SemanticEncoder:
    """
    Core semantic encoder for DCASS steganography.
    
    Transforms secret messages into sequences of existing media items
    (images, text, audio) by semantic similarity matching.
    
    Key Features:
    - Multi-modal encoding (can select from images, text, or audio)
    - Synonym expansion for better corpus matching
    - Score normalization across modalities
    - Alternative selection for diversity
    
    Usage:
        # Initialize with index
        encoder = SemanticEncoder()
        encoder.load()  # Loads indices
        
        # Encode a message
        result = encoder.encode("Meet me at the cafe at noon")
        
        # Get media sequence
        for media in result.media_sequence:
            print(f"{media.modality}: {media.id}")
        
        # Just get IDs for transmission
        ids = result.media_ids  # ['flickr8k_00123', 'wiki_00456', ...]
    
    Attributes:
        index: UnifiedSemanticIndex for corpus search
        chunker: SemanticChunker for message splitting
        default_modalities: Modalities to search by default
    """
    
    def __init__(
        self,
        index: UnifiedSemanticIndex = None,
        chunker: SemanticChunker = None,
        expand_synonyms: bool = True,
        default_modalities: list[Modality] = None,
        base_path: Path = None,
        device: str = None
    ):
        """
        Initialize the encoder.
        
        Args:
            index: Pre-configured UnifiedSemanticIndex (created if None)
            chunker: Pre-configured SemanticChunker (created if None)
            expand_synonyms: Whether to expand chunks with synonyms
            default_modalities: Default modalities to search
            base_path: Base path for indices (passed to index if created)
            device: Device for models (passed to index if created)
        """
        self._index = index
        self._chunker = chunker or SemanticChunker(expand_synonyms=expand_synonyms)
        self.default_modalities = default_modalities or ["image", "text", "audio"]
        self._base_path = base_path
        self._device = device
        self._loaded = False
    
    @property
    def index(self) -> UnifiedSemanticIndex:
        """Get the unified index (creates if needed)."""
        if self._index is None:
            self._index = UnifiedSemanticIndex(
                base_path=self._base_path,
                device=self._device,
                enabled_modalities=self.default_modalities
            )
        return self._index
    
    @property
    def chunker(self) -> SemanticChunker:
        """Get the chunker."""
        return self._chunker
    
    def load(self, modalities: list[Modality] = None) -> dict[str, bool]:
        """
        Load the underlying indices.
        
        Args:
            modalities: Specific modalities to load
            
        Returns:
            Dict mapping modality to load success status
        """
        print("Loading semantic encoder...")
        status = self.index.load(modalities)
        self._loaded = any(status.values())
        return status
    
    def is_loaded(self) -> bool:
        """Check if encoder is ready for use."""
        return self._loaded
    
    def encode(
        self,
        message: str,
        modalities: list[Modality] = None,
        k_per_chunk: int = 1,
        keep_alternatives: int = 3,
        min_score: float = 0.0,
        avoid_duplicates: bool = True,
        diversity_mode: DiversityMode = "best"
    ) -> EncodingResult:
        """
        Encode a message into a media sequence.
        
        Args:
            message: Secret message to encode
            modalities: Modalities to search (uses default if None)
            k_per_chunk: Number of media items per chunk (usually 1)
            keep_alternatives: How many alternatives to store
            min_score: Minimum normalized score threshold
            avoid_duplicates: If True, each chunk gets a unique media item
            diversity_mode: How to select modalities for each chunk:
                - "best": Select the highest-scoring item regardless of modality (default)
                - "round_robin": Cycle through modalities (image -> text -> audio -> ...)
                - "balanced": Prefer underrepresented modalities to balance output
            
        Returns:
            EncodingResult with encoded chunks and media sequence
        """
        if not self._loaded:
            raise RuntimeError("Encoder not loaded. Call load() first.")
        
        modalities = modalities or self.default_modalities
        
        # Step 1: Chunk the message
        chunks = self.chunker.chunk(message)
        
        if not chunks:
            raise ValueError("Message produced no valid chunks")
        
        # Step 2: Encode each chunk
        encoded_chunks = []
        used_ids: set[str] = set()  # Track used media IDs to avoid duplicates
        modality_counts: dict[str, int] = {m: 0 for m in modalities}  # For balanced mode
        
        for chunk_idx, chunk in enumerate(chunks):
            # Determine which modality to search based on diversity mode
            if diversity_mode == "round_robin":
                # Cycle through modalities
                target_modality = modalities[chunk_idx % len(modalities)]
                search_modalities = [target_modality]
            elif diversity_mode == "balanced":
                # Prefer underrepresented modalities
                # Sort modalities by count (ascending) and use the least common
                sorted_modalities = sorted(modalities, key=lambda m: modality_counts.get(m, 0))
                search_modalities = sorted_modalities  # Search all but prefer least common
            else:
                # "best" mode - search all modalities
                search_modalities = modalities
            
            # Search for matching media (request more to account for filtering)
            search_k = (k_per_chunk + keep_alternatives) * 3 if avoid_duplicates else k_per_chunk + keep_alternatives
            
            results = self.index.search(
                query=chunk.text,
                k=search_k,
                modalities=search_modalities,
                min_score=min_score
            )
            
            if not results:
                # Fallback: try with original text (without synonym expansion)
                results = self.index.search(
                    query=chunk.original,
                    k=search_k,
                    modalities=search_modalities
                )
            
            # For round_robin: if no results in target modality, fall back to all modalities
            if not results and diversity_mode == "round_robin":
                results = self.index.search(
                    query=chunk.text,
                    k=search_k,
                    modalities=modalities,
                    min_score=min_score
                )
            
            if not results:
                raise RuntimeError(f"No media found for chunk: '{chunk.original}'")
            
            # Filter out already-used IDs if avoiding duplicates
            if avoid_duplicates:
                results = [r for r in results if r.id not in used_ids]
            
            if not results:
                raise RuntimeError(f"No unique media found for chunk: '{chunk.original}' (all candidates already used)")
            
            # For balanced mode: re-sort results to prefer underrepresented modalities
            if diversity_mode == "balanced":
                # Sort by: (modality_count, -normalized_score) to prefer rare modalities with good scores
                results = sorted(
                    results,
                    key=lambda r: (modality_counts.get(r.modality, 0), -r.normalized_score)
                )
            else:
                # Prefer candidates whose decoded text preserves chunk keywords.
                results = sorted(
                    results,
                    key=lambda r: _candidate_text_score(chunk.original, r),
                    reverse=True,
                )
            
            # Select best match(es)
            selected = results[:k_per_chunk]
            alternatives = results[k_per_chunk:k_per_chunk + keep_alternatives]
            
            for media in selected:
                encoded_chunks.append(EncodedChunk(
                    chunk=chunk,
                    media=media,
                    alternatives=alternatives
                ))
                # Mark this ID as used
                used_ids.add(media.id)
                # Update modality count for balanced mode
                modality_counts[media.modality] = modality_counts.get(media.modality, 0) + 1
        
        return EncodingResult(
            original_message=message,
            chunks=chunks,
            encoded=encoded_chunks
        )
    
    def encode_to_ids(
        self,
        message: str,
        modalities: list[Modality] = None
    ) -> list[str]:
        """
        Encode message and return just the media IDs.
        
        Convenience method for when you just need the ID sequence.
        
        Args:
            message: Secret message
            modalities: Modalities to search
            
        Returns:
            List of media IDs
        """
        result = self.encode(message, modalities=modalities)
        return result.media_ids
    
    def encode_images_only(self, message: str) -> list[str]:
        """
        Encode using only images.
        
        Args:
            message: Secret message
            
        Returns:
            List of image IDs
        """
        return self.encode_to_ids(message, modalities=["image"])
    
    def encode_text_only(self, message: str) -> list[str]:
        """
        Encode using only text.
        
        Args:
            message: Secret message
            
        Returns:
            List of text IDs
        """
        return self.encode_to_ids(message, modalities=["text"])
    
    def encode_audio_only(self, message: str) -> list[str]:
        """
        Encode using only audio.
        
        Args:
            message: Secret message
            
        Returns:
            List of audio IDs
        """
        return self.encode_to_ids(message, modalities=["audio"])
    
    def status(self) -> dict:
        """Get encoder status."""
        return {
            "loaded": self._loaded,
            "chunker": repr(self.chunker),
            "default_modalities": self.default_modalities,
            "index": self.index.status() if self._loaded else "not loaded"
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"SemanticEncoder(modalities={self.default_modalities}, {status})"


# Convenience function for quick encoding
def encode_message(
    message: str,
    modalities: list[Modality] = None,
    base_path: Path = None
) -> EncodingResult:
    """
    Quick encode a message.
    
    Creates encoder, loads indices, and encodes in one call.
    
    Args:
        message: Message to encode
        modalities: Modalities to use
        base_path: Custom index path
        
    Returns:
        EncodingResult
    """
    encoder = SemanticEncoder(
        default_modalities=modalities or ["image", "text", "audio"],
        base_path=base_path
    )
    encoder.load()
    return encoder.encode(message)

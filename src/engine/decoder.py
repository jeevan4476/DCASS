"""
Semantic Decoder

Decodes messages from sequences of media references.

In DCASS steganography:
- The sender (Alice) encodes a message into a sequence of media
- The receiver (Bob) receives the media sequence
- Bob uses this decoder to recover the original meaning

The decoder works by:
1. Taking the sequence of media IDs/paths
2. Looking up each media's semantic embedding
3. Finding the closest semantic match for each chunk
4. Reconstructing the message from the matched meanings

NOTE: Full message recovery requires a shared codebook or semantic mapping.
This implementation provides the infrastructure for the decoding process.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json

from src.corpus.index.unified_index import UnifiedSemanticIndex, SearchResult


@dataclass
class DecodedMessage:
    """
    Represents a decoded message from a media sequence.
    
    Attributes:
        media_sequence: The input media IDs/paths
        semantic_chunks: The semantic meaning of each media item
        reconstructed_text: Attempted text reconstruction (if possible)
        confidence_scores: Confidence score for each decoded chunk
        metadata: Additional decoding metadata
    """
    media_sequence: List[str]
    semantic_chunks: List[str]
    reconstructed_text: str
    confidence_scores: List[float]
    metadata: Dict[str, Any]
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence score across all chunks."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "media_sequence": self.media_sequence,
            "semantic_chunks": self.semantic_chunks,
            "reconstructed_text": self.reconstructed_text,
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def __repr__(self) -> str:
        return f"DecodedMessage(chunks={len(self.semantic_chunks)}, confidence={self.avg_confidence:.2f})"


class SemanticDecoder:
    """
    Decodes messages from sequences of media references.
    
    The decoder reverses the encoding process by looking up
    the semantic meaning of each media item in the sequence.
    
    Usage:
        >>> decoder = SemanticDecoder()
        >>> decoded = decoder.decode(["img_001.jpg", "img_042.jpg", "img_103.jpg"])
        >>> print(decoded.reconstructed_text)
        "sunrise park meeting"
    
    Attributes:
        index: The UnifiedSemanticIndex for lookups
        default_modality: Default modality for decoding
    """
    
    def __init__(
        self,
        index: Optional[UnifiedSemanticIndex] = None,
        default_modality: str = "image"
    ):
        """
        Initialize the decoder.
        
        Args:
            index: UnifiedSemanticIndex instance. If None, creates new one.
            default_modality: Default modality for decoding
        """
        self.index = index or UnifiedSemanticIndex()
        self.default_modality = default_modality
        self._loaded = False
        self._metadata_lookup: Dict[str, Dict[str, Any]] = {}
    
    def load(self, modalities: Optional[List[str]] = None) -> None:
        """
        Load indices and build metadata lookup.
        
        Args:
            modalities: List of modalities to load. If None, loads all.
        """
        self.index.load(modalities)
        self._loaded = True
        
        # Build reverse lookup from metadata
        self._build_metadata_lookup()
    
    def _build_metadata_lookup(self) -> None:
        """Build lookup table from media ID to metadata."""
        self._metadata_lookup.clear()
        
        for modality in self.index.loaded_modalities:
            idx = self.index.get_index(modality)
            for meta in idx.metadata:
                media_id = meta.get("id", "")
                if media_id:
                    self._metadata_lookup[media_id] = {
                        **meta,
                        "_modality": modality
                    }
    
    def decode(
        self,
        media_sequence: List[str],
        modality: Optional[str] = None
    ) -> DecodedMessage:
        """
        Decode a sequence of media references into semantic meaning.
        
        Args:
            media_sequence: List of media IDs or file paths
            modality: Which modality the media belongs to
            
        Returns:
            DecodedMessage containing the decoded information
            
        Raises:
            RuntimeError: If index not loaded
        """
        if not self._loaded:
            raise RuntimeError("Index not loaded. Call load() first.")
        
        modality = modality or self.default_modality
        
        semantic_chunks: List[str] = []
        confidence_scores: List[float] = []
        
        for media_ref in media_sequence:
            # Try to find this media in our index
            chunk, confidence = self._decode_single(media_ref, modality)
            semantic_chunks.append(chunk)
            confidence_scores.append(confidence)
        
        # Reconstruct text by joining semantic chunks
        reconstructed_text = " ".join(semantic_chunks)
        
        metadata = {
            "modality": modality,
            "num_items": len(media_sequence),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        }
        
        return DecodedMessage(
            media_sequence=media_sequence,
            semantic_chunks=semantic_chunks,
            reconstructed_text=reconstructed_text,
            confidence_scores=confidence_scores,
            metadata=metadata
        )
    
    def _decode_single(
        self,
        media_ref: str,
        modality: str
    ) -> tuple:
        """
        Decode a single media reference.
        
        Args:
            media_ref: Media ID or file path
            modality: Which modality to use
            
        Returns:
            Tuple of (semantic_chunk, confidence_score)
        """
        # First, try direct lookup by ID
        if media_ref in self._metadata_lookup:
            meta = self._metadata_lookup[media_ref]
            
            # The semantic content is in the metadata
            # For images, this might be a caption or description
            # For text, this is the text itself
            content = meta.get("caption", meta.get("content", meta.get("text", "")))
            
            if content:
                return (content, 1.0)
        
        # Try lookup by filename (strip path if present)
        filename = Path(media_ref).name
        for media_id, meta in self._metadata_lookup.items():
            if Path(media_id).name == filename:
                content = meta.get("caption", meta.get("content", meta.get("text", "")))
                if content:
                    return (content, 0.9)  # Slightly lower confidence for filename match
        
        # If we can't find it, return placeholder
        return (f"[unknown:{media_ref}]", 0.0)
    
    def decode_from_encoded(
        self,
        encoded_path: Union[str, Path]
    ) -> DecodedMessage:
        """
        Decode from an EncodedMessage JSON file.
        
        This is useful for verification - decoding what was encoded.
        
        Args:
            encoded_path: Path to the encoded message JSON
            
        Returns:
            DecodedMessage
        """
        from src.engine.encoder import EncodedMessage
        
        encoded = EncodedMessage.load(Path(encoded_path))
        media_ids = encoded.media_ids
        
        return self.decode(media_ids, modality=encoded.modality_used)
    
    def verify_encoding(
        self,
        encoded_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Verify an encoding by comparing decoded to original.
        
        Args:
            encoded_path: Path to the encoded message JSON
            
        Returns:
            Dictionary with verification results
        """
        from src.engine.encoder import EncodedMessage
        
        encoded = EncodedMessage.load(Path(encoded_path))
        decoded = self.decode(encoded.media_ids, modality=encoded.modality_used)
        
        # Compare chunks
        chunk_matches = []
        for orig_chunk, decoded_chunk in zip(encoded.chunks, decoded.semantic_chunks):
            # Simple similarity check (could be enhanced with embeddings)
            match_score = self._simple_similarity(orig_chunk, decoded_chunk)
            chunk_matches.append(match_score)
        
        return {
            "original_message": encoded.original_message,
            "reconstructed_text": decoded.reconstructed_text,
            "num_chunks": len(encoded.chunks),
            "chunk_match_scores": chunk_matches,
            "avg_match_score": sum(chunk_matches) / len(chunk_matches) if chunk_matches else 0,
            "avg_confidence": decoded.avg_confidence
        }
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Simple word overlap similarity.
        
        This is a basic metric - in production, use embedding similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def __repr__(self) -> str:
        loaded_str = "loaded" if self._loaded else "not loaded"
        return f"SemanticDecoder(modality={self.default_modality}, {loaded_str})"


# Convenience function for quick decoding
def decode_sequence(
    media_sequence: List[str],
    modality: str = "image"
) -> DecodedMessage:
    """
    Convenience function to decode a media sequence.
    
    Creates a decoder, loads the index, and decodes the sequence.
    
    Args:
        media_sequence: List of media IDs/paths
        modality: Which modality to use
        
    Returns:
        DecodedMessage object
    """
    decoder = SemanticDecoder(default_modality=modality)
    decoder.load([modality])
    return decoder.decode(media_sequence)

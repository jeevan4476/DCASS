# src/engine/decoder.py
"""
SemanticDecoder - Core decoding engine for DCASS.

Decodes media sequences back into semantic meaning by:
1. Looking up each media item in the corpus
2. Extracting semantic content (caption/text)
3. Verifying items exist in the corpus (tamper detection)
4. Reconstructing the original semantic meaning

Architecture:
    Media Sequence -> Lookup Each ID -> Extract Content -> Verify -> Semantic Meaning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from src.corpus.index.unified_index import (
    UnifiedSemanticIndex,
    MediaItem,
    Modality,
    extract_semantic_content,
)


@dataclass
class DecodedItem:
    """Represents a decoded media item."""
    media_id: str
    modality: Modality
    content: str  # The semantic content (text/caption)
    verified: bool  # Whether item was found in corpus
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "VERIFIED" if self.verified else "UNVERIFIED"
        return f"DecodedItem({self.modality}:{self.media_id}, {status})"


@dataclass
class DecodingResult:
    """Complete result of decoding a media sequence."""
    media_ids: list[str]
    decoded: list[DecodedItem]
    
    @property
    def contents(self) -> list[str]:
        """Get the semantic content from each decoded item."""
        return [d.content for d in self.decoded if d.content]
    
    @property
    def reconstructed_meaning(self) -> str:
        """Reconstruct the semantic meaning from all items."""
        return " | ".join(self.contents)
    
    @property
    def verification_rate(self) -> float:
        """Percentage of items that were verified in corpus."""
        if not self.decoded:
            return 0.0
        verified = sum(1 for d in self.decoded if d.verified)
        return verified / len(self.decoded)
    
    @property
    def all_verified(self) -> bool:
        """Check if all items were verified."""
        return all(d.verified for d in self.decoded)
    
    def summary(self) -> str:
        """Human-readable summary of decoding."""
        lines = [
            f"Media IDs: {self.media_ids}",
            f"Decoded items: {len(self.decoded)}",
            f"Verification rate: {self.verification_rate:.1%}",
            "",
            "Decoded content:",
        ]
        for i, item in enumerate(self.decoded, 1):
            status = "✓" if item.verified else "✗"
            lines.append(f"  {i}. [{status}] {item.modality}:{item.media_id}")
            lines.append(f"      \"{item.content[:60]}...\"" if len(item.content) > 60 else f"      \"{item.content}\"")
        
        lines.append("")
        lines.append(f"Reconstructed: \"{self.reconstructed_meaning}\"")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"DecodingResult(items={len(self.decoded)}, verified={self.verification_rate:.0%})"


class SemanticDecoder:
    """
    Core semantic decoder for DCASS steganography.
    
    Transforms sequences of media IDs back into semantic meaning
    by looking up items in the unified corpus index.
    
    Key Features:
    - Multi-modal decoding (images, text, audio)
    - Corpus verification (tamper detection)
    - Semantic content extraction
    
    Usage:
        # Initialize with index
        decoder = SemanticDecoder()
        decoder.load()  # Loads indices
        
        # Decode a media sequence
        result = decoder.decode(["flickr8k_00123", "wiki_00456"])
        
        # Get reconstructed meaning
        print(result.reconstructed_meaning)
        
        # Check verification
        if result.all_verified:
            print("All items verified in corpus")
    
    Attributes:
        index: UnifiedSemanticIndex for corpus lookup
    """
    
    def __init__(
        self,
        index: UnifiedSemanticIndex = None,
        base_path: Path = None,
        device: str = None
    ):
        """
        Initialize the decoder.
        
        Args:
            index: Pre-configured UnifiedSemanticIndex (created if None)
            base_path: Base path for indices (passed to index if created)
            device: Device for models (passed to index if created)
        """
        self._index = index
        self._base_path = base_path
        self._device = device
        self._loaded = False
    
    @property
    def index(self) -> UnifiedSemanticIndex:
        """Get the unified index (creates if needed)."""
        if self._index is None:
            self._index = UnifiedSemanticIndex(
                base_path=self._base_path,
                device=self._device
            )
        return self._index
    
    def load(self, modalities: list[Modality] = None) -> dict[str, bool]:
        """
        Load the underlying indices.
        
        Args:
            modalities: Specific modalities to load
            
        Returns:
            Dict mapping modality to load success status
        """
        print("Loading semantic decoder...")
        status = self.index.load(modalities)
        self._loaded = any(status.values())
        return status
    
    def is_loaded(self) -> bool:
        """Check if decoder is ready for use."""
        return self._loaded
    
    def decode(self, media_ids: list[str]) -> DecodingResult:
        """
        Decode a sequence of media IDs into semantic meaning.
        
        Args:
            media_ids: List of media IDs to decode
            
        Returns:
            DecodingResult with decoded items and reconstructed meaning
        """
        if not self._loaded:
            raise RuntimeError("Decoder not loaded. Call load() first.")
        
        decoded_items = []
        
        for media_id in media_ids:
            # Look up item in corpus
            item = self.index.get_by_id(media_id)
            
            if item:
                # Item found - extract content
                content = extract_semantic_content(item.metadata, item.modality)
                
                decoded_items.append(DecodedItem(
                    media_id=media_id,
                    modality=item.modality,
                    content=content,
                    verified=True,
                    metadata=item.metadata
                ))
            else:
                # Item not found - unverified
                decoded_items.append(DecodedItem(
                    media_id=media_id,
                    modality="text",  # Default
                    content=f"[UNVERIFIED: {media_id}]",
                    verified=False,
                    metadata={}
                ))
        
        return DecodingResult(
            media_ids=media_ids,
            decoded=decoded_items
        )
    
    def decode_to_text(self, media_ids: list[str]) -> str:
        """
        Decode media IDs and return just the reconstructed text.
        
        Convenience method for simple decoding.
        
        Args:
            media_ids: List of media IDs
            
        Returns:
            Reconstructed semantic meaning as string
        """
        result = self.decode(media_ids)
        return result.reconstructed_meaning
    
    def verify_sequence(self, media_ids: list[str]) -> tuple[bool, float]:
        """
        Verify that all media IDs exist in the corpus.
        
        Args:
            media_ids: List of media IDs to verify
            
        Returns:
            Tuple of (all_verified, verification_rate)
        """
        result = self.decode(media_ids)
        return result.all_verified, result.verification_rate
    
    def status(self) -> dict:
        """Get decoder status."""
        return {
            "loaded": self._loaded,
            "index": self.index.status() if self._loaded else "not loaded"
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"SemanticDecoder({status})"


# Convenience function for quick decoding
def decode_media_sequence(
    media_ids: list[str],
    base_path: Path = None
) -> DecodingResult:
    """
    Quick decode a media sequence.
    
    Creates decoder, loads indices, and decodes in one call.
    
    Args:
        media_ids: Media IDs to decode
        base_path: Custom index path
        
    Returns:
        DecodingResult
    """
    decoder = SemanticDecoder(base_path=base_path)
    decoder.load()
    return decoder.decode(media_ids)

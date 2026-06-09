# src/corpus/index/unified_index.py
"""
UnifiedSemanticIndex - Multi-modal FAISS index with score normalization.

This module provides a unified interface for searching across image, text, 
and audio modalities using CLIP/CLAP embeddings.

Architecture:
    Query → CLIP encode → Search each index → Normalize scores → Merge & rank → Return top-k
"""

from __future__ import annotations

import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import clip

Modality = Literal["image", "text", "audio"]


def resolve_indices_base_path(base_path: Path | None = None) -> Path:
    """
    Resolve the indices directory.

    Prefers an explicit path when provided. Otherwise, tries the current
    `storage/data/indices` location first, then falls back to the older
    `storage/indices` layout when that is what exists locally.
    """
    if base_path is not None:
        return Path(base_path)

    project_root = Path(__file__).parent.parent.parent.parent
    candidates = [
        project_root / "storage" / "data" / "indices",
        project_root / "storage" / "indices",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _is_path_like(value: str) -> bool:
    value = value.strip().lower()
    return (
        "/" in value
        or "\\" in value
        or value.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".mp3", ".wav", ".flac"))
    )


def extract_semantic_content(meta: dict, modality: Modality) -> str:
    """
    Extract the most meaningful human-readable content for a media item.

    This prefers captions/transcripts/text over filesystem paths so decoded
    output and encoder reranking operate on semantic text instead of raw paths.
    """
    if modality == "image":
        captions = meta.get("captions")
        if isinstance(captions, list):
            for caption in captions:
                if isinstance(caption, str) and caption.strip():
                    return caption.strip()

        for key in ["caption", "text", "content"]:
            value = meta.get(key)
            if isinstance(value, str) and value.strip() and not _is_path_like(value):
                return value.strip()

        path = meta.get("path", "")
        return path.strip() if isinstance(path, str) else ""

    if modality == "audio":
        for key in ["text", "transcript", "content"]:
            value = meta.get(key)
            if isinstance(value, str) and value.strip() and not _is_path_like(value):
                return value.strip()

    if modality == "text":
        for key in ["text", "content"]:
            value = meta.get(key)
            if isinstance(value, str) and value.strip() and not _is_path_like(value):
                return value.strip()

    for key in ["content", "text", "path"]:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


@dataclass
class MediaItem:
    """Represents a media item from the corpus."""
    id: str
    modality: Modality
    content: str  # path for image/audio, text content for text
    score: float  # raw similarity score
    normalized_score: float  # normalized score (0-1)
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"MediaItem({self.modality}:{self.id}, score={self.normalized_score:.3f})"


class ScoreNormalizer:
    """
    Normalizes similarity scores across different modalities.
    
    Problem: CLIP similarity scores vary significantly by modality:
        - text-to-text: ~0.6-0.9
        - text-to-image: ~0.2-0.4
        - text-to-audio: ~0.2-0.4
    
    Solution: Z-score normalization + sigmoid scaling per modality.
    
    Each modality has calibration parameters (mean, std) learned from
    typical query results. These are used to normalize raw scores to
    a comparable 0-1 range.
    """
    
    # Default calibration values based on empirical measurement (20 queries x 5 results)
    # Format: {modality: (mean, std)}
    # Updated: 2026-02-17 based on test_encoding.py calibration measurement
    DEFAULT_CALIBRATION = {
        "image": (0.271, 0.028),  # CLIP text-to-image measured range
        "text": (0.885, 0.053),   # CLIP text-to-text measured range  
        "audio": (0.100, 0.021),  # CLAP text-to-audio measured range
    }
    
    def __init__(self, calibration: dict[str, tuple[float, float]] = None):
        """
        Args:
            calibration: Dict mapping modality to (mean, std) tuple.
                        Uses DEFAULT_CALIBRATION if not provided.
        """
        self.calibration = calibration or self.DEFAULT_CALIBRATION.copy()
    
    def normalize(self, score: float, modality: Modality) -> float:
        """
        Normalize a raw similarity score to 0-1 range.
        
        Args:
            score: Raw similarity score from FAISS search
            modality: The modality of the result
            
        Returns:
            Normalized score in [0, 1] range
        """
        if modality not in self.calibration:
            # Fallback: no normalization
            return float(np.clip(score, 0, 1))
        
        mean, std = self.calibration[modality]
        
        # Z-score normalization
        if std > 0:
            z_score = (score - mean) / std
        else:
            z_score = 0.0
        
        # Sigmoid to map to [0, 1]
        normalized = 1 / (1 + np.exp(-z_score))
        
        return float(normalized)
    
    def update_calibration(self, modality: Modality, scores: list[float]):
        """
        Update calibration for a modality based on observed scores.
        
        Args:
            modality: Modality to update
            scores: List of observed raw scores
        """
        if scores:
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            if std > 0.01:  # Avoid division by near-zero
                self.calibration[modality] = (mean, std)


class UnifiedSemanticIndex:
    """
    Unified multi-modal semantic index for DCASS.
    
    Manages FAISS indices for multiple modalities (image, text, audio)
    and provides a unified search interface with score normalization.
    
    Usage:
        index = UnifiedSemanticIndex()
        index.load()  # Loads all available indices
        
        results = index.search("a dog running on beach", k=5)
        for item in results:
            print(f"{item.modality}: {item.id} (score: {item.normalized_score:.3f})")
    """
    
    def __init__(
        self,
        base_path: Path = None,
        device: str = None,
        enabled_modalities: list[Modality] = None
    ):
        """
        Args:
            base_path: Base path for data directory. Defaults to project data/indices/
            device: Device for CLIP model ('cuda' or 'cpu'). Auto-detected if None.
            enabled_modalities: List of modalities to load. Defaults to all available.
        """
        self.base_path = resolve_indices_base_path(base_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled_modalities = enabled_modalities or ["image", "text", "audio"]
        
        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_preprocess = None
        
        # FAISS indices and metadata (loaded on demand)
        self.indices: dict[Modality, faiss.Index] = {}
        self.metadata: dict[Modality, list[dict]] = {}
        
        # Score normalizer
        self.normalizer = ScoreNormalizer()
        
        # Track loaded state
        self._loaded = False
    
    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            print(f"Loading CLIP model (ViT-B/32) on {self.device}...")
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self._clip_model.eval()
        return self._clip_model
    
    def _get_index_paths(self, modality: Modality) -> tuple[Path, Path]:
        """Get paths for index and metadata files."""
        index_path = self.base_path / f"{modality}.index"
        metadata_path = self.base_path / f"{modality}_metadata.json"
        return index_path, metadata_path
    
    def load(self, modalities: list[Modality] = None) -> dict[str, bool]:
        """
        Load FAISS indices and metadata for specified modalities.
        
        Args:
            modalities: List of modalities to load. Uses enabled_modalities if None.
            
        Returns:
            Dict mapping modality to success status
        """
        modalities = modalities or self.enabled_modalities
        status = {}
        
        for modality in modalities:
            index_path, metadata_path = self._get_index_paths(modality)
            
            try:
                if not index_path.exists():
                    print(f"  [{modality}] Index not found: {index_path}")
                    status[modality] = False
                    continue
                
                if not metadata_path.exists():
                    print(f"  [{modality}] Metadata not found: {metadata_path}")
                    status[modality] = False
                    continue
                
                # Load FAISS index
                self.indices[modality] = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata[modality] = json.load(f)
                
                n_items = self.indices[modality].ntotal
                print(f"  [{modality}] Loaded {n_items} items")
                status[modality] = True
                
            except Exception as e:
                print(f"  [{modality}] Error loading: {e}")
                status[modality] = False
        
        self._loaded = any(status.values())
        return status
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text query using CLIP."""
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self.clip_model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype("float32")
    
    def search(
        self,
        query: str,
        k: int = 5,
        modalities: list[Modality] = None,
        min_score: float = 0.0
    ) -> list[MediaItem]:
        """
        Search across all loaded modalities for semantically similar media.
        
        Args:
            query: Text query to search for
            k: Number of results to return (total, not per modality)
            modalities: Modalities to search. Uses all loaded if None.
            min_score: Minimum normalized score threshold
            
        Returns:
            List of MediaItem objects, sorted by normalized score (descending)
        """
        if not self._loaded:
            raise RuntimeError("Index not loaded. Call load() first.")
        
        modalities = modalities or list(self.indices.keys())
        
        # Encode query once
        query_embedding = self._encode_text(query)
        
        all_results = []
        
        for modality in modalities:
            if modality not in self.indices:
                continue
            
            index = self.indices[modality]
            meta_list = self.metadata[modality]
            
            # Search this modality's index
            # Request more results than k to allow for filtering
            n_search = min(k * 2, index.ntotal)
            scores, indices = index.search(query_embedding, n_search)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(meta_list):
                    continue
                
                meta = meta_list[idx]
                normalized = self.normalizer.normalize(score, modality)
                
                if normalized < min_score:
                    continue
                
                content = extract_semantic_content(meta, modality)

                item = MediaItem(
                    id=meta.get("id", f"{modality}_{idx}"),
                    modality=modality,
                    content=content,
                    score=float(score),
                    normalized_score=normalized,
                    metadata=meta
                )
                all_results.append(item)
        
        # Sort by normalized score and return top k
        all_results.sort(key=lambda x: x.normalized_score, reverse=True)
        return all_results[:k]
    
    def search_modality(
        self,
        query: str,
        modality: Modality,
        k: int = 5
    ) -> list[MediaItem]:
        """
        Search a single modality.
        
        Args:
            query: Text query
            modality: Specific modality to search
            k: Number of results
            
        Returns:
            List of MediaItem objects from that modality
        """
        return self.search(query, k=k, modalities=[modality])
    
    def get_by_id(self, item_id: str) -> Optional[MediaItem]:
        """
        Retrieve a media item by its ID.
        
        Args:
            item_id: The unique ID of the item
            
        Returns:
            MediaItem if found, None otherwise
        """
        for modality, meta_list in self.metadata.items():
            for meta in meta_list:
                if meta.get("id") == item_id:
                    content = extract_semantic_content(meta, modality)
                    return MediaItem(
                        id=item_id,
                        modality=modality,
                        content=content,
                        score=1.0,
                        normalized_score=1.0,
                        metadata=meta
                    )
        return None
    
    def status(self) -> dict:
        """
        Get status of loaded indices.
        
        Returns:
            Dict with status information
        """
        return {
            "loaded": self._loaded,
            "device": self.device,
            "modalities": {
                modality: {
                    "loaded": modality in self.indices,
                    "count": self.indices[modality].ntotal if modality in self.indices else 0
                }
                for modality in self.enabled_modalities
            },
            "base_path": str(self.base_path)
        }
    
    def __repr__(self) -> str:
        loaded = list(self.indices.keys())
        counts = [f"{m}:{self.indices[m].ntotal}" for m in loaded]
        return f"UnifiedSemanticIndex(loaded=[{', '.join(counts)}])"


# Convenience function for quick setup
def create_index(base_path: Path = None, device: str = None) -> UnifiedSemanticIndex:
    """
    Create and load a UnifiedSemanticIndex.
    
    Args:
        base_path: Optional custom path to indices
        device: Optional device override
        
    Returns:
        Loaded UnifiedSemanticIndex instance
    """
    index = UnifiedSemanticIndex(base_path=base_path, device=device)
    print("Loading unified semantic index...")
    index.load()
    return index

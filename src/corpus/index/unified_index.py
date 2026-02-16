"""
Unified Semantic Index

CLIP-based unified embedding space for cross-modal semantic search.

KEY DESIGN: All modalities (text, images) are embedded into the SAME 
vector space using CLIP. This allows direct comparison of similarity
scores across modalities.

When encoding a message:
1. Each chunk is embedded using CLIP's text encoder
2. Search ALL indices (text and image) simultaneously  
3. Pick the BEST match regardless of modality
4. Result: Mixed sequence of images AND texts

SCORE NORMALIZATION:
CLIP text-to-text similarity is naturally higher than text-to-image.
To enable fair comparison, we normalize scores within each modality
using calibration statistics and apply optional modality boosting.

This makes detection harder because the output is heterogeneous.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore


@dataclass
class SearchResult:
    """
    Represents a single search result from the index.
    
    Attributes:
        id: Unique identifier of the matched item
        score: Similarity score (higher is better, normalized 0-1)
        modality: Which index this came from ('text', 'image', 'audio')
        content: The content (text string or file path)
        metadata: Additional metadata dict
    """
    id: str
    score: float
    modality: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, score={self.score:.4f}, modality={self.modality})"


class ModalityIndex:
    """
    Wrapper for a single-modality FAISS index.
    
    All indices use CLIP embeddings (512-dim) for unified vector space.
    This allows cross-modal comparison of similarity scores.
    
    Attributes:
        modality: The modality type ('text', 'image', 'audio')
        index_path: Path to the FAISS index file
        metadata_path: Path to the metadata JSON file
    """
    
    def __init__(
        self,
        modality: str,
        index_path: Path,
        metadata_path: Path
    ):
        """
        Initialize a modality index.
        
        Args:
            modality: The modality type
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata JSON
        """
        self.modality = modality
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: Optional[Any] = None  # faiss.Index
        self.metadata: List[Dict[str, Any]] = []
    
    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Build index from embeddings and metadata.
        
        IMPORTANT: Embeddings must be CLIP embeddings (512-dim, normalized).
        
        Args:
            embeddings: numpy array of shape (n_items, 512), L2-normalized
            metadata: List of metadata dicts, one per item
        """
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        dim = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (vectors must be L2-normalized)
        self.index = faiss.IndexFlatIP(dim)
        
        # Normalize embeddings to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        self.index.add(normalized.astype("float32"))
        self.metadata = metadata
        
        print(f"Built {self.modality} index: {self.index.ntotal} items, {dim}-dim")
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")
        
        if faiss is None:
            raise ImportError("faiss is required")
        
        # Create parent directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata as JSON
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {self.modality} index to {self.index_path}")
    
    def load(self) -> None:
        """Load index and metadata from disk."""
        if faiss is None:
            raise ImportError("faiss is required")
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {self.modality} index: {self.index.ntotal} items")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """
        Search this index for similar items.
        
        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,), L2-normalized
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")
        
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype("float32"), k
        )
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            meta = self.metadata[idx]
            results.append(SearchResult(
                id=meta["id"],
                score=float(score),  # Already cosine similarity (0 to 1)
                modality=self.modality,
                content=meta.get("content", ""),
                metadata=meta
            ))
        
        return results
    
    @property
    def size(self) -> int:
        """Return number of items in index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def exists(self) -> bool:
        """Check if index files exist on disk."""
        return self.index_path.exists() and self.metadata_path.exists()


class ScoreNormalizer:
    """
    Normalizes similarity scores across modalities for fair comparison.
    
    PROBLEM: CLIP text-to-text similarity scores are naturally higher (~0.8-0.9)
    than text-to-image similarity (~0.25-0.35). This causes text to always win.
    
    SOLUTION: Apply per-modality normalization using calibration statistics
    and optional modality boosting.
    
    Methods:
    1. Z-score normalization: (score - mean) / std
    2. Min-max normalization: (score - min) / (max - min)
    3. Percentile normalization: rank-based scaling
    4. Modality boosting: multiply by modality-specific factor
    
    Default calibration values are based on empirical CLIP measurements.
    """
    
    # Empirical calibration statistics for CLIP ViT-B/32
    # Based on Flickr8k dataset measurements (actual measured values)
    DEFAULT_CALIBRATION = {
        "text": {
            "mean": 0.79,
            "std": 0.05,
            "min": 0.68,
            "max": 0.87,
            "boost": 1.0,  # No boost for text (baseline)
        },
        "image": {
            "mean": 0.27,
            "std": 0.04,
            "min": 0.18,
            "max": 0.35,
            "boost": 1.0,  # Normalized scores don't need boost with combined method
        },
        "audio": {
            "mean": 0.30,
            "std": 0.08,
            "min": 0.10,
            "max": 0.50,
            "boost": 1.0,
        }
    }
    
    def __init__(
        self,
        method: Literal["zscore", "minmax", "boost", "combined"] = "combined",
        calibration: Optional[Dict[str, Dict[str, float]]] = None,
        image_boost: float = 2.2,
        diversity_ratio: float = 0.0,
    ):
        """
        Initialize the score normalizer.
        
        Args:
            method: Normalization method to use:
                - "zscore": Z-score normalization
                - "minmax": Min-max to [0, 1]
                - "boost": Simple modality boosting
                - "combined": Z-score + clipping (recommended)
            calibration: Optional custom calibration stats per modality
            image_boost: Boost factor for image scores (used with "boost" method)
            diversity_ratio: Force minimum ratio of each modality (0.0-1.0)
        """
        self.method = method
        self.calibration = calibration or self.DEFAULT_CALIBRATION
        self.image_boost = image_boost
        self.diversity_ratio = diversity_ratio
    
    def normalize(
        self,
        results: List[SearchResult],
        modality: str
    ) -> List[SearchResult]:
        """
        Normalize scores for a single modality.
        
        Args:
            results: List of search results from one modality
            modality: The modality name
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        cal = self.calibration.get(modality, self.calibration["text"])
        
        normalized = []
        for r in results:
            if self.method == "zscore":
                # Z-score normalization
                new_score = (r.score - cal["mean"]) / (cal["std"] + 1e-8)
                # Shift to positive range [0, ~2] for comparison
                new_score = (new_score + 2) / 4  # Rough scaling
            
            elif self.method == "minmax":
                # Min-max normalization to [0, 1]
                new_score = (r.score - cal["min"]) / (cal["max"] - cal["min"] + 1e-8)
                new_score = max(0.0, min(1.0, new_score))
            
            elif self.method == "boost":
                # Simple boosting
                new_score = r.score * cal["boost"]
            
            elif self.method == "combined":
                # Combined: Z-score normalization + sigmoid + boost
                z = (r.score - cal["mean"]) / (cal["std"] + 1e-8)
                # Sigmoid to squash to [0, 1]
                new_score = 1 / (1 + np.exp(-z))
                # Apply small boost for underrepresented modalities
                new_score = new_score * cal.get("boost", 1.0)
            
            else:
                new_score = r.score
            
            normalized.append(SearchResult(
                id=r.id,
                score=float(new_score),
                modality=r.modality,
                content=r.content,
                metadata={**r.metadata, "raw_score": r.score}
            ))
        
        return normalized
    
    def normalize_cross_modal(
        self,
        results_by_modality: Dict[str, List[SearchResult]],
        k: int = 5
    ) -> List[SearchResult]:
        """
        Normalize and merge results from multiple modalities.
        
        Args:
            results_by_modality: Dict mapping modality to results
            k: Number of results to return
            
        Returns:
            Merged and sorted results with normalized scores
        """
        all_normalized = []
        
        for modality, results in results_by_modality.items():
            normalized = self.normalize(results, modality)
            all_normalized.extend(normalized)
        
        # Sort by normalized score
        all_normalized.sort(key=lambda x: x.score, reverse=True)
        
        # Apply diversity constraint if specified
        if self.diversity_ratio > 0:
            all_normalized = self._apply_diversity(all_normalized, k)
        
        return all_normalized[:k]
    
    def _apply_diversity(
        self,
        results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """
        Apply diversity constraint to ensure minimum modality representation.
        
        Args:
            results: Sorted results
            k: Target number of results
            
        Returns:
            Diversified results
        """
        if not results:
            return results
        
        # Count modalities
        modalities = set(r.modality for r in results)
        min_per_modality = max(1, int(k * self.diversity_ratio))
        
        # Select minimum from each modality first
        selected = []
        remaining = []
        counts: Dict[str, int] = {m: 0 for m in modalities}
        
        for r in results:
            if counts[r.modality] < min_per_modality:
                selected.append(r)
                counts[r.modality] += 1
            else:
                remaining.append(r)
        
        # Fill remaining slots with best overall
        remaining.sort(key=lambda x: x.score, reverse=True)
        while len(selected) < k and remaining:
            selected.append(remaining.pop(0))
        
        # Re-sort by score
        selected.sort(key=lambda x: x.score, reverse=True)
        return selected


class UnifiedSemanticIndex:
    """
    Unified multi-modal semantic index using CLIP embeddings.
    
    KEY FEATURE: All modalities share the same CLIP embedding space.
    This enables TRUE cross-modal search where we can directly compare
    similarity scores between text and images.
    
    When modality="auto":
    - Searches ALL loaded indices simultaneously
    - Returns best matches REGARDLESS of modality
    - Results can be mixed: [image, text, image, text, ...]
    
    This is the core of DCASS's steganographic approach:
    - Message chunks map to a MIX of images and texts
    - Makes pattern detection much harder
    
    Usage:
        index = UnifiedSemanticIndex()
        index.load()  # Load all available indices
        
        # Cross-modal search - returns best match from ANY modality
        results = index.search("a dog running", modality="auto", k=5)
        
        # Encode message into mixed media sequence
        sequence = index.encode_message("Secret meeting at dawn")
        # sequence might be: [image, text, image, image, text]
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        normalize_scores: bool = True,
        normalization_method: Literal["zscore", "minmax", "boost", "combined"] = "combined",
        diversity_ratio: float = 0.0,
    ):
        """
        Initialize the unified index.
        
        Args:
            config: Optional configuration object. If None, loads from default.
            normalize_scores: Whether to normalize scores across modalities
            normalization_method: Method for score normalization
            diversity_ratio: Force minimum ratio of each modality (0.0-1.0)
        """
        if config is None:
            from config.settings import config as default_config
            config = default_config
        
        self.config = config
        self._indices: Dict[str, ModalityIndex] = {}
        self._clip_embedder: Optional[Any] = None  # Single CLIP embedder for all
        
        # Score normalization settings
        self.normalize_scores = normalize_scores
        self._normalizer = ScoreNormalizer(
            method=normalization_method,
            diversity_ratio=diversity_ratio,
        ) if normalize_scores else None
        
        # Initialize index objects (not loaded yet)
        self._init_indices()
    
    def _init_indices(self) -> None:
        """Initialize ModalityIndex objects based on configuration."""
        # Text index (uses CLIP text embeddings)
        if self.config.get("corpus.text.enabled", True):
            self._indices["text"] = ModalityIndex(
                modality="text",
                index_path=self.config.get_path("index.text.path"),
                metadata_path=self.config.get_path("index.text.metadata_path")
            )
        
        # Image index (uses CLIP image embeddings)
        if self.config.get("corpus.image.enabled", True):
            self._indices["image"] = ModalityIndex(
                modality="image",
                index_path=self.config.get_path("index.image.path"),
                metadata_path=self.config.get_path("index.image.metadata_path")
            )
        
        # Audio index (future - would need AudioCLIP or similar)
        if self.config.get("corpus.audio.enabled", False):
            self._indices["audio"] = ModalityIndex(
                modality="audio",
                index_path=self.config.get_path("index.audio.path"),
                metadata_path=self.config.get_path("index.audio.metadata_path")
            )
    
    def _get_clip_embedder(self) -> Any:
        """
        Get or create the CLIP embedder.
        
        We use a SINGLE CLIP model for all modalities to ensure
        embeddings are in the same vector space.
        
        Returns:
            ImageEmbedder (which is actually CLIP and can encode both text and images)
        """
        if self._clip_embedder is None:
            from src.corpus.embedders import ImageEmbedder
            
            device = self.config.get_device()
            model_name = self.config.get("embeddings.image.model", "ViT-B/32")
            
            self._clip_embedder = ImageEmbedder(
                model_name=model_name,
                device=device
            )
            print(f"Initialized CLIP embedder: {model_name} on {device}")
        
        return self._clip_embedder
    
    def load(self, modalities: Optional[List[str]] = None) -> None:
        """
        Load indices from disk.
        
        Args:
            modalities: List of modalities to load. If None, loads all available.
        """
        if modalities is None:
            modalities = list(self._indices.keys())
        
        loaded_count = 0
        for mod in modalities:
            if mod in self._indices:
                idx = self._indices[mod]
                if idx.exists():
                    idx.load()
                    loaded_count += 1
                else:
                    print(f"Warning: {mod} index not found at {idx.index_path}")
        
        if loaded_count == 0:
            print("Warning: No indices loaded. Run 'python scripts/build_indices.py' first.")
    
    def search(
        self,
        query: str,
        modality: Literal["text", "image", "audio", "auto"] = "auto",
        k: int = 5,
        normalize: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Search for semantically similar content.
        
        IMPORTANT: When modality="auto", searches ALL modalities and returns
        the best matches regardless of type. This enables mixed-modality encoding.
        
        Score normalization is applied by default when modality="auto" to ensure
        fair comparison between text and image scores.
        
        Args:
            query: Text query to search for
            modality: Which index to search:
                - "text": Search text index only
                - "image": Search image index only  
                - "audio": Search audio index only
                - "auto": Search ALL indices, return best matches
            k: Number of results to return
            normalize: Override score normalization (None = use default)
            
        Returns:
            List of SearchResult objects, sorted by score (highest first)
        """
        # Get CLIP embedder and encode query text
        embedder = self._get_clip_embedder()
        query_emb = embedder.encode_text([query])  # Shape: (1, 512)
        
        # Determine if we should normalize
        should_normalize = normalize if normalize is not None else self.normalize_scores
        
        if modality == "auto":
            # Search ALL modalities
            results_by_modality: Dict[str, List[SearchResult]] = {}
            
            for mod_name, mod_idx in self._indices.items():
                if mod_idx.index is not None:
                    results = mod_idx.search(query_emb, k=k)
                    results_by_modality[mod_name] = results
            
            # Apply normalization if enabled
            if should_normalize and self._normalizer:
                return self._normalizer.normalize_cross_modal(results_by_modality, k)
            else:
                # No normalization - just merge and sort
                all_results = []
                for results in results_by_modality.values():
                    all_results.extend(results)
                all_results.sort(key=lambda x: x.score, reverse=True)
                return all_results[:k]
        else:
            # Search single modality
            if modality not in self._indices:
                raise ValueError(f"Unknown modality: {modality}")
            
            idx = self._indices[modality]
            if idx.index is None:
                raise RuntimeError(f"{modality} index not loaded. Call load() first.")
            
            return idx.search(query_emb, k=k)
            
            return idx.search(query_emb, k=k)
    
    def search_all_modalities(
        self,
        query: str,
        k_per_modality: int = 3
    ) -> Dict[str, List[SearchResult]]:
        """
        Search all modalities and return results grouped by modality.
        
        Useful for debugging and understanding what each index contains.
        
        Args:
            query: Text query
            k_per_modality: Number of results per modality
            
        Returns:
            Dict mapping modality name to list of results
        """
        embedder = self._get_clip_embedder()
        query_emb = embedder.encode_text([query])
        
        results = {}
        for mod_name, mod_idx in self._indices.items():
            if mod_idx.index is not None:
                results[mod_name] = mod_idx.search(query_emb, k=k_per_modality)
            else:
                results[mod_name] = []
        
        return results
    
    def encode_message(
        self,
        message: str,
        modality: Literal["text", "image", "auto"] = "auto",
        k_per_chunk: int = 1
    ) -> List[SearchResult]:
        """
        Encode a message into a sequence of media references.
        
        This is the CORE DCASS encoding function.
        
        When modality="auto" (default):
        - Each chunk searches ALL modalities
        - Best match is selected regardless of type
        - Result is a MIX of images and texts
        
        Example:
            Input: "Secret meeting at dawn in the park"
            Output: [
                SearchResult(content="whisper.jpg", modality="image"),
                SearchResult(content="The sun rises...", modality="text"),
                SearchResult(content="park_bench.jpg", modality="image"),
            ]
        
        Args:
            message: The secret message to encode
            modality: "auto" for mixed, or specific modality
            k_per_chunk: Number of candidates per chunk (1 = deterministic)
            
        Returns:
            List of SearchResult objects representing the encoded sequence
        """
        from src.engine.chunker import SemanticChunker
        
        # Chunk the message
        chunker = SemanticChunker()
        chunks = chunker.chunk(message)
        
        if not chunks:
            chunks = [message.strip()]
        
        # Encode each chunk - using "auto" for mixed modality
        encoded_sequence = []
        for chunk in chunks:
            results = self.search(chunk, modality=modality, k=k_per_chunk)
            if results:
                encoded_sequence.append(results[0])
        
        return encoded_sequence
    
    def get_index(self, modality: str) -> ModalityIndex:
        """
        Get a specific modality index.
        
        Args:
            modality: The modality type
            
        Returns:
            ModalityIndex object
        """
        if modality not in self._indices:
            raise ValueError(f"Unknown modality: {modality}")
        return self._indices[modality]
    
    @property
    def available_modalities(self) -> List[str]:
        """Return list of configured modalities."""
        return list(self._indices.keys())
    
    @property
    def loaded_modalities(self) -> List[str]:
        """Return list of modalities with loaded indices."""
        return [m for m, idx in self._indices.items() if idx.index is not None]
    
    def status(self) -> Dict[str, Any]:
        """
        Get status of all indices.
        
        Returns:
            Dict with status information for each modality
        """
        status = {}
        for mod, idx in self._indices.items():
            status[mod] = {
                "configured": True,
                "exists_on_disk": idx.exists(),
                "loaded": idx.index is not None,
                "size": idx.size,
                "index_path": str(idx.index_path),
                "metadata_path": str(idx.metadata_path),
            }
        return status
    
    def __repr__(self) -> str:
        loaded = self.loaded_modalities
        return f"UnifiedSemanticIndex(loaded={loaded})"

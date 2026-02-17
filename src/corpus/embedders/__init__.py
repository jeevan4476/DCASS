# src/corpus/embedders/__init__.py
"""
Embedder modules for DCASS.

Provides unified embedding interfaces for different modalities:
- CLIPEmbedder: For images and text (512-dim, cross-modal)
- AudioEmbedder: For audio using CLAP (512-dim, compatible with CLIP)
- VectorEngine: Legacy SentenceTransformer-based embedder (384-dim)
"""

from .clip_embedder import CLIPEmbedder
from .vector_engine import VectorEngine

__all__ = ["CLIPEmbedder", "VectorEngine"]

# AudioEmbedder imported conditionally to avoid CLAP dependency issues
try:
    from .audio_embedder import AudioEmbedder
    __all__.append("AudioEmbedder")
except ImportError:
    pass

# src/corpus/embedders/__init__.py
"""
Embedder modules for DCASS.

Provides unified embedding interfaces for different modalities:
- ImageEmbedder: CLIP ViT-B/32 (512-dim, cross-modal; used for images AND text)
- AudioEmbedder: CLAP (512-dim, compatible with the CLIP latent width)
"""

from .image_embedder import ImageEmbedder

__all__ = ["ImageEmbedder"]

# AudioEmbedder imported conditionally to avoid CLAP dependency issues
try:
    from .audio_embedder import AudioEmbedder  # noqa: F401 (re-exported)

    __all__.append("AudioEmbedder")
except ImportError:
    pass

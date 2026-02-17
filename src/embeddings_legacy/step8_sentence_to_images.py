# src/embeddings/step8_sentence_to_images.py
"""
Legacy sentence-to-image encoder.

This module provides backward-compatible encoding using CLIP and FAISS.
For new code, prefer using src.engine.encoder.SemanticEncoder instead.

Usage:
    from src.embeddings.step8_sentence_to_images import sentence_to_image_sequence
    
    images = sentence_to_image_sequence("a dog running on the beach")
"""

import json
import torch
import clip
import faiss
import numpy as np
from pathlib import Path
import re

# -------- Paths (Fixed to use actual data location) --------
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMB_DIR = PROJECT_ROOT / "data" / "indices"
FAISS_INDEX_PATH = EMB_DIR / "image.index"
IMAGE_METADATA_PATH = EMB_DIR / "image_metadata.json"

TOP_K = 1

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Lazy Loading (avoid loading on import) --------
_model = None
_index = None
_image_metadata = None


def _ensure_loaded():
    """Lazy load CLIP model and FAISS index."""
    global _model, _index, _image_metadata
    
    if _model is None:
        print(f"Loading CLIP model (ViT-B/32) on {device}...")
        _model, _ = clip.load("ViT-B/32", device=device)
        _model.eval()
    
    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}\n"
                "Please ensure the index has been built."
            )
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    if _image_metadata is None:
        if not IMAGE_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Image metadata not found at {IMAGE_METADATA_PATH}\n"
                "Please ensure the metadata file exists."
            )
        print(f"Loading image metadata from {IMAGE_METADATA_PATH}...")
        with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as f:
            _image_metadata = json.load(f)


def _get_image_id(index_position: int) -> str:
    """Get image ID from metadata by index position."""
    if index_position < len(_image_metadata):
        return _image_metadata[index_position].get("id", f"image_{index_position}")
    return f"image_{index_position}"


def _get_image_caption(index_position: int) -> str:
    """Get image caption from metadata by index position."""
    if index_position < len(_image_metadata):
        meta = _image_metadata[index_position]
        return meta.get("caption", meta.get("captions", [""])[0] if meta.get("captions") else "")
    return ""


# -------- Chunking --------
def chunk_text(text: str) -> list[str]:
    """Split text into semantic chunks."""
    chunks = re.split(r",\s*|\s+and\s+", text.lower())
    return [c.strip() for c in chunks if c.strip()]


# ======================================================
# PUBLIC API — THIS IS WHAT CLI IMPORTS
# ======================================================
def sentence_to_image_sequence(sentence: str, verbose: bool = True) -> list[str]:
    """
    Encodes a sentence into a list of image IDs.
    
    This is the main public function for encoding messages into
    image sequences using CLIP semantic search.
    
    Args:
        sentence: The message to encode
        verbose: Whether to print progress
        
    Returns:
        List of image IDs representing the encoded message
    """
    _ensure_loaded()
    
    chunks = chunk_text(sentence)
    results = []

    if verbose:
        print("\nSentence:")
        print(sentence)
        print("\nChunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. {c}")
        print("\nEncoded image sequence:")

    with torch.no_grad():
        for i, chunk in enumerate(chunks, 1):
            tokens = clip.tokenize([chunk], truncate=True).to(device)
            emb = _model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy().astype("float32")

            scores, indices = _index.search(emb, TOP_K)
            idx = indices[0][0]
            img_id = _get_image_id(idx)
            
            if verbose:
                caption = _get_image_caption(idx)
                score = scores[0][0]
                print(f"  {i}. '{chunk}' -> {img_id} (score: {score:.3f})")
                if caption:
                    print(f"      Caption: \"{caption[:60]}...\"" if len(caption) > 60 else f"      Caption: \"{caption}\"")
            
            results.append(img_id)

    return results


def sentence_to_images_with_scores(sentence: str) -> list[tuple[str, float, str]]:
    """
    Encode sentence and return IDs with scores and captions.
    
    Args:
        sentence: Message to encode
        
    Returns:
        List of (image_id, score, caption) tuples
    """
    _ensure_loaded()
    
    chunks = chunk_text(sentence)
    results = []
    
    with torch.no_grad():
        for chunk in chunks:
            tokens = clip.tokenize([chunk], truncate=True).to(device)
            emb = _model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy().astype("float32")
            
            scores, indices = _index.search(emb, TOP_K)
            idx = indices[0][0]
            
            img_id = _get_image_id(idx)
            score = float(scores[0][0])
            caption = _get_image_caption(idx)
            
            results.append((img_id, score, caption))
    
    return results


# ======================================================
# SCRIPT MODE (kept for standalone testing)
# ======================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step8_sentence_to_images.py <sentence>")
        print("\nExample:")
        print('  python step8_sentence_to_images.py "a dog running on the beach"')
        sys.exit(1)

    sentence = sys.argv[1]
    sentence_to_image_sequence(sentence)

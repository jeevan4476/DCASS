#!/usr/bin/env python3
"""
Add Wikipedia sentences to the text FAISS index.

This script:
1. Loads existing text.index and text_metadata.json
2. Loads Wikipedia sentences from sentences_10k.json
3. Encodes Wikipedia text using CLIP (same model as captions)
4. Appends to existing FAISS index
5. Updates metadata JSON
6. Saves updated index (30,000 + 10,000 = 40,000 vectors)

Usage:
    python scripts/add_wikipedia_to_index.py

Estimated time: ~5-10 minutes on CPU
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

import torch
import clip

# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Input files
WIKIPEDIA_JSON = DATA_DIR / "raw" / "wikipedia" / "sentences_10k.json"

# Index files (will be modified)
TEXT_INDEX_PATH = DATA_DIR / "indices" / "text.index"
TEXT_METADATA_PATH = DATA_DIR / "indices" / "text_metadata.json"

# CLIP settings
CLIP_MODEL = "ViT-B/32"
BATCH_SIZE = 64  # Process texts in batches for efficiency

# ============================================


def load_wikipedia_sentences(json_path: Path) -> list[dict]:
    """Load Wikipedia sentences from JSON file."""
    print(f"Loading Wikipedia sentences from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} sentences")
    return data


def load_existing_index(index_path: Path, metadata_path: Path):
    """Load existing FAISS index and metadata."""
    print(f"Loading existing index from {index_path}...")
    index = faiss.read_index(str(index_path))
    print(f"  Index has {index.ntotal} vectors, dimension {index.d}")
    
    print(f"Loading existing metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  Metadata has {len(metadata)} entries")
    
    return index, metadata


def encode_texts_clip(texts: list[str], model, device: str, batch_size: int = 64) -> np.ndarray:
    """Encode texts using CLIP model."""
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            
            # Tokenize and encode
            tokens = clip.tokenize(batch, truncate=True).to(device)
            embeddings = model.encode_text(tokens)
            
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings).astype("float32")


def main():
    print("=" * 60)
    print(" Adding Wikipedia to Text Index")
    print("=" * 60)
    
    # Check files exist
    if not WIKIPEDIA_JSON.exists():
        raise FileNotFoundError(f"Wikipedia JSON not found: {WIKIPEDIA_JSON}")
    if not TEXT_INDEX_PATH.exists():
        raise FileNotFoundError(f"Text index not found: {TEXT_INDEX_PATH}")
    if not TEXT_METADATA_PATH.exists():
        raise FileNotFoundError(f"Text metadata not found: {TEXT_METADATA_PATH}")
    
    # Load Wikipedia data
    wiki_data = load_wikipedia_sentences(WIKIPEDIA_JSON)
    
    # Load existing index
    index, metadata = load_existing_index(TEXT_INDEX_PATH, TEXT_METADATA_PATH)
    
    # Check for duplicates (already added?)
    existing_sources = set(m.get("source", "") for m in metadata)
    if "wikipedia" in existing_sources:
        wiki_count = sum(1 for m in metadata if m.get("source") == "wikipedia")
        print(f"\n WARNING: Wikipedia already in index ({wiki_count} entries)")
        response = input("Continue and add more? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return
    
    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP model ({CLIP_MODEL}) on {device}...")
    model, _ = clip.load(CLIP_MODEL, device=device)
    model.eval()
    
    # Extract texts
    wiki_texts = [item["text"] for item in wiki_data]
    print(f"\nEncoding {len(wiki_texts)} Wikipedia sentences...")
    
    # Encode
    wiki_embeddings = encode_texts_clip(wiki_texts, model, device, BATCH_SIZE)
    print(f"  Embeddings shape: {wiki_embeddings.shape}")
    
    # Add to FAISS index
    print(f"\nAdding {len(wiki_embeddings)} vectors to FAISS index...")
    index.add(wiki_embeddings)
    print(f"  New index size: {index.ntotal} vectors")
    
    # Update metadata
    print(f"Updating metadata...")
    for item in wiki_data:
        # Ensure consistent format
        metadata.append({
            "id": item["id"],
            "text": item["text"],
            "content": item.get("content", item["text"]),
            "source": "wikipedia",
            "article": item.get("article", "unknown"),
            "modality": "text"
        })
    print(f"  New metadata size: {len(metadata)} entries")
    
    # Save updated index
    print(f"\nSaving updated index to {TEXT_INDEX_PATH}...")
    faiss.write_index(index, str(TEXT_INDEX_PATH))
    
    print(f"Saving updated metadata to {TEXT_METADATA_PATH}...")
    with open(TEXT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print(" COMPLETE")
    print("=" * 60)
    print(f"  Total vectors: {index.ntotal}")
    
    # Count by source
    sources = {}
    for m in metadata:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"  By source: {sources}")
    
    print("\nYou can now test with:")
    print('  python scripts/demo_dcass.py "Your message here"')


if __name__ == "__main__":
    main()

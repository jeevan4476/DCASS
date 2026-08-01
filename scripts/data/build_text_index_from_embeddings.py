#!/usr/bin/env python3
"""
Build Text FAISS Index from HuggingFace Caption Embeddings and Wikipedia Sentences.

Reads pre-computed caption_embeddings.npy and combines them with raw Wikipedia
sentences to generate `storage/data/indices/text.index` and `text_metadata.json`.
"""

import json
from pathlib import Path
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "storage" / "data" / "embeddings"
RAW_WIKI_DIR = PROJECT_ROOT / "storage" / "data" / "raw" / "wikipedia"
INDICES_DIR = PROJECT_ROOT / "storage" / "data" / "indices"

def main():
    print("Loading caption embeddings...")
    npy_path = EMBEDDINGS_DIR / "caption_embeddings.npy"
    if not npy_path.exists():
        print(f"Error: {npy_path} not found.")
        return

    embeddings = np.load(npy_path).astype(np.float32)
    print(f"  Shape: {embeddings.shape}")
    faiss.normalize_L2(embeddings)

    metadata = []
    map_csv = EMBEDDINGS_DIR / "caption_map.csv"
    if map_csv.exists():
        with open(map_csv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    metadata.append({
                        "id": parts[0],
                        "content": parts[1],
                        "modality": "text"
                    })
    
    wiki_json = RAW_WIKI_DIR / "sentences.json"
    if wiki_json.exists():
        with open(wiki_json, "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
            for item in wiki_data:
                metadata.append({
                    "id": item.get("id", f"wiki_{len(metadata)}"),
                    "content": item.get("text", item.get("content", "")),
                    "modality": "text"
                })

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDICES_DIR / "text.index"
    meta_path = INDICES_DIR / "text_metadata.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved text.index ({index.ntotal} vectors) to {index_path}")
    print(f"Saved text_metadata.json ({len(metadata)} entries) to {meta_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fit Spherical K-Means Voronoi Codebook Partitioning (VCP) across Multi-Modal FAISS Embeddings.

Fits 256 centroids on the 153,281 512-dim vectors (Image, Text, Audio)
and saves `storage/data/indices/voronoi_codebook.npz`.
"""

import sys
from pathlib import Path
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus.cluster.voronoi_codebook import VoronoiCodebook

INDICES_DIR = PROJECT_ROOT / "storage" / "data" / "indices"

def main():
    print("=" * 70)
    print("DCASS Voronoi Codebook Partitioning (VCP) - Spherical K-Means Fitting")
    print("=" * 70)

    # 1. Load FAISS indices
    img_idx_path = INDICES_DIR / "image.index"
    txt_idx_path = INDICES_DIR / "text.index"
    aud_idx_path = INDICES_DIR / "audio.index"

    all_embeddings = []

    if img_idx_path.exists():
        idx = faiss.read_index(str(img_idx_path))
        vec_arr = idx.reconstruct_n(0, idx.ntotal)
        all_embeddings.append(vec_arr)
        print(f"Loaded Image Index: {idx.ntotal:,} vectors ({idx.d}d)")

    if txt_idx_path.exists():
        idx = faiss.read_index(str(txt_idx_path))
        vec_arr = idx.reconstruct_n(0, idx.ntotal)
        all_embeddings.append(vec_arr)
        print(f"Loaded Text Index:  {idx.ntotal:,} vectors ({idx.d}d)")

    if aud_idx_path.exists():
        idx = faiss.read_index(str(aud_idx_path))
        vec_arr = idx.reconstruct_n(0, idx.ntotal)
        all_embeddings.append(vec_arr)
        print(f"Loaded Audio Index: {idx.ntotal:,} vectors ({idx.d}d)")

    if not all_embeddings:
        print("Error: No indices found in storage/data/indices!")
        return

    corpus_vectors = np.vstack(all_embeddings).astype(np.float32)
    print(f"\nTotal Unified Multi-Modal Vectors: {corpus_vectors.shape[0]:,} (Dimension: {corpus_vectors.shape[1]})")

    # 2. Fit Voronoi Codebook
    codebook = VoronoiCodebook(num_clusters=256, dim=512, delta_margin=0.05)
    codebook.fit(corpus_vectors, max_iters=25, device="cuda")

    # 3. Save Codebook
    save_path = INDICES_DIR / "voronoi_codebook.npz"
    codebook.save(save_path)
    print(f"Saved fitted Voronoi Codebook to {save_path}")

if __name__ == "__main__":
    main()

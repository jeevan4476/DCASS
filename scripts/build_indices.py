#!/usr/bin/env python3
"""
Build FAISS Indices using CLIP Embeddings

Builds FAISS indices for all modalities using CLIP embeddings.

IMPORTANT: All indices use CLIP embeddings (512-dim) to enable
cross-modal comparison. This means:
- Images are encoded with CLIP's image encoder
- Texts are encoded with CLIP's text encoder
- Both produce vectors in the SAME 512-dim space
- This allows direct similarity comparison across modalities

Usage:
    python scripts/build_indices.py
    python scripts/build_indices.py --modality image
    python scripts/build_indices.py --modality text --batch-size 64
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def build_image_index(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    max_items: Optional[int] = None
) -> bool:
    """
    Build FAISS index for images using CLIP.
    
    Args:
        data_dir: Directory containing Flickr8k data
        output_dir: Directory to save index files
        batch_size: Batch size for embedding generation
        max_items: Maximum items to process (for testing)
        
    Returns:
        True if successful
    """
    from src.corpus.loaders.flickr_loader import FlickrLoader
    from src.corpus.embedders.image_embedder import ImageEmbedder
    from src.corpus.index.unified_index import ModalityIndex
    
    print("\n" + "=" * 60)
    print("Building Image Index (CLIP embeddings)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    loader = FlickrLoader(data_dir)
    items = list(loader.load())  # Convert to list for indexing
    
    if not items:
        print("No items loaded. Check data directory.")
        return False
    
    if max_items:
        items = items[:max_items]
    
    print(f"Loaded {len(items)} items")
    
    # Initialize CLIP embedder
    print("\nInitializing CLIP embedder (ViT-B/32)...")
    embedder = ImageEmbedder(model_name="ViT-B/32")
    
    # Generate embeddings
    print(f"\nGenerating CLIP image embeddings (batch_size={batch_size})...")
    start_time = time.time()
    
    all_embeddings = []
    all_metadata = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Get image paths
        image_paths = [item["image_path"] for item in batch]
        
        # Encode images with CLIP
        try:
            embeddings = embedder.encode_images(image_paths, show_progress=False)
            
            if embeddings is not None:
                all_embeddings.append(embeddings)
                
                # Build metadata for each item
                for item in batch:
                    all_metadata.append({
                        "id": item["id"],
                        "content": item["image_path"],
                        "caption": item.get("caption", ""),
                        "captions": item.get("captions", []),
                        "modality": "image"
                    })
        except Exception as e:
            print(f"\n  Warning: Failed to encode batch {i}: {e}")
            continue
        
        # Progress
        processed = min(i + batch_size, len(items))
        print(f"  Processed {processed}/{len(items)} images...", end="\r")
    
    print()  # Newline after progress
    
    if not all_embeddings:
        print("No embeddings generated!")
        return False
    
    # Stack embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    elapsed = time.time() - start_time
    print(f"Embedding time: {elapsed:.1f}s ({len(items)/elapsed:.1f} items/s)")
    
    # Build index
    print("\nBuilding FAISS index...")
    output_dir = Path(output_dir)
    index = ModalityIndex(
        modality="image",
        index_path=output_dir / "image.index",
        metadata_path=output_dir / "image_metadata.json"
    )
    
    index.build(embeddings_array, all_metadata)
    index.save()
    
    print(f"\nImage index saved to {output_dir}")
    return True


def build_text_index_from_captions(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    max_items: Optional[int] = None
) -> bool:
    """
    Build FAISS index for text using CLIP.
    
    Uses Flickr captions as the text corpus and embeds them
    using CLIP's text encoder for cross-modal compatibility.
    
    Args:
        data_dir: Directory containing Flickr8k data
        output_dir: Directory to save index files
        batch_size: Batch size for embedding generation
        max_items: Maximum items to process
        
    Returns:
        True if successful
    """
    from src.corpus.loaders.flickr_loader import FlickrLoader
    from src.corpus.embedders.image_embedder import ImageEmbedder  # CLIP can encode text too!
    from src.corpus.index.unified_index import ModalityIndex
    
    print("\n" + "=" * 60)
    print("Building Text Index (CLIP embeddings from captions)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading captions from {data_dir}...")
    loader = FlickrLoader(data_dir)
    items = list(loader.load())
    
    if not items:
        print("No items loaded.")
        return False
    
    # Extract unique captions
    captions = []
    for item in items:
        for i, caption in enumerate(item.get("captions", [])):
            if caption.strip():  # Skip empty captions
                captions.append({
                    "id": f"{item['id']}_cap{i}",
                    "text": caption.strip(),
                    "content": caption.strip(),  # content field for consistency
                    "source": "flickr8k",
                    "image_id": item["id"],
                    "modality": "text"
                })
    
    if max_items:
        captions = captions[:max_items]
    
    print(f"Extracted {len(captions)} captions")
    
    # Initialize CLIP embedder (same model for both text and images!)
    print("\nInitializing CLIP embedder (ViT-B/32) for text...")
    embedder = ImageEmbedder(model_name="ViT-B/32")
    
    # Generate embeddings
    print(f"\nGenerating CLIP text embeddings (batch_size={batch_size})...")
    start_time = time.time()
    
    all_embeddings = []
    all_metadata = []
    
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        try:
            # Use CLIP's text encoder
            embeddings = embedder.encode_text(texts)
            
            if embeddings is not None:
                all_embeddings.append(embeddings)
                all_metadata.extend(batch)
        except Exception as e:
            print(f"\n  Warning: Failed to encode batch {i}: {e}")
            continue
        
        processed = min(i + batch_size, len(captions))
        print(f"  Processed {processed}/{len(captions)} captions...", end="\r")
    
    print()
    
    if not all_embeddings:
        print("No embeddings generated!")
        return False
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    elapsed = time.time() - start_time
    print(f"Embedding time: {elapsed:.1f}s ({len(captions)/elapsed:.1f} items/s)")
    
    # Build index
    print("\nBuilding FAISS index...")
    output_dir = Path(output_dir)
    index = ModalityIndex(
        modality="text",
        index_path=output_dir / "text.index",
        metadata_path=output_dir / "text_metadata.json"
    )
    
    index.build(embeddings_array, all_metadata)
    index.save()
    
    print(f"\nText index saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices using CLIP embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: All indices use CLIP embeddings (512-dim) for cross-modal search.
This means you can compare text and image similarity directly!

Examples:
    # Build all indices (both image and text)
    python scripts/build_indices.py
    
    # Build only image index
    python scripts/build_indices.py --modality image
    
    # Build only text index (from Flickr captions)
    python scripts/build_indices.py --modality text
    
    # Quick test with limited items
    python scripts/build_indices.py --max-items 100
        """
    )
    
    parser.add_argument(
        "--modality", "-m",
        choices=["image", "text", "all"],
        default="all",
        help="Which modality to build (default: all)"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data/raw/flickr8k"),
        help="Data directory (default: data/raw/flickr8k)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/indices"),
        help="Output directory for indices (default: data/indices)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum items to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    if args.modality in ["image", "all"]:
        if not build_image_index(
            args.data_dir,
            args.output_dir,
            args.batch_size,
            args.max_items
        ):
            success = False
    
    if args.modality in ["text", "all"]:
        if not build_text_index_from_captions(
            args.data_dir,
            args.output_dir,
            args.batch_size,
            args.max_items
        ):
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("Index building complete!")
        print(f"\nIndices saved to: {args.output_dir.absolute()}")
        print("\nBoth indices use CLIP embeddings (512-dim) for cross-modal search.")
        print("\nNext steps:")
        print("  python -m src.cli.main encode 'your secret message'")
        print("  # This will return a MIX of images and texts!")
    else:
        print("Index building completed with errors.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

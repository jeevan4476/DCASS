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
    python scripts/build_indices.py --include-wikipedia  # Add Wikipedia sentences
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def _save_faiss_index(embeddings: np.ndarray, metadata, index_path: Path, metadata_path: Path) -> None:
    """Write embeddings to a FAISS IndexFlatIP and metadata to JSON (cosine sim on normalized vectors)."""
    import faiss

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)

    for i, m in enumerate(metadata):
        m["index"] = i

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Wrote {index.ntotal} vectors ({dim}d) to {index_path}")
    print(f"  Wrote metadata to {metadata_path}")


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
    _save_faiss_index(
        embeddings_array,
        all_metadata,
        output_dir / "image.index",
        output_dir / "image_metadata.json",
    )

    print(f"\nImage index saved to {output_dir}")
    return True


def build_text_index(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    max_items: Optional[int] = None,
    include_wikipedia: bool = False,
    wikipedia_dir: Optional[Path] = None,
) -> bool:
    """
    Build FAISS index for text using CLIP.
    
    Uses Flickr captions + optionally Wikipedia sentences.
    
    Args:
        data_dir: Directory containing Flickr8k data
        output_dir: Directory to save index files
        batch_size: Batch size for embedding generation
        max_items: Maximum items to process
        include_wikipedia: Whether to include Wikipedia sentences
        wikipedia_dir: Directory containing Wikipedia data
        
    Returns:
        True if successful
    """
    from src.corpus.loaders.flickr_loader import FlickrLoader
    from src.corpus.embedders.image_embedder import ImageEmbedder  # CLIP can encode text too!
    
    print("\n" + "=" * 60)
    print("Building Text Index (CLIP embeddings)")
    print("=" * 60)
    
    all_texts = []

    # Try Flickr30k CSV parser first (Captions.csv / results.csv / flickr_annotations_30k.csv).
    # FlickrLoader was built for the Flickr8k token-file format and doesn't
    # understand the Kaggle/HuggingFace Flickr30k CSVs, so we fall through.
    flickr30k_loaded = False
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_flickr30k_index import find_flickr30k_files, load_flickr30k_captions  # type: ignore

        _, captions_file = find_flickr30k_files(Path(data_dir))
        if captions_file is not None:
            print(f"\nDetected Flickr30k-style captions at {captions_file}")
            f30k_captions = load_flickr30k_captions(captions_file)
            for image_name, caps in f30k_captions.items():
                image_id = f"flickr30k_{Path(image_name).stem}"
                for i, caption in enumerate(caps):
                    caption = caption.strip()
                    if not caption:
                        continue
                    all_texts.append({
                        "id": f"{image_id}_cap{i}",
                        "text": caption,
                        "content": caption,
                        "source": "flickr30k",
                        "image_id": image_id,
                        "modality": "text",
                    })
            if all_texts:
                print(f"  Loaded {len(all_texts)} Flickr30k captions")
                flickr30k_loaded = True
    except Exception as e:
        print(f"  (Flickr30k parser unavailable: {e}) — falling back to FlickrLoader")

    # Fall back to Flickr8k-style loader
    if not flickr30k_loaded:
        print(f"\nLoading Flickr captions from {data_dir}...")
        loader = FlickrLoader(data_dir)
        items = list(loader.load())

        if items:
            for item in items:
                for i, caption in enumerate(item.get("captions", [])):
                    if caption.strip():
                        all_texts.append({
                            "id": f"{item['id']}_cap{i}",
                            "text": caption.strip(),
                            "content": caption.strip(),
                            "source": "flickr8k",
                            "image_id": item["id"],
                            "modality": "text"
                        })
            print(f"  Loaded {len(all_texts)} Flickr captions")
    
    # Load Wikipedia sentences if requested
    if include_wikipedia:
        wiki_dir = wikipedia_dir or Path("data/raw/wikipedia")
        wiki_file = wiki_dir / "sentences.json"
        
        if wiki_file.exists():
            print(f"\nLoading Wikipedia sentences from {wiki_file}...")
            with open(wiki_file, "r", encoding="utf-8") as f:
                wiki_sentences = json.load(f)
            
            # Add Wikipedia sentences
            for sent in wiki_sentences:
                all_texts.append({
                    "id": sent.get("id", f"wiki_{len(all_texts)}"),
                    "text": sent["text"],
                    "content": sent["text"],
                    "source": "wikipedia",
                    "article": sent.get("article", ""),
                    "modality": "text"
                })
            
            print(f"  Loaded {len(wiki_sentences)} Wikipedia sentences")
        else:
            print(f"\nWarning: Wikipedia data not found at {wiki_file}")
            print("  Run: python scripts/download_wikipedia.py")
    
    if not all_texts:
        print("No text data loaded.")
        return False
    
    print(f"\nTotal texts to index: {len(all_texts)}")
    
    if max_items:
        all_texts = all_texts[:max_items]
        print(f"  Limited to {max_items} items")
    
    # Initialize CLIP embedder
    print("\nInitializing CLIP embedder (ViT-B/32) for text...")
    embedder = ImageEmbedder(model_name="ViT-B/32")
    
    # Generate embeddings
    print(f"\nGenerating CLIP text embeddings (batch_size={batch_size})...")
    start_time = time.time()
    
    all_embeddings = []
    all_metadata = []
    
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        try:
            embeddings = embedder.encode_text(texts)
            
            if embeddings is not None:
                all_embeddings.append(embeddings)
                all_metadata.extend(batch)
        except Exception as e:
            print(f"\n  Warning: Failed to encode batch {i}: {e}")
            continue
        
        processed = min(i + batch_size, len(all_texts))
        print(f"  Processed {processed}/{len(all_texts)} texts...", end="\r")
    
    print()
    
    if not all_embeddings:
        print("No embeddings generated!")
        return False
    
    embeddings_array = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    elapsed = time.time() - start_time
    print(f"Embedding time: {elapsed:.1f}s ({len(all_texts)/elapsed:.1f} items/s)")
    
    # Build index
    print("\nBuilding FAISS index...")
    output_dir = Path(output_dir)
    _save_faiss_index(
        embeddings_array,
        all_metadata,
        output_dir / "text.index",
        output_dir / "text_metadata.json",
    )
    
    # Print source breakdown
    sources = {}
    for m in all_metadata:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\nText sources: {sources}")
    
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
    
    # Build text index with Wikipedia sentences
    python scripts/build_indices.py --modality text --include-wikipedia
    
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
        default=Path("storage/data/raw/flickr8k"),
        help="Data directory (default: data/raw/flickr8k)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("storage/data/indices"),
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
    
    parser.add_argument(
        "--include-wikipedia", "-w",
        action="store_true",
        help="Include Wikipedia sentences in text index"
    )
    
    parser.add_argument(
        "--wikipedia-dir",
        type=Path,
        default=Path("storage/data/raw/wikipedia"),
        help="Wikipedia data directory"
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
        if not build_text_index(
            args.data_dir,
            args.output_dir,
            args.batch_size,
            args.max_items,
            include_wikipedia=args.include_wikipedia,
            wikipedia_dir=args.wikipedia_dir,
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

#!/usr/bin/env python3
"""
Build FAISS Index from Flickr30K Dataset

Generates CLIP embeddings for Flickr30K images and builds/merges FAISS indices.
This expands the image corpus from ~6K (Flickr8K) to ~36K images for better
semantic coverage and encoding diversity.

Requirements:
    - Flickr30K downloaded to data/raw/flickr30k/ (run download_flickr30k.py first)
    - CLIP model (pip install git+https://github.com/openai/CLIP.git)
    - FAISS (pip install faiss-cpu or faiss-gpu)
    - ~8GB RAM, GPU recommended for faster embedding generation

Usage:
    # Build index with GPU (recommended, ~30 min)
    python scripts/build_flickr30k_index.py --gpu
    
    # Build index with CPU only (~2-3 hours)
    python scripts/build_flickr30k_index.py
    
    # Quick test with limited images
    python scripts/build_flickr30k_index.py --max-images 1000 --gpu
    
    # Merge with existing index (default behavior)
    python scripts/build_flickr30k_index.py --gpu --merge
    
    # Replace existing index entirely
    python scripts/build_flickr30k_index.py --gpu --no-merge

Output:
    - data/indices/image.index (FAISS index)
    - data/indices/image_metadata.json (metadata for each image)
"""

import os
import sys
import argparse
import json
import time
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def find_flickr30k_files(data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate Flickr30K images and captions files.
    
    The Kaggle dataset structure can vary, so we search common locations.
    
    Args:
        data_dir: Base directory (e.g., data/raw/flickr30k)
        
    Returns:
        Tuple of (images_dir, captions_file) or (None, None) if not found
    """
    print(f"\nSearching for Flickr30K files in {data_dir}...")
    
    # Possible image directory locations
    image_dirs = [
        data_dir / "images",
        data_dir / "flickr30k_images" / "flickr30k_images",
        data_dir / "flickr30k_images",
        data_dir / "flickr-image-dataset" / "flickr30k_images" / "flickr30k_images",
        data_dir,  # Images might be directly in data_dir
    ]
    
    # Possible caption file locations
    caption_files = [
        data_dir / "results.csv",
        data_dir / "flickr30k_images" / "results.csv",
        data_dir / "flickr-image-dataset" / "results.csv",
        data_dir / "captions.csv",
    ]
    
    found_images_dir = None
    found_captions_file = None
    
    # Find images directory
    for img_dir in image_dirs:
        if img_dir.exists() and img_dir.is_dir():
            jpg_files = list(img_dir.glob("*.jpg"))[:5]  # Quick check
            if jpg_files:
                found_images_dir = img_dir
                print(f"  Found images: {img_dir} ({len(list(img_dir.glob('*.jpg')))} .jpg files)")
                break
    
    # Find captions file
    for cap_file in caption_files:
        if cap_file.exists():
            found_captions_file = cap_file
            print(f"  Found captions: {cap_file}")
            break
    
    return found_images_dir, found_captions_file


def load_flickr30k_captions(captions_file: Path) -> Dict[str, List[str]]:
    """
    Load captions from Flickr30K results.csv file.
    
    The file format is: image_name|comment_number|comment
    Example: 1000092795.jpg|0|Two young guys with shaggy hair look at their hands.
    
    Args:
        captions_file: Path to results.csv
        
    Returns:
        Dict mapping image filename to list of captions
    """
    print(f"\nLoading captions from {captions_file}...")
    
    captions = defaultdict(list)
    
    try:
        with open(captions_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Try to detect format
            first_line = f.readline().strip()
            f.seek(0)
            
            # Check if it has a header
            if 'image' in first_line.lower() or 'comment' in first_line.lower():
                # CSV with header
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    # Handle different column names
                    image_name = row.get('image_name') or row.get('image') or row.get('filename')
                    caption = row.get('comment') or row.get('caption') or row.get('text')
                    if image_name and caption:
                        # Normalize filename
                        if not image_name.endswith('.jpg'):
                            image_name = image_name + '.jpg'
                        captions[image_name].append(caption.strip())
            else:
                # Simple pipe-delimited format without header
                f.seek(0)
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        image_name = parts[0]
                        # parts[1] is comment number (0-4)
                        caption = parts[2] if len(parts) == 3 else '|'.join(parts[2:])
                        if not image_name.endswith('.jpg'):
                            image_name = image_name + '.jpg'
                        captions[image_name].append(caption.strip())
        
        print(f"  Loaded captions for {len(captions)} images")
        
        # Show sample
        sample_key = list(captions.keys())[0]
        print(f"  Sample ({sample_key}): {captions[sample_key][0][:60]}...")
        
    except Exception as e:
        print(f"  Error loading captions: {e}")
        return {}
    
    return dict(captions)


def build_flickr30k_index(
    images_dir: Path,
    captions: Dict[str, List[str]],
    output_dir: Path,
    batch_size: int = 32,
    max_images: Optional[int] = None,
    use_gpu: bool = True,
    merge_existing: bool = True
) -> bool:
    """
    Build FAISS index for Flickr30K images using CLIP embeddings.
    
    Args:
        images_dir: Directory containing Flickr30K images
        captions: Dict mapping image filename to captions list
        output_dir: Directory to save index files
        batch_size: Batch size for embedding generation
        max_images: Maximum images to process (for testing)
        use_gpu: Whether to use GPU for embedding generation
        merge_existing: Whether to merge with existing index
        
    Returns:
        True if successful
    """
    import torch
    import faiss
    
    print("\n" + "=" * 70)
    print("Building Flickr30K Image Index")
    print("=" * 70)
    
    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if use_gpu and not torch.cuda.is_available():
        print("\nWarning: GPU requested but CUDA not available. Using CPU.")
    print(f"\nDevice: {device}")
    
    # Initialize CLIP embedder
    print("\nLoading CLIP model (ViT-B/32)...")
    from src.corpus.embedders.image_embedder import ImageEmbedder
    embedder = ImageEmbedder(model_name="ViT-B/32", device=device)
    embedder.load_model()
    
    # Collect image files
    print(f"\nScanning images in {images_dir}...")
    image_files = sorted(images_dir.glob("*.jpg"))
    
    if not image_files:
        # Try .png as well
        image_files = sorted(images_dir.glob("*.png"))
    
    if not image_files:
        print("No image files found!")
        return False
    
    print(f"  Found {len(image_files)} images")
    
    if max_images:
        image_files = image_files[:max_images]
        print(f"  Limited to {max_images} images for testing")
    
    # Generate embeddings
    print(f"\nGenerating CLIP embeddings (batch_size={batch_size})...")
    print("  This may take 30-60 minutes on GPU, 2-3 hours on CPU...")
    
    start_time = time.time()
    all_embeddings = []
    all_metadata = []
    failed_images = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [str(f) for f in batch_files]
        
        try:
            # Encode batch
            embeddings = embedder.encode_images(batch_paths, show_progress=False)
            
            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.append(embeddings)
                
                # Build metadata for each image
                for img_path in batch_files:
                    filename = img_path.name
                    img_captions = captions.get(filename, [])
                    
                    # Create unique ID
                    img_id = f"flickr30k_{img_path.stem}"
                    
                    all_metadata.append({
                        "id": img_id,
                        "content": str(img_path.absolute()),
                        "filename": filename,
                        "caption": img_captions[0] if img_captions else "",
                        "captions": img_captions,
                        "source": "flickr30k",
                        "modality": "image"
                    })
        
        except Exception as e:
            failed_images += len(batch_files)
            if failed_images < 10:  # Only show first few errors
                print(f"\n  Warning: Failed to encode batch {i}: {e}")
            continue
        
        # Progress update
        processed = min(i + batch_size, len(image_files))
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(image_files) - processed) / rate if rate > 0 else 0
        
        print(f"\r  Processed {processed}/{len(image_files)} images "
              f"({rate:.1f} img/s, ETA: {eta/60:.1f} min)   ", end="")
    
    print()  # Newline after progress
    
    if not all_embeddings:
        print("\nNo embeddings generated!")
        return False
    
    # Stack embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"\nNew embeddings: {embeddings_array.shape}")
    print(f"  Failed images: {failed_images}")
    
    elapsed = time.time() - start_time
    print(f"  Embedding time: {elapsed/60:.1f} minutes ({len(image_files)/elapsed:.1f} img/s)")
    
    # Load existing index if merging
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "image.index"
    metadata_path = output_dir / "image_metadata.json"
    
    existing_embeddings = None
    existing_metadata = []
    
    if merge_existing and index_path.exists() and metadata_path.exists():
        print(f"\nLoading existing index for merge...")
        try:
            existing_index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            
            # Extract embeddings from existing index
            n_existing = existing_index.ntotal
            if n_existing > 0:
                # Reconstruct embeddings from index
                existing_embeddings = np.zeros((n_existing, embeddings_array.shape[1]), dtype=np.float32)
                for i in range(n_existing):
                    existing_embeddings[i] = existing_index.reconstruct(i)
                
                print(f"  Loaded {n_existing} existing items from index")
        except Exception as e:
            print(f"  Warning: Could not load existing index: {e}")
            print("  Building new index from scratch...")
            existing_embeddings = None
            existing_metadata = []
    
    # Merge embeddings and metadata
    if existing_embeddings is not None:
        print(f"\nMerging indices...")
        print(f"  Existing: {existing_embeddings.shape[0]} items")
        print(f"  New: {embeddings_array.shape[0]} items")
        
        # Check for duplicates by ID
        existing_ids = set(m.get("id", "") for m in existing_metadata)
        
        # Filter out duplicates from new data
        unique_indices = []
        for i, meta in enumerate(all_metadata):
            if meta["id"] not in existing_ids:
                unique_indices.append(i)
        
        if len(unique_indices) < len(all_metadata):
            print(f"  Skipping {len(all_metadata) - len(unique_indices)} duplicate items")
            embeddings_array = embeddings_array[unique_indices]
            all_metadata = [all_metadata[i] for i in unique_indices]
        
        # Combine
        final_embeddings = np.vstack([existing_embeddings, embeddings_array])
        final_metadata = existing_metadata + all_metadata
        
        print(f"  Combined: {final_embeddings.shape[0]} items")
    else:
        final_embeddings = embeddings_array
        final_metadata = all_metadata
    
    # Build FAISS index
    print(f"\nBuilding FAISS index...")
    dimension = final_embeddings.shape[1]
    
    # Use IndexFlatIP for cosine similarity (embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(final_embeddings)
    
    # Add to index
    index.add(final_embeddings)
    
    print(f"  Index size: {index.ntotal} vectors ({dimension} dimensions)")
    
    # Backup existing files
    if index_path.exists():
        backup_path = output_dir / "image.index.backup"
        print(f"\n  Backing up existing index to {backup_path}")
        os.replace(str(index_path), str(backup_path))
    
    if metadata_path.exists():
        backup_meta = output_dir / "image_metadata.json.backup"
        print(f"  Backing up existing metadata to {backup_meta}")
        os.replace(str(metadata_path), str(backup_meta))
    
    # Save new index
    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, str(index_path))
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Index Build Complete!")
    print("=" * 70)
    
    # Count sources
    sources = defaultdict(int)
    for meta in final_metadata:
        sources[meta.get("source", "unknown")] += 1
    
    print(f"\nTotal images indexed: {index.ntotal}")
    print("\nBreakdown by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
    
    print(f"\nIndex files:")
    print(f"  {index_path}")
    print(f"  {metadata_path}")
    
    print("\nNext steps:")
    print("  1. Run benchmarks to measure improvement:")
    print("     python -m src.cli.main benchmark --quick --markdown")
    print("  2. Test encoding:")
    print("     python -m src.cli.main encode 'your test message'")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from Flickr30K images using CLIP embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
    1. Download Flickr30K first: python scripts/download_flickr30k.py
    2. GPU recommended for faster processing (~30 min vs 2-3 hours on CPU)

Examples:
    # Build with GPU (recommended)
    python scripts/build_flickr30k_index.py --gpu
    
    # Quick test with 1000 images
    python scripts/build_flickr30k_index.py --gpu --max-images 1000
    
    # Build CPU-only (slower but works without CUDA)
    python scripts/build_flickr30k_index.py
    
    # Replace existing index instead of merging
    python scripts/build_flickr30k_index.py --gpu --no-merge
        """
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data/raw/flickr30k"),
        help="Flickr30K data directory (default: data/raw/flickr30k)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/indices"),
        help="Output directory for index files (default: data/indices)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32, use 64+ for GPU)"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process (for testing)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for embedding generation (recommended)"
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge with existing index (default: True)"
    )
    
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace existing index instead of merging"
    )
    
    args = parser.parse_args()
    
    # Handle merge flag
    merge_existing = not args.no_merge
    
    # Find Flickr30K files
    images_dir, captions_file = find_flickr30k_files(args.data_dir)
    
    if images_dir is None:
        print(f"\nError: Could not find Flickr30K images in {args.data_dir}")
        print("\nPlease run the download script first:")
        print("  python scripts/download_flickr30k.py")
        sys.exit(1)
    
    # Load captions
    captions = {}
    if captions_file:
        captions = load_flickr30k_captions(captions_file)
    else:
        print("\nWarning: Captions file not found. Images will have empty captions.")
    
    # Build index
    success = build_flickr30k_index(
        images_dir=images_dir,
        captions=captions,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_images=args.max_images,
        use_gpu=args.gpu,
        merge_existing=merge_existing
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

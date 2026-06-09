#!/usr/bin/env python3
"""
Download FAISS indices and embeddings from HuggingFace.

Downloads pre-built indices from khrshtt/images dataset and sets up
the proper directory structure for DCASS.

Usage:
    python scripts/data/download_hf_indices.py
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# HuggingFace dataset info
HF_REPO = "khrshtt/images"
HF_FILES = {
    "faiss_image.index": "image.index",  # Rename to match expected format
    "image_embeddings.npy": "image_embeddings.npy",
    "caption_embeddings.npy": "caption_embeddings.npy",
    "caption_map.csv": "caption_map.csv",
    "image_ids.txt": "image_ids.txt",
}

# Output directories
INDICES_DIR = PROJECT_ROOT / "storage" / "data" / "indices"
EMBEDDINGS_DIR = PROJECT_ROOT / "storage" / "data" / "embeddings"


def download_from_hf():
    """Download files from HuggingFace using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    # Create output directories
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading from HuggingFace: {HF_REPO}")
    print(f"Indices directory: {INDICES_DIR}")
    print(f"Embeddings directory: {EMBEDDINGS_DIR}")
    print("-" * 60)
    
    downloaded = []
    
    for hf_filename, local_filename in HF_FILES.items():
        print(f"\nDownloading: {hf_filename} -> {local_filename}")
        
        try:
            # Download from HuggingFace
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_filename,
                repo_type="dataset",
            )
            
            # Determine destination based on file type
            if local_filename.endswith(".index"):
                dest_path = INDICES_DIR / local_filename
            else:
                dest_path = EMBEDDINGS_DIR / local_filename
            
            # Copy to destination (hf_hub caches files)
            shutil.copy2(downloaded_path, dest_path)
            print(f"  Saved to: {dest_path}")
            downloaded.append(local_filename)
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return downloaded


def create_image_metadata():
    """
    Create image_metadata.json from caption_map.csv and image_ids.txt.
    
    The unified_index.py expects:
    - {modality}.index (FAISS index)
    - {modality}_metadata.json (metadata with id, content/path fields)
    """
    print("\n" + "-" * 60)
    print("Creating image_metadata.json...")
    
    caption_map_path = EMBEDDINGS_DIR / "caption_map.csv"
    image_ids_path = EMBEDDINGS_DIR / "image_ids.txt"
    metadata_path = INDICES_DIR / "image_metadata.json"
    
    metadata = []
    
    # First, try to load image IDs
    if image_ids_path.exists():
        with open(image_ids_path, "r", encoding="utf-8") as f:
            image_ids = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(image_ids)} image IDs")
    else:
        image_ids = []
        print("  WARNING: image_ids.txt not found")
    
    # Load captions from caption_map.csv
    captions_by_image = {}
    if caption_map_path.exists():
        import csv
        with open(caption_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id", "")
                caption = row.get("caption", "")
                if img_id:
                    if img_id not in captions_by_image:
                        captions_by_image[img_id] = []
                    captions_by_image[img_id].append(caption)
        print(f"  Loaded captions for {len(captions_by_image)} images")
    
    # Create metadata for each image
    # If we have image_ids, use that order (matches FAISS index order)
    if image_ids:
        for idx, img_id in enumerate(image_ids):
            captions = captions_by_image.get(img_id, [])
            metadata.append({
                "id": img_id,
                "index": idx,
                "path": f"storage/data/raw/flickr8k/images/{img_id}.jpg",
                "captions": captions,
                "content": captions[0] if captions else img_id,  # Primary caption or ID
            })
    else:
        # Fallback: use unique image IDs from caption map
        for idx, img_id in enumerate(sorted(captions_by_image.keys())):
            captions = captions_by_image[img_id]
            metadata.append({
                "id": img_id,
                "index": idx,
                "path": f"storage/data/raw/flickr8k/images/{img_id}.jpg",
                "captions": captions,
                "content": captions[0] if captions else img_id,
            })
    
    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Created metadata for {len(metadata)} images")
    print(f"  Saved to: {metadata_path}")
    
    return metadata_path


def verify_setup():
    """Verify the index setup is correct."""
    print("\n" + "-" * 60)
    print("Verifying setup...")
    
    required_files = [
        INDICES_DIR / "image.index",
        INDICES_DIR / "image_metadata.json",
    ]
    
    all_ok = True
    for f in required_files:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  OK: {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"  MISSING: {f}")
            all_ok = False
    
    # Try to load the FAISS index
    if all_ok:
        try:
            import faiss
            index = faiss.read_index(str(INDICES_DIR / "image.index"))
            print(f"\n  FAISS index loaded: {index.ntotal} vectors")
            
            with open(INDICES_DIR / "image_metadata.json", "r") as f:
                meta = json.load(f)
            print(f"  Metadata entries: {len(meta)}")
            
            if index.ntotal == len(meta):
                print("\n  SUCCESS: Index and metadata match!")
            else:
                print(f"\n  WARNING: Mismatch! Index has {index.ntotal}, metadata has {len(meta)}")
                
        except ImportError:
            print("\n  Note: faiss not installed, skipping verification")
        except Exception as e:
            print(f"\n  ERROR loading index: {e}")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("DCASS HuggingFace Index Downloader")
    print("=" * 60)
    
    # Step 1: Download files
    downloaded = download_from_hf()
    
    if not downloaded:
        print("\nERROR: No files were downloaded!")
        return 1
    
    # Step 2: Create metadata JSON
    create_image_metadata()
    
    # Step 3: Verify setup
    if verify_setup():
        print("\n" + "=" * 60)
        print("Setup complete! You can now run the DCASS system.")
        print("=" * 60)
        return 0
    else:
        print("\nSetup completed with warnings. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

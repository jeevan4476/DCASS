#!/usr/bin/env python3
"""
Download Flickr30K Dataset for DCASS

Downloads the Flickr30K dataset to expand the image corpus from 6K to 30K+ images.

The Flickr30K dataset contains:
- 31,783 images
- 5 captions per image (158,915 total captions)

This significantly improves semantic coverage and reduces the "same items 
matching everything" problem observed in benchmarks.

Prerequisites:
    1. Install Kaggle CLI: pip install kaggle
    2. Setup Kaggle API credentials:
       - Go to https://www.kaggle.com/settings
       - Click "Create New Token" to download kaggle.json
       - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
       - Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)

Usage:
    # Download full dataset (~4GB)
    python scripts/download_flickr30k.py
    
    # Custom output directory
    python scripts/download_flickr30k.py --output data/raw/flickr30k
    
    # Skip images (captions only, for testing)
    python scripts/download_flickr30k.py --captions-only

Dataset source: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_kaggle_setup() -> bool:
    """
    Check if Kaggle CLI is installed and configured.
    
    Returns:
        True if Kaggle is ready to use
    """
    print("Checking Kaggle setup...")
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("  [OK] Kaggle package installed")
    except ImportError:
        print("  [ERROR] Kaggle package not installed")
        print("  Run: pip install kaggle")
        return False
    
    # Check for credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print(f"  [ERROR] Kaggle credentials not found at {kaggle_json}")
        print("\n  Setup instructions:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token' to download kaggle.json")
        print(f"  4. Move kaggle.json to {kaggle_dir}/")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("  [OK] Kaggle credentials found")
    
    # Test API connection
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("  [OK] Kaggle API authenticated")
        return True
    except Exception as e:
        print(f"  [ERROR] Kaggle API authentication failed: {e}")
        return False


def download_flickr30k(output_dir: Path, captions_only: bool = False) -> bool:
    """
    Download the Flickr30K dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        captions_only: If True, only extract captions (skip images)
        
    Returns:
        True if successful
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("\n" + "=" * 60)
    print("Flickr30K Dataset Downloader")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaggle dataset identifier
    DATASET = "hsankesara/flickr-image-dataset"
    
    print(f"\nDataset: {DATASET}")
    print(f"Output: {output_dir.absolute()}")
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    zip_path = output_dir / "flickr-image-dataset.zip"
    
    if not zip_path.exists():
        print("\n[1/3] Downloading dataset from Kaggle (~4GB)...")
        print("      This may take 10-30 minutes depending on your connection.")
        
        try:
            api.dataset_download_files(
                DATASET,
                path=str(output_dir),
                quiet=False,
                unzip=False  # We'll handle extraction ourselves
            )
            print("      Download complete!")
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            print("\nAlternative: Download manually from Kaggle:")
            print(f"  https://www.kaggle.com/datasets/{DATASET}")
            return False
    else:
        print(f"\n[1/3] Dataset archive already exists: {zip_path}")
        print("      Skipping download.")
    
    # Extract dataset
    print("\n[2/3] Extracting dataset...")
    
    images_dir = output_dir / "images"
    captions_file = output_dir / "results.csv"
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List contents
            names = zf.namelist()
            print(f"      Archive contains {len(names)} items")
            
            # Find the captions file
            caption_files = [n for n in names if n.endswith('.csv') or 'caption' in n.lower()]
            image_folders = [n for n in names if 'flickr30k' in n.lower() or 'images' in n.lower()]
            
            print(f"      Found caption files: {caption_files[:3]}...")
            print(f"      Found image folders: {image_folders[:3]}...")
            
            # Extract captions first (small)
            print("\n      Extracting captions...")
            for name in names:
                if name.endswith('.csv') or name.endswith('.txt'):
                    zf.extract(name, output_dir)
                    print(f"        Extracted: {name}")
            
            # Extract images (large) unless skipped
            if not captions_only:
                print("\n      Extracting images (this may take several minutes)...")
                
                # Count images for progress
                image_files = [n for n in names if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images = len(image_files)
                print(f"      Total images: {total_images}")
                
                # Extract all
                zf.extractall(output_dir)
                print("      Extraction complete!")
            else:
                print("\n      Skipping image extraction (--captions-only)")
        
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        return False
    
    # Organize files
    print("\n[3/3] Organizing files...")
    
    # The Kaggle dataset structure varies, let's find and organize
    # Look for the images folder
    possible_image_dirs = [
        output_dir / "flickr30k_images" / "flickr30k_images",
        output_dir / "flickr30k-images" / "flickr30k-images", 
        output_dir / "flickr30k_images",
        output_dir / "flickr-image-dataset" / "flickr30k_images" / "flickr30k_images",
        output_dir / "flickr-image-dataset" / "flickr30k-images",
    ]
    
    found_images_dir = None
    for pdir in possible_image_dirs:
        if pdir.exists() and pdir.is_dir():
            # Check if it contains images
            jpg_files = list(pdir.glob("*.jpg"))
            if jpg_files:
                found_images_dir = pdir
                print(f"      Found images at: {pdir}")
                break
    
    if found_images_dir and found_images_dir != images_dir:
        if not images_dir.exists():
            print(f"      Moving images to {images_dir}...")
            shutil.move(str(found_images_dir), str(images_dir))
        else:
            print(f"      Images directory already exists at {images_dir}")
    
    # Look for captions file
    possible_caption_files = [
        output_dir / "results.csv",
        output_dir / "flickr30k_images" / "results.csv",
        output_dir / "flickr-image-dataset" / "results.csv",
        output_dir / "captions.txt",
    ]
    
    found_captions = None
    for pcap in possible_caption_files:
        if pcap.exists():
            found_captions = pcap
            print(f"      Found captions at: {pcap}")
            break
    
    if found_captions and found_captions != captions_file:
        if not captions_file.exists():
            print(f"      Moving captions to {captions_file}...")
            shutil.copy(str(found_captions), str(captions_file))
    
    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    # Count images
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"\nImages: {len(image_files)} files in {images_dir}")
    else:
        # Try to find images anywhere in output_dir
        all_images = list(output_dir.rglob("*.jpg"))
        print(f"\nImages: {len(all_images)} .jpg files found in {output_dir}")
        if all_images:
            print(f"        Sample: {all_images[0]}")
    
    # Check captions
    if captions_file.exists():
        with open(captions_file, 'r', encoding='utf-8', errors='ignore') as f:
            num_lines = sum(1 for _ in f)
        print(f"Captions: {num_lines} lines in {captions_file}")
    else:
        # Look for any CSV
        csv_files = list(output_dir.rglob("*.csv"))
        print(f"Captions: Found {len(csv_files)} CSV files")
        for cf in csv_files[:3]:
            print(f"          {cf}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python scripts/build_flickr30k_index.py")
    print("     This will generate CLIP embeddings and build the FAISS index")
    print("\n  2. The script will automatically merge with existing indices")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Flickr30K dataset from Kaggle for DCASS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
    1. pip install kaggle
    2. Setup Kaggle API key:
       - Go to https://www.kaggle.com/settings
       - Click "Create New Token" 
       - Place kaggle.json in ~/.kaggle/
       - chmod 600 ~/.kaggle/kaggle.json

Examples:
    python scripts/download_flickr30k.py
    python scripts/download_flickr30k.py --output data/raw/flickr30k
    python scripts/download_flickr30k.py --captions-only
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/raw/flickr30k"),
        help="Output directory (default: data/raw/flickr30k)"
    )
    
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Only download/extract captions, skip images (for testing)"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true", 
        help="Skip Kaggle setup check"
    )
    
    args = parser.parse_args()
    
    # Check Kaggle setup
    if not args.skip_check:
        if not check_kaggle_setup():
            print("\nPlease setup Kaggle CLI first (see instructions above)")
            sys.exit(1)
    
    # Download
    success = download_flickr30k(args.output, captions_only=args.captions_only)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

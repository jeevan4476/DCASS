#!/usr/bin/env python3
"""
Download Flickr8k Dataset

Downloads the Flickr8k dataset for DCASS image corpus.

The Flickr8k dataset contains:
- 8,000 images
- 5 captions per image (40,000 total captions)

This is an excellent dataset for semantic steganography because:
- Diverse everyday scenes
- High-quality human-written captions
- Good semantic coverage

Usage:
    python scripts/download_flickr8k.py
    python scripts/download_flickr8k.py --output data/raw/flickr8k
    
Note:
    The official Flickr8k requires signing a form. This script
    downloads from a commonly used mirror. For official access,
    visit: https://forms.illinois.edu/sec/1713398
"""

import os
import sys
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest: Destination path
        desc: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install requests tqdm")
        import requests
        from tqdm import tqdm
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """
    Extract a zip or tar.gz archive.
    
    Args:
        archive_path: Path to archive file
        dest_dir: Destination directory
        
    Returns:
        True if successful
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                print(f"Extracting {archive_path.name}...")
                zf.extractall(dest_dir)
        elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tf:
                print(f"Extracting {archive_path.name}...")
                tf.extractall(dest_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_flickr8k(output_dir: Path, skip_images: bool = False) -> bool:
    """
    Download the Flickr8k dataset.
    
    Args:
        output_dir: Directory to save the dataset
        skip_images: If True, only download captions (faster for testing)
        
    Returns:
        True if successful
    """
    print("=" * 60)
    print("Flickr8k Dataset Downloader")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for Flickr8k (using common mirror)
    # Note: These URLs may change - update as needed
    IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    CAPTIONS_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Alternative: Kaggle dataset (requires kaggle API)
    # kaggle datasets download -d adityajn105/flickr8k
    
    success = True
    
    # Download images
    if not skip_images:
        images_zip = output_dir / "Flickr8k_Dataset.zip"
        if not images_zip.exists():
            print("\n[1/2] Downloading images (~1GB)...")
            if not download_file(IMAGES_URL, images_zip, "Images"):
                print("Failed to download images.")
                print("\nAlternative: Download manually from Kaggle:")
                print("  kaggle datasets download -d adityajn105/flickr8k")
                success = False
        else:
            print("\n[1/2] Images archive already exists, skipping download.")
        
        # Extract images
        images_dir = output_dir / "images"
        if images_zip.exists() and not images_dir.exists():
            if not extract_archive(images_zip, output_dir):
                success = False
            else:
                # Rename extracted folder if needed
                extracted = output_dir / "Flicker8k_Dataset"
                if extracted.exists() and not images_dir.exists():
                    extracted.rename(images_dir)
    else:
        print("\n[1/2] Skipping images (--skip-images flag)")
    
    # Download captions
    captions_zip = output_dir / "Flickr8k_text.zip"
    if not captions_zip.exists():
        print("\n[2/2] Downloading captions...")
        if not download_file(CAPTIONS_URL, captions_zip, "Captions"):
            print("Failed to download captions.")
            success = False
    else:
        print("\n[2/2] Captions archive already exists, skipping download.")
    
    # Extract captions
    captions_dir = output_dir / "text"
    if captions_zip.exists() and not captions_dir.exists():
        if not extract_archive(captions_zip, output_dir):
            success = False
        else:
            # Rename if needed
            extracted = output_dir / "Flickr8k_text" 
            if extracted.exists():
                extracted.rename(captions_dir)
    
    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    images_path = output_dir / "images"
    captions_path = output_dir / "text"
    
    if images_path.exists():
        num_images = len(list(images_path.glob("*.jpg")))
        print(f"Images: {num_images} files in {images_path}")
    else:
        print(f"Images: NOT FOUND at {images_path}")
    
    if captions_path.exists():
        token_file = captions_path / "Flickr8k.token.txt"
        if token_file.exists():
            with open(token_file) as f:
                num_captions = sum(1 for _ in f)
            print(f"Captions: {num_captions} lines in {token_file}")
        else:
            print(f"Captions: token file not found at {token_file}")
    else:
        print(f"Captions: NOT FOUND at {captions_path}")
    
    print("\n" + "=" * 60)
    if success:
        print("Download complete!")
        print(f"\nDataset location: {output_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Update config/default.yaml with the dataset path")
        print("  2. Run: python scripts/build_indices.py")
    else:
        print("Download completed with errors. Check messages above.")
    print("=" * 60)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download Flickr8k dataset for DCASS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_flickr8k.py
    python scripts/download_flickr8k.py --output data/raw/flickr8k
    python scripts/download_flickr8k.py --skip-images  # Captions only (faster)
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("storage/data/raw/flickr8k"),
        help="Output directory (default: data/raw/flickr8k)"
    )
    
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip downloading images (captions only, for testing)"
    )
    
    args = parser.parse_args()
    
    success = download_flickr8k(args.output, skip_images=args.skip_images)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

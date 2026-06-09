#!/usr/bin/env python3
"""
Download Flickr30k Dataset from HuggingFace

Downloads the Flickr30k dataset for expanding the DCASS image corpus.

The Flickr30k dataset contains:
- ~31,000 images
- 5 captions per image (~155,000 total captions)

This expands the image corpus from ~8K (Flickr8k) to ~39K images.

Usage:
    python scripts/data/download_flickr30k.py
    python scripts/data/download_flickr30k.py --output storage/data/raw/flickr30k

Requirements:
    pip install huggingface_hub pillow tqdm
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_flickr30k(
    output_dir: Path,
    skip_existing: bool = True
) -> bool:
    """
    Download Flickr30k dataset from HuggingFace using huggingface_hub.
    
    Args:
        output_dir: Directory to save images and captions
        skip_existing: Skip if images already exist
        
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install huggingface_hub pillow tqdm")
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    
    print("=" * 60)
    print("Flickr30k Dataset Downloader (HuggingFace)")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_images = list(images_dir.glob("*.jpg"))
    if skip_existing and len(existing_images) > 30000:
        print(f"\nFound {len(existing_images)} existing images in {images_dir}")
        print("Skipping download. Use --force to re-download.")
        return True
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Images directory: {images_dir}")
    
    # Download files from HuggingFace
    repo_id = "nlphuji/flickr30k"
    
    # Step 1: Download captions CSV
    print("\n[1/3] Downloading captions file...")
    try:
        captions_path = hf_hub_download(
            repo_id=repo_id,
            filename="flickr_annotations_30k.csv",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded: {captions_path}")
    except Exception as e:
        print(f"  Error downloading captions: {e}")
        return False
    
    # Step 2: Download images zip
    print("\n[2/3] Downloading images zip (~4GB)...")
    print("  This may take 10-30 minutes depending on your connection...")
    try:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="flickr30k-images.zip",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded: {zip_path}")
    except Exception as e:
        print(f"  Error downloading images: {e}")
        return False
    
    # Step 3: Extract images
    print("\n[3/3] Extracting images...")
    zip_file = output_dir / "flickr30k-images.zip"
    
    if zip_file.exists():
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Get list of files
                file_list = [f for f in zf.namelist() if f.endswith('.jpg')]
                print(f"  Found {len(file_list)} images in archive")
                
                # Extract with progress bar
                for filename in tqdm(file_list, desc="  Extracting"):
                    # Extract to images directory, flattening any subdirectories
                    basename = os.path.basename(filename)
                    if basename:  # Skip directories
                        source = zf.read(filename)
                        dest_path = images_dir / basename
                        with open(dest_path, 'wb') as f:
                            f.write(source)
            
            print("  Extraction complete!")
        except Exception as e:
            print(f"  Error extracting: {e}")
            return False
    else:
        print(f"  Error: Zip file not found at {zip_file}")
        return False
    
    # Convert captions to results.csv format for compatibility with build script
    print("\nConverting captions to results.csv format...")
    convert_captions_format(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    # Verify
    final_images = list(images_dir.glob("*.jpg"))
    print(f"\nImages: {len(final_images)} files in {images_dir}")
    
    results_csv = output_dir / "results.csv"
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            caption_lines = sum(1 for _ in f) - 1  # -1 for header
        print(f"Captions: {caption_lines} entries in {results_csv}")
    
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/                    ({len(final_images)} .jpg files)")
    print(f"  ├── flickr_annotations_30k.csv (original captions)")
    print(f"  └── results.csv                (formatted for build script)")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  Build the FAISS index:")
    print("    python scripts/data/build_flickr30k_index.py --gpu --merge")
    print("=" * 60)
    
    return len(final_images) > 0


def convert_captions_format(output_dir: Path) -> None:
    """
    Convert flickr_annotations_30k.csv to results.csv format.
    
    The annotation CSV has columns: raw, sentids, split, filename, img_id
    where 'raw' contains a JSON-like list of 5 captions.
    
    We convert to: image_name|comment_number|comment
    """
    import csv
    import json
    import ast
    
    annotations_file = output_dir / "flickr_annotations_30k.csv"
    results_file = output_dir / "results.csv"
    
    if not annotations_file.exists():
        print(f"  Warning: {annotations_file} not found")
        return
    
    caption_count = 0
    
    with open(annotations_file, 'r', encoding='utf-8') as infile, \
         open(results_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        outfile.write("image_name|comment_number|comment\n")
        
        for row in reader:
            filename = row.get('filename', '').strip()
            if not filename:
                continue
            
            # Ensure .jpg extension
            if not filename.lower().endswith('.jpg'):
                filename = filename + '.jpg'
            
            # Parse captions from 'raw' field (JSON-like list)
            raw_captions = row.get('raw', '[]')
            try:
                captions = json.loads(raw_captions)
            except json.JSONDecodeError:
                try:
                    captions = ast.literal_eval(raw_captions)
                except:
                    captions = []
            
            # Write each caption
            for i, caption in enumerate(captions):
                if caption:
                    # Clean caption
                    caption = str(caption).replace('\n', ' ').replace('|', ' ').strip()
                    outfile.write(f"{filename}|{i}|{caption}\n")
                    caption_count += 1
    
    print(f"  Converted {caption_count} captions to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Flickr30k dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download full dataset (~31k images)
    python scripts/data/download_flickr30k.py
    
    # Download to custom directory
    python scripts/data/download_flickr30k.py --output storage/data/raw/flickr30k
    
    # Force re-download even if images exist
    python scripts/data/download_flickr30k.py --force
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("storage/data/raw/flickr30k"),
        help="Output directory (default: storage/data/raw/flickr30k)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if images exist"
    )
    
    args = parser.parse_args()
    
    success = download_flickr30k(
        output_dir=args.output,
        skip_existing=not args.force
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1: Download audio dataset from HuggingFace.

Downloads the libretta-tts dataset which contains ~10k audio clips
with text transcriptions - perfect for semantic audio search.

Usage:
    python scripts/audio_step1_download.py

Time: ~5-10 minutes depending on internet speed
Size: ~2-3 GB
"""

import os
from pathlib import Path

# Check dependencies
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed.")
    print("Run: pip install datasets")
    exit(1)

# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "storage" / "data" / "audio"

# Dataset from HuggingFace
DATASET_NAME = "GrigoriiA/libretta-tts-merged-dataset-audio-L10k"

# Where to cache the dataset
CACHE_DIR = DATA_DIR / "cache"
# ============================================

def main():
    print("=" * 60)
    print(" Audio Dataset Download")
    print("=" * 60)
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"\nThis will download ~2-3 GB of audio data.")
    print("Press Ctrl+C to cancel.\n")
    
    # Download dataset
    print("Downloading dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True
    )
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Features: {dataset.features}")
    
    # Save dataset info
    info_file = DATA_DIR / "dataset_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Samples: {len(dataset)}\n")
        f.write(f"Features: {dataset.features}\n")
        f.write(f"Cache: {CACHE_DIR}\n")
    
    print(f"\nDataset info saved to: {info_file}")
    print("\nNext step: Run audio_step2_build_index.py")


if __name__ == "__main__":
    main()

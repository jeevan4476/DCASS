#!/usr/bin/env python3
"""
Fix audio metadata to include transcriptions.

The original build script looked for 'text' field but the libretta dataset
uses 'transcription' and 'transcription_normalised'. This script rebuilds
the metadata with correct field names.
"""

import json
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
CACHE_DIR = BASE_DIR / "storage" / "data" / "audio" / "cache"
METADATA_PATH = BASE_DIR / "storage" / "data" / "indices" / "audio_metadata.json"
DATASET_NAME = "GrigoriiA/libretta-tts-merged-dataset-audio-L10k"
SAMPLE_RATE = 48000

def main():
    print("Loading dataset...")
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=str(CACHE_DIR)
    )
    
    # Cast audio column (for duration calculation)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    print(f"Dataset features: {dataset.features}")
    print(f"Sample example keys: {list(dataset[0].keys())}")
    print(f"Sample transcription: {dataset[0].get('transcription_normalised', 'N/A')[:100]}")
    
    print(f"\nRebuilding metadata for {len(dataset)} items...")
    
    metadata = []
    for i, example in enumerate(tqdm(dataset, desc="Building metadata")):
        # Get transcription (prefer normalised version)
        text = (
            example.get("transcription_normalised", "") or 
            example.get("transcription", "") or 
            ""
        )
        
        # Calculate duration
        audio_data = example.get("audio", {})
        if isinstance(audio_data, dict) and "array" in audio_data:
            duration = len(audio_data["array"]) / SAMPLE_RATE
        else:
            duration = 0.0
        
        metadata.append({
            "id": f"audio_{i:06d}",
            "text": text,
            "content": text,
            "source": "libretta",
            "modality": "audio",
            "duration": duration
        })
    
    # Save metadata
    print(f"\nSaving to {METADATA_PATH}...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Show samples
    print("\nSample entries:")
    for m in metadata[:3]:
        text_preview = m['text'][:60] + "..." if len(m['text']) > 60 else m['text']
        print(f"  {m['id']}: \"{text_preview}\"")
    
    # Count non-empty
    non_empty = sum(1 for m in metadata if m['text'])
    print(f"\nItems with transcriptions: {non_empty}/{len(metadata)}")
    print("\nDone!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 2: Build audio FAISS index using CLAP embeddings.

This script:
1. Loads the downloaded audio dataset
2. Encodes audio using CLAP model (512-dim, compatible with CLIP)
3. Creates audio.index and audio_metadata.json in data/indices/

IMPORTANT: CLAP produces 512-dimensional embeddings, same as CLIP,
allowing cross-modal search between text, images, and audio.

Usage:
    python scripts/audio_step2_build_index.py

Time: ~15-20 minutes on CPU, ~5 minutes on GPU
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

# Check dependencies
missing = []
try:
    import torch
except ImportError:
    missing.append("torch")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    missing.append("transformers")

try:
    from datasets import load_dataset, Audio
except ImportError:
    missing.append("datasets")

try:
    import librosa
except ImportError:
    missing.append("librosa")

if missing:
    print("ERROR: Missing packages:", ", ".join(missing))
    print("Run: pip install", " ".join(missing))
    exit(1)

# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "storage" / "data"

# Dataset settings
DATASET_NAME = "GrigoriiA/libretta-tts-merged-dataset-audio-L10k"
CACHE_DIR = DATA_DIR / "audio" / "cache"

# Output paths
INDEX_DIR = DATA_DIR / "indices"
AUDIO_INDEX_PATH = INDEX_DIR / "audio.index"
AUDIO_METADATA_PATH = INDEX_DIR / "audio_metadata.json"

# CLAP model (512-dim embeddings, compatible with CLIP)
CLAP_MODEL = "laion/clap-htsat-unfused"

# Processing settings
BATCH_SIZE = 8  # Reduce if running out of memory
SAMPLE_RATE = 48000  # CLAP expects 48kHz
MAX_SAMPLES = None  # Set to int to limit (e.g., 1000 for testing)
# ============================================


def load_audio_from_dataset(example, target_sr=48000):
    """Load and resample audio from dataset example."""
    audio_data = example["audio"]
    
    # Get audio array and sample rate
    if isinstance(audio_data, dict):
        array = audio_data["array"]
        sr = audio_data["sampling_rate"]
    else:
        # Fallback for different dataset formats
        array = audio_data
        sr = target_sr
    
    # Resample if needed
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    
    return array.astype(np.float32)


def main():
    print("=" * 60)
    print(" Audio FAISS Index Builder (CLAP)")
    print("=" * 60)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create output directory
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_NAME}")
    print(f"Cache: {CACHE_DIR}")
    
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=str(CACHE_DIR)
    )
    
    # Decode audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    total_samples = len(dataset)
    if MAX_SAMPLES:
        total_samples = min(total_samples, MAX_SAMPLES)
        dataset = dataset.select(range(total_samples))
    
    print(f"Total samples to process: {total_samples}")
    
    # Load CLAP model
    print(f"\nLoading CLAP model: {CLAP_MODEL}")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL).to(device)
    model.eval()
    
    # Check embedding dimension
    print("Checking CLAP embedding dimension...")
    with torch.no_grad():
        # Use feature_extractor for audio input (ClapProcessor delegates to it)
        test_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second silence
        test_input = processor.feature_extractor(
            raw_speech=[test_audio],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        test_input = {k: v.to(device) for k, v in test_input.items()}
        test_emb = model.get_audio_features(**test_input)
        # Handle both tensor and BaseModelOutputWithPooling returns
        if hasattr(test_emb, 'pooler_output'):
            test_emb = test_emb.pooler_output
        elif hasattr(test_emb, 'embeds'):
            test_emb = test_emb.embeds
        embed_dim = test_emb.shape[1]
    print(f"Embedding dimension: {embed_dim}")
    
    if embed_dim != 512:
        print(f"WARNING: Expected 512-dim embeddings, got {embed_dim}")
        print("This may not be compatible with CLIP index!")
    
    # Process audio files
    print(f"\nEncoding {total_samples} audio files...")
    
    embeddings = []
    metadata = []
    errors = 0
    
    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Encoding"):
        batch_indices = range(i, min(i + BATCH_SIZE, total_samples))
        batch = dataset.select(batch_indices)
        
        try:
            # Load audio arrays
            audio_arrays = []
            batch_metadata = []
            
            for j, example in enumerate(batch):
                try:
                    audio_array = example["audio"]["array"].astype(np.float32)
                    audio_arrays.append(audio_array)
                    
                    # Get text/transcript if available
                    # The libretta dataset uses 'transcription' and 'transcription_normalised'
                    text = (
                        example.get("transcription_normalised", "") or 
                        example.get("transcription", "") or 
                        example.get("text", "") or 
                        example.get("transcript", "") or 
                        example.get("sentence", "")
                    )
                    
                    batch_metadata.append({
                        "id": f"audio_{i + j:06d}",
                        "text": text,
                        "content": text,  # For compatibility with unified index
                        "source": "libretta",
                        "modality": "audio",
                        "duration": len(audio_array) / SAMPLE_RATE
                    })
                except Exception as e:
                    errors += 1
                    continue
            
            if not audio_arrays:
                continue
            
            # Encode batch using feature_extractor
            inputs = processor.feature_extractor(
                raw_speech=audio_arrays,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                batch_embeddings = model.get_audio_features(**inputs)
                # Handle both tensor and BaseModelOutputWithPooling returns
                if hasattr(batch_embeddings, 'pooler_output'):
                    batch_embeddings = batch_embeddings.pooler_output
                elif hasattr(batch_embeddings, 'embeds'):
                    batch_embeddings = batch_embeddings.embeds
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            embeddings.append(batch_embeddings.cpu().numpy())
            metadata.extend(batch_metadata)
            
        except Exception as e:
            print(f"\nError in batch {i}: {e}")
            errors += 1
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created!")
        return
    
    # Stack all embeddings
    embeddings = np.vstack(embeddings).astype("float32")
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Errors/skipped: {errors}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
    index.add(embeddings)
    print(f"Index size: {index.ntotal} vectors")
    
    # Save index
    print(f"\nSaving index to: {AUDIO_INDEX_PATH}")
    faiss.write_index(index, str(AUDIO_INDEX_PATH))
    
    # Save metadata
    print(f"Saving metadata to: {AUDIO_METADATA_PATH}")
    with open(AUDIO_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print(" COMPLETE")
    print("=" * 60)
    print(f"  Audio index: {AUDIO_INDEX_PATH}")
    print(f"  Metadata: {AUDIO_METADATA_PATH}")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dim}")
    
    # Show sample metadata
    print("\nSample entries:")
    for m in metadata[:3]:
        text_preview = m['text'][:50] + "..." if len(m['text']) > 50 else m['text']
        print(f"  {m['id']}: \"{text_preview}\"")
    
    print("\nYou can now test audio search with:")
    print('  python scripts/demo_dcass.py "birds singing in the forest"')


if __name__ == "__main__":
    main()

# Flickr30K Corpus Expansion Guide

This guide walks through expanding the DCASS image corpus from ~6K images (Flickr8K) to ~36K images by adding the Flickr30K dataset.

## Why Flickr30K?

Our benchmark analysis revealed a **diversity problem**: with only 6K images, the same corpus items frequently match many different queries, reducing encoding quality.

**Expected improvements after expansion:**
- Image corpus: 6K → ~36K (6x increase)
- Better semantic coverage for diverse messages
- Reduced "same item matching everything" issue
- Improved CLIP similarity scores (target: 0.75 → 0.85+)

## Prerequisites

### 1. Kaggle Account & API Setup

The Flickr30K dataset is hosted on Kaggle. You'll need:

1. **Create Kaggle account** (if you don't have one):
   - Go to https://www.kaggle.com
   - Sign up / Sign in

2. **Get API credentials**:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json`

3. **Install credentials**:
   ```bash
   # Linux/Mac
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   
   # Windows (PowerShell)
   mkdir $HOME\.kaggle -Force
   mv $HOME\Downloads\kaggle.json $HOME\.kaggle\
   ```

4. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

5. **Verify setup**:
   ```bash
   kaggle datasets list -s flickr
   # Should show list of flickr-related datasets
   ```

### 2. System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk Space | 10 GB | 15 GB |
| GPU | - | CUDA-capable |
| Time (GPU) | - | ~30 min |
| Time (CPU) | ~3 hours | - |

### 3. Python Dependencies

Ensure these are installed:
```bash
pip install kaggle torch clip faiss-cpu tqdm numpy pillow

# OR with GPU support:
pip install kaggle torch clip faiss-gpu tqdm numpy pillow
```

## Step-by-Step Instructions

### Step 1: Download Flickr30K Dataset (~10-30 min)

```bash
cd /home/shanks/projects/dcass/dcass

# Download dataset from Kaggle (~4GB)
python scripts/download_flickr30k.py
```

**What this does:**
- Downloads `flickr-image-dataset.zip` from Kaggle (~4GB)
- Extracts 31,783 images and captions
- Organizes files into `data/raw/flickr30k/`

**Expected output:**
```
data/raw/flickr30k/
├── images/           # 31,783 .jpg files
└── results.csv       # 158,915 captions (5 per image)
```

**Troubleshooting:**
- If download fails, try downloading manually from:
  https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
- Extract to `data/raw/flickr30k/`

### Step 2: Build FAISS Index (~30 min GPU / 3 hours CPU)

```bash
# With GPU (recommended)
python scripts/build_flickr30k_index.py --gpu

# With GPU and larger batch size (faster if you have VRAM)
python scripts/build_flickr30k_index.py --gpu --batch-size 64

# CPU only (slower but works without CUDA)
python scripts/build_flickr30k_index.py

# Quick test first (1000 images)
python scripts/build_flickr30k_index.py --gpu --max-images 1000
```

**What this does:**
1. Loads Flickr30K images and captions
2. Generates CLIP embeddings (ViT-B/32) for each image
3. Merges with existing Flickr8K index
4. Saves combined FAISS index

**Expected output:**
```
data/indices/
├── image.index           # Updated: ~36K vectors
├── image_metadata.json   # Updated: metadata for all images
├── image.index.backup    # Backup of previous index
└── image_metadata.json.backup
```

### Step 3: Verify the Index

```bash
# Check index status
python -c "
from src.corpus.index.unified_index import create_index
idx = create_index()
print(idx.status())
"

# Expected output should show:
# 'image': {'loaded': True, 'count': ~36000}
```

### Step 4: Run Benchmarks

```bash
# Quick benchmark
python -m src.cli.main benchmark --quick --markdown

# Full benchmark (recommended)
python -m src.cli.main benchmark --markdown
```

Compare results with previous benchmark to measure improvement.

### Step 5: Test Encoding

```bash
# Test with a sample message
python -m src.cli.main encode "Meeting at the coffee shop tomorrow at 3pm"

# Verify diverse image selection
python -m src.cli.main encode "Hello world" --mode balanced
```

## Command Reference

### Download Script
```bash
python scripts/download_flickr30k.py [OPTIONS]

Options:
  --output, -o PATH      Output directory (default: data/raw/flickr30k)
  --captions-only        Skip images, download captions only (for testing)
  --skip-check           Skip Kaggle setup verification
```

### Index Builder Script
```bash
python scripts/build_flickr30k_index.py [OPTIONS]

Options:
  --data-dir, -d PATH    Flickr30K data directory (default: data/raw/flickr30k)
  --output-dir, -o PATH  Output directory (default: data/indices)
  --batch-size, -b INT   Batch size for embeddings (default: 32)
  --max-images INT       Limit images for testing
  --gpu                  Use GPU acceleration (recommended)
  --no-merge             Replace existing index instead of merging
```

## Troubleshooting

### "Kaggle credentials not found"
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If missing, download from https://www.kaggle.com/settings
```

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/build_flickr30k_index.py --gpu --batch-size 16
```

### "No image files found"
```bash
# Check the data directory structure
ls -la data/raw/flickr30k/
ls -la data/raw/flickr30k/images/

# The script searches multiple locations, but images should be in:
# data/raw/flickr30k/images/*.jpg
```

### "Import error: No module named 'clip'"
```bash
# Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git
```

### Slow embedding generation (CPU)
- Consider using Google Colab or a cloud GPU
- Or let it run overnight (expect ~3 hours)

## File Structure After Completion

```
data/
├── raw/
│   ├── flickr8k/           # Original ~6K images
│   │   ├── images/
│   │   └── text/
│   └── flickr30k/          # New ~31K images
│       ├── images/
│       └── results.csv
├── indices/
│   ├── image.index         # Combined ~36K image index
│   ├── image_metadata.json
│   ├── text.index          # Unchanged
│   ├── text_metadata.json
│   ├── audio.index         # Unchanged
│   └── audio_metadata.json
└── benchmarks/
    └── results/            # Benchmark results
```

## Expected Results

After successful completion:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Image corpus size | ~6,000 | ~36,000 |
| CLIP similarity (benchmark) | 0.754 | 0.80+ |
| Encoding diversity | Low | High |
| Index file size | ~12 MB | ~72 MB |

## Contact

If you encounter issues not covered here:
1. Check the script output for specific error messages
2. Verify all prerequisites are installed
3. Try the `--max-images 100` flag for quick debugging

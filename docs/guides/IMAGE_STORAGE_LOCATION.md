# Image Storage Location Guide

**Date:** April 6, 2026  
**Purpose:** Explain where images are stored in the refactored codebase

---

## Quick Answer

**Images will be stored at:**
```
storage/data/raw/flickr8k/images/
```

This directory will contain **8,000 images** after you run the download script.

---

## Directory Structure

### Complete Image Storage Path

```
dcass/
└── storage/
    └── data/
        └── raw/
            └── flickr8k/
                ├── images/              ← 8,000 image files (.jpg)
                │   ├── 1000268201_693b08cb0e.jpg
                │   ├── 1001773457_577c3a7d70.jpg
                │   ├── ...
                │   └── 997722733_0cb5439472.jpg
                ├── text/                ← Caption files
                │   ├── Flickr8k.token.txt
                │   ├── Flickr_8k.trainImages.txt
                │   └── ...
                └── Flickr8k_Dataset.zip ← Original download (can be deleted)
```

---

## How to Download Images

### Option 1: Using Make (Recommended)

```bash
make download-data
```

This will:
1. Download Flickr8k dataset (~1GB)
2. Extract to `storage/data/raw/flickr8k/`
3. Verify 8,000 images are present

### Option 2: Using Python Script Directly

```bash
python scripts/data/download_flickr8k.py
```

### Option 3: Custom Location

```bash
python scripts/data/download_flickr8k.py --output /custom/path
```

---

## After Download

### Verify Images

```bash
# Check images directory exists
ls storage/data/raw/flickr8k/images/

# Count images (should be 8,000)
ls storage/data/raw/flickr8k/images/*.jpg | wc -l

# View sample image names
ls storage/data/raw/flickr8k/images/ | head -10
```

### Expected Output

```
storage/data/raw/flickr8k/
├── images/                  # 8,000 .jpg files
├── text/                    # Caption files
└── Flickr8k_Dataset.zip     # Can be deleted after extraction
```

---

## Image Specifications

### Flickr8k Dataset Details

- **Total Images:** 8,000
- **Format:** JPEG (.jpg)
- **Size:** ~1GB total
- **Content:** Everyday scenes (people, animals, objects, activities)
- **Captions:** 5 per image (40,000 total captions)

### Example Image Names

```
1000268201_693b08cb0e.jpg
1001773457_577c3a7d70.jpg
1002674143_1b742ab4b8.jpg
1003163366_44323f5815.jpg
...
```

---

## Configuration

### Config File Reference

The image paths are configured in **`config/default.yaml`**:

```yaml
corpus:
  image:
    enabled: true
    source_dir: "storage/data/raw/flickr8k"
    captions_file: "storage/data/raw/flickr8k/captions.txt"
    images_dir: "storage/data/raw/flickr8k/images"      ← Image directory
    supported_formats: [".jpg", ".jpeg", ".png", ".webp"]
```

---

## Building Image Index

After downloading, build the FAISS index:

```bash
# Build all indices (text + image)
make build-index

# Or build only image index
make build-image-index

# Or use Python directly
python scripts/data/build_indices.py --modality image
```

This creates:
- `storage/data/indices/image.index` - FAISS vector index
- `storage/data/indices/image_metadata.json` - Image metadata

---

## Using Images in DCASS

### Encode Message (Alice)

```bash
# Encode using images
python scripts/runtime/run_sender.py \
  --message "Hello World" \
  --mode auto \
  --modality image
```

### Retrieve Similar Images

```python
from src.corpus.index.unified_index import UnifiedIndex

# Load index
index = UnifiedIndex()
index.load("image", "storage/data/indices/image.index")

# Search for similar images
query = "a dog playing in water"
results = index.search(query, modality="image", k=5)

# Results contain image IDs that map to:
# storage/data/raw/flickr8k/images/{image_id}.jpg
```

---

## Disk Space Requirements

### Before Download
- Required: ~1.5GB free space

### After Download
- Images: ~1GB
- Indices: ~50MB
- Captions: ~5MB
- **Total: ~1.1GB**

### Clean Up

After building indices, you can delete the zip file:

```bash
rm storage/data/raw/flickr8k/Flickr8k_Dataset.zip
rm storage/data/raw/flickr8k/Flickr8k_text.zip
```

This frees up ~400MB.

---

## Docker Usage

### Images in Docker Container

When using Docker, the images are mounted at:

```
Host:      ./storage/data/raw/flickr8k/images/
Container: /app/storage/data/raw/flickr8k/images/
```

### Docker Volume Mount

In `docker-compose.yml`:

```yaml
volumes:
  - ./storage/data:/app/storage/data:ro  # Read-only for safety
```

---

## Troubleshooting

### Images Not Found

**Problem:** Scripts can't find images

**Solution:**
1. Check if directory exists:
   ```bash
   ls storage/data/raw/flickr8k/images/
   ```

2. If empty, download again:
   ```bash
   make download-data
   ```

3. Verify config:
   ```bash
   grep images_dir config/default.yaml
   ```

### Wrong Path in Code

**Problem:** Code references old `data/` path

**Solution:**
1. Check if script was updated:
   ```bash
   grep "storage/data" scripts/data/download_flickr8k.py
   ```

2. If not, update manually or re-apply refactoring

### Permission Issues

**Problem:** Can't write to storage directory

**Solution:**
```bash
chmod -R 755 storage/
```

---

## Alternative Datasets

### Flickr30k (30,000 images)

```bash
python scripts/data/download_flickr30k.py
# Images will be at: storage/data/raw/flickr30k/images/
```

### Custom Images

Place your own images at:
```
storage/data/raw/custom_images/
```

Then update `config/default.yaml`:
```yaml
corpus:
  image:
    source_dir: "storage/data/raw/custom_images"
    images_dir: "storage/data/raw/custom_images"
```

---

## Summary

**Image Location:** `storage/data/raw/flickr8k/images/`

**Download Command:** `make download-data` or `python scripts/data/download_flickr8k.py`

**Build Index:** `make build-index`

**Verify:** `ls storage/data/raw/flickr8k/images/ | wc -l` (should show 8000)

---

**All image paths have been updated in the refactored codebase!** ✅

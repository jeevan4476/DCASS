import csv
from pathlib import Path
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------- Paths (match your structure) --------
CAPTIONS_CSV = Path("")
IMAGES_DIR = Path("")
OUTPUT_DIR = Path("")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EMB_PATH = OUTPUT_DIR / "image_embeddings.npy"
IMAGE_IDS_PATH = OUTPUT_DIR / "image_ids.txt"

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------- Load CLIP --------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -------- Get unique image IDs --------
image_ids = set()

with CAPTIONS_CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_ids.add(row["image_id"])

image_ids = sorted(image_ids)
print(f"Images to embed: {len(image_ids)}")

# -------- Embed images --------
embeddings = []

with torch.no_grad():
    for image_id in tqdm(image_ids, desc="Embedding images"):
        image_path = IMAGES_DIR / f"{image_id}.jpg"

        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings.append(embedding.cpu().numpy()[0])

# -------- Save outputs --------
embeddings = np.array(embeddings, dtype=np.float32)

np.save(IMAGE_EMB_PATH, embeddings)

with IMAGE_IDS_PATH.open("w", encoding="utf-8") as f:
    for img_id in image_ids:
        f.write(img_id + "\n")

print("\nSaved:")
print(f"- {IMAGE_EMB_PATH}")
print(f"- {IMAGE_IDS_PATH}")
print(f"Embedding shape: {embeddings.shape}")

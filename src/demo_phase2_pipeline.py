import re
import csv
import torch
import clip
import faiss
import numpy as np
from pathlib import Path

# ==============================
# CONFIG
# ==============================
SENTENCE = "a dog running through water and a man riding a bicycle"
TOP_K_IMAGE = 1
TOP_K_CAPTION = 1

# ==============================
# PATHS
# ==============================
EMB_DIR = Path(r"D:\DCASS\location")
FAISS_INDEX_PATH = EMB_DIR / "faiss_image.index"
IMAGE_IDS_PATH = EMB_DIR / "image_ids.txt"
CAPTION_MAP_PATH = EMB_DIR / "caption_map.csv"
IMAGE_EMB_PATH = EMB_DIR / "image_embeddings.npy"
CAPTION_EMB_PATH = EMB_DIR / "caption_embeddings.npy"

# ==============================
# LOAD DATA
# ==============================
print("Loading embeddings and index...")

image_embeddings = np.load(IMAGE_EMB_PATH)
caption_embeddings = np.load(CAPTION_EMB_PATH)

image_ids = IMAGE_IDS_PATH.read_text().splitlines()

caption_map = []
with CAPTION_MAP_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption_map.append(row)

index = faiss.read_index(str(FAISS_INDEX_PATH))

# ==============================
# LOAD CLIP
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# ==============================
# STEP 1 — CHUNK TEXT
# ==============================
def chunk_text(text):
    chunks = re.split(r",| and ", text.lower())
    return [c.strip() for c in chunks if c.strip()]

chunks = chunk_text(SENTENCE)

print("\nINPUT SENTENCE:")
print(SENTENCE)

print("\nSEMANTIC CHUNKS:")
for i, c in enumerate(chunks, 1):
    print(f"{i}. {c}")

# ==============================
# STEP 2 — TEXT → IMAGE SEQUENCE
# ==============================
encoded_images = []

print("\nENCODING (Text → Images):")

with torch.no_grad():
    for i, chunk in enumerate(chunks, 1):
        tokens = clip.tokenize([chunk]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype("float32")

        scores, indices = index.search(emb, TOP_K_IMAGE)
        image_id = image_ids[indices[0][0]]
        encoded_images.append(image_id)

        print(f"{i}. '{chunk}' → {image_id}")

# ==============================
# STEP 3 — IMAGE SEQUENCE → TEXT
# ==============================
print("\nDECODING (Images → Text):")

decoded_captions = []

for image_id in encoded_images:
    image_idx = image_ids.index(image_id)
    image_vec = image_embeddings[image_idx]

    sims = caption_embeddings @ image_vec
    best_idx = np.argsort(sims)[-TOP_K_CAPTION:][::-1][0]

    caption = caption_map[best_idx]["caption"]
    decoded_captions.append(caption)

    print(f"{image_id} → {caption}")

# ==============================
# FINAL OUTPUT
# ==============================
print("\nRECONSTRUCTED MESSAGE:")
print(". ".join(decoded_captions))

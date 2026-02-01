import torch
import clip
import faiss
import numpy as np
from pathlib import Path
import re

# -------- Paths --------
EMB_DIR = Path("")
FAISS_INDEX_PATH = EMB_DIR / "faiss_image.index"
IMAGE_IDS_PATH = EMB_DIR / "image_ids.txt"

# -------- Config --------
SENTENCE = "a horse running in a field and people watching from the side"
TOP_K = 1

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------- Load CLIP --------
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# -------- Load FAISS --------
index = faiss.read_index(str(FAISS_INDEX_PATH))
image_ids = IMAGE_IDS_PATH.read_text().splitlines()

# -------- Chunking --------
def chunk_text(text):
    chunks = re.split(r",| and ", text.lower())
    return [c.strip() for c in chunks if c.strip()]

chunks = chunk_text(SENTENCE)

print("\nSentence:")
print(SENTENCE)
print("\nChunks:")
for i, c in enumerate(chunks, 1):
    print(f"{i}. {c}")

# -------- Encode + Search --------
print("\nEncoded image sequence:")

with torch.no_grad():
    for i, chunk in enumerate(chunks, 1):
        tokens = clip.tokenize([chunk]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype("float32")

        scores, indices = index.search(emb, TOP_K)
        img_id = image_ids[indices[0][0]]

        print(f"{i}. '{chunk}' → {img_id}")

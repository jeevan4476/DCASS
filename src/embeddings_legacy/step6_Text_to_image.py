import torch
import clip
import faiss
import numpy as np
from pathlib import Path

# -------- Paths --------
EMB_DIR = Path("data/embeddings")
FAISS_INDEX_PATH = EMB_DIR / "faiss_image.index"
IMAGE_IDS_PATH = EMB_DIR / "image_ids.txt"

# -------- Config --------
TOP_K = 5
QUERY = "a cat flying in the sky"



# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------- Load CLIP --------
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# -------- Load FAISS index --------
index = faiss.read_index(str(FAISS_INDEX_PATH))

# -------- Load image IDs --------
image_ids = IMAGE_IDS_PATH.read_text().splitlines()

# -------- Embed query --------
with torch.no_grad():
    text_tokens = clip.tokenize([QUERY]).to(device)
    text_emb = model.encode_text(text_tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().astype("float32")

# -------- Search --------
scores, indices = index.search(text_emb, TOP_K)

print("\nQuery:")
print(f"  {QUERY}")

print("\nTop matches:")
for rank, idx in enumerate(indices[0], 1):
    print(f"{rank}. {image_ids[idx]} (score={scores[0][rank-1]:.4f})")

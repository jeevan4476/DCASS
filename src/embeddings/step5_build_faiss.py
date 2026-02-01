import numpy as np
import faiss
from pathlib import Path

# -------- Paths --------
EMB_DIR = Path("")
IMAGE_EMB_PATH = EMB_DIR / "image_embeddings.npy"
FAISS_INDEX_PATH = EMB_DIR / "faiss_image.index"

# -------- Load embeddings --------
image_embeddings = np.load(IMAGE_EMB_PATH).astype("float32")

num_images, dim = image_embeddings.shape
print(f"Loaded image embeddings: {num_images} x {dim}")

# -------- Build FAISS index --------
index = faiss.IndexFlatIP(dim)  # cosine similarity (vectors already normalized)
index.add(image_embeddings)

print(f"FAISS index size: {index.ntotal}")

# -------- Save index --------
faiss.write_index(index, str(FAISS_INDEX_PATH))
print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")

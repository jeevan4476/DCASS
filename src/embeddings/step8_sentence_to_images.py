import torch
import clip
import faiss
import numpy as np
from pathlib import Path
import re

# -------- Paths --------
EMB_DIR = Path("") #add the path to the directory where you saved the FAISS index and image IDs
FAISS_INDEX_PATH = EMB_DIR / "faiss_image.index"
IMAGE_IDS_PATH = EMB_DIR / "image_ids.txt"

TOP_K = 1

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load CLIP (ONCE) --------
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# -------- Load FAISS (ONCE) --------
index = faiss.read_index(str(FAISS_INDEX_PATH))
image_ids = IMAGE_IDS_PATH.read_text().splitlines()


# -------- Chunking --------
def chunk_text(text: str):
    chunks = re.split(r",| and ", text.lower())
    return [c.strip() for c in chunks if c.strip()]


# ======================================================
# ✅ PUBLIC API — THIS IS WHAT CLI IMPORTS
# ======================================================
def sentence_to_image_sequence(sentence: str) -> list[str]:
    """
    Encodes a sentence into a list of image IDs.
    This is the ONLY function Phase 3 should call.
    """

    chunks = chunk_text(sentence)
    results = []

    print("\nSentence:")
    print(sentence)

    print("\nChunks:")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")

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
            results.append(img_id)

    return results


# ======================================================
# SCRIPT MODE (kept for standalone testing)
# ======================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step8_sentence_to_images.py <sentence>")
        sys.exit(1)

    sentence = sys.argv[1]
    sentence_to_image_sequence(sentence)

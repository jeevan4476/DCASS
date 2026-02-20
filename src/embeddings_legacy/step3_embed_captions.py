import csv
from pathlib import Path
import torch
import clip
import numpy as np
from tqdm import tqdm

# -------- Paths --------
CAPTIONS_CSV = Path("data/metadata/captions.csv")
OUTPUT_DIR = Path("data/embeddings")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_EMB_PATH = OUTPUT_DIR / "caption_embeddings.npy"
CAP_MAP_PATH = OUTPUT_DIR / "caption_map.csv"

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------- Load CLIP --------
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

captions = []
caption_map = []

# -------- Read captions --------
with CAPTIONS_CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        captions.append(row["caption"])
        caption_map.append({
            "caption_index": idx,
            "image_id": row["image_id"],
            "caption": row["caption"]
        })

print(f"Captions to embed: {len(captions)}")

# -------- Embed captions --------
embeddings = []

with torch.no_grad():
    for caption in tqdm(captions, desc="Embedding captions"):
        text_tokens = clip.tokenize(caption).to(device)
        embedding = model.encode_text(text_tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embeddings.append(embedding.cpu().numpy()[0])

# -------- Save outputs --------
embeddings = np.array(embeddings, dtype=np.float32)
np.save(CAP_EMB_PATH, embeddings)

with CAP_MAP_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["caption_index", "image_id", "caption"]
    )
    writer.writeheader()
    writer.writerows(caption_map)

print("\nSaved:")
print(f"- {CAP_EMB_PATH}")
print(f"- {CAP_MAP_PATH}")
print(f"Embedding shape: {embeddings.shape}")

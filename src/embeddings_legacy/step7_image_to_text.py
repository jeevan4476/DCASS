import numpy as np
import csv
from pathlib import Path

# -------- Paths --------
EMB_DIR = Path("data/embedding")

IMAGE_EMB = np.load(EMB_DIR / "image_embeddings.npy")
CAPTION_EMB = np.load(EMB_DIR / "caption_embeddings.npy")

IMAGE_IDS = (EMB_DIR / "image_ids.txt").read_text().splitlines()

CAPTION_MAP = []
with (EMB_DIR / "caption_map.csv").open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        CAPTION_MAP.append(row)

# -------- Config --------
IMAGE_ID = "3441531010_8eebbb507e"  # try any ID you retrieved earlier
TOP_K = 5

# -------- Locate image embedding --------
image_idx = IMAGE_IDS.index(IMAGE_ID)
image_vec = IMAGE_EMB[image_idx]

# -------- Similarity search over captions --------
scores = CAPTION_EMB @ image_vec
top_idx = np.argsort(scores)[-TOP_K:][::-1]

print(f"\nImage ID: {IMAGE_ID}")
print("\nTop decoded captions:")

for rank, idx in enumerate(top_idx, 1):
    cap = CAPTION_MAP[idx]
    print(f"{rank}. {cap['caption']}")

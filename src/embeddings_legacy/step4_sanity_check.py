import numpy as np
import csv
import random
from pathlib import Path

# -------- Paths --------
EMB_DIR = Path("data/embeddings")

IMAGE_EMB = np.load(EMB_DIR / "image_embeddings.npy")
CAPTION_EMB = np.load(EMB_DIR / "caption_embeddings.npy")

IMAGE_IDS = (EMB_DIR / "image_ids.txt").read_text().splitlines()

CAPTION_MAP = []
with (EMB_DIR / "caption_map.csv").open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        CAPTION_MAP.append(row)

# -------- Helper: cosine similarity --------
def cosine_sim(a, b):
    return np.dot(a, b)

# --------------------------------------------------
# TEST 1 — Caption → Image
# --------------------------------------------------
print("\n=== TEST 1: Caption → Image ===")

cap_idx = random.randint(0, len(CAPTION_EMB) - 1)
cap_vec = CAPTION_EMB[cap_idx]
cap_info = CAPTION_MAP[cap_idx]

sims = IMAGE_EMB @ cap_vec
top_img_idx = np.argsort(sims)[-5:][::-1]

print("\nCaption:")
print(cap_info["caption"])
print("\nExpected image_id:", cap_info["image_id"])
print("\nTop-5 retrieved images:")

for rank, idx in enumerate(top_img_idx, 1):
    print(f"{rank}. {IMAGE_IDS[idx]}")

# --------------------------------------------------
# TEST 2 — Image → Caption
# --------------------------------------------------
print("\n=== TEST 2: Image → Caption ===")

image_id = cap_info["image_id"]
image_idx = IMAGE_IDS.index(image_id)
img_vec = IMAGE_EMB[image_idx]

sims = CAPTION_EMB @ img_vec
top_cap_idx = np.argsort(sims)[-5:][::-1]

print("\nImage ID:", image_id)
print("\nTop-5 retrieved captions:")

for rank, idx in enumerate(top_cap_idx, 1):
    cap = CAPTION_MAP[idx]
    print(f"{rank}. [{cap['image_id']}] {cap['caption']}")

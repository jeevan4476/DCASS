import os
from datasets import load_dataset
from huggingface_hub import login

# OPTIONAL: login if dataset ever needs auth
# login(token="YOUR_HF_TOKEN")

# ================================
# CONFIG
# ================================
DATASET_NAME = "GrigoriiA/libretta-tts-merged-dataset-audio-L10k"
TARGET_DIR = r"D:\DCASS\3gone\raw_audio"

os.makedirs(TARGET_DIR, exist_ok=True)

print("📥 Downloading dataset to:", TARGET_DIR)

dataset = load_dataset(
    DATASET_NAME,
    cache_dir=TARGET_DIR,   # 👈 forces D drive usage
    split="train"
)

print("✅ Dataset downloaded successfully")
print(dataset)

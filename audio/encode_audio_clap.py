import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import ClapModel, ClapProcessor
import faiss

# ================= CONFIG =================
DATASET_NAME = "GrigoriiA/libretta-tts-merged-dataset-audio-l10k"
CACHE_DIR = r"D:\DCASS\hf_cache"   # <- CLEAN, SAFE LOCATION
EMBEDDINGS_OUT = "audio_embeddings.npy"
FAISS_INDEX_OUT = "audio.index"
BATCH_SIZE = 4
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Device: {device}")

# ---------- LOAD DATASET ----------
dataset = load_dataset(
    DATASET_NAME,
    split="train",
    cache_dir=CACHE_DIR
)

# DO NOT DECODE AUDIO (avoids torchcodec + ffmpeg)
dataset = dataset.cast_column("audio", Audio(decode=False))

print(f"✅ Dataset loaded: {len(dataset)} samples")

# ---------- LOAD CLAP ----------
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
model.eval()

embeddings = []

# ---------- ENCODE ----------
for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Encoding audio"):
    batch = dataset[i:i + BATCH_SIZE]
    audio_paths = [a["path"] for a in batch["audio"]]

    inputs = processor(
        audios=audio_paths,
        sampling_rate=48000,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        emb = model.get_audio_features(**inputs)

    embeddings.append(emb.cpu().numpy())

# ---------- SAVE ----------
embeddings = np.vstack(embeddings)
np.save(EMBEDDINGS_OUT, embeddings)

# ---------- FAISS ----------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_OUT)

print("✅ Audio embeddings + FAISS index saved")

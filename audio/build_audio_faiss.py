import os
import pickle
import numpy as np
import torch
import librosa
import faiss
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_data")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "audio_faiss.index")
AUDIO_PATHS_FILE = os.path.join(BASE_DIR, "audio_paths.pkl")

SAMPLE_RATE = 16000
MIN_AUDIO_LEN = 0.5  # seconds
MODEL_NAME = "facebook/wav2vec2-base-960h"
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Device: {DEVICE}")

# -------- LOAD MODEL --------
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# -------- FIND AUDIO FILES --------
audio_files = []
for root, _, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            audio_files.append(os.path.join(root, f))

print(f"🎧 Found {len(audio_files)} audio files")

if len(audio_files) == 0:
    raise RuntimeError("❌ No WAV files found in audio_data/")

# -------- ENCODE AUDIO --------
embeddings = []
valid_paths = []

for path in tqdm(audio_files, desc="Encoding audio"):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio")

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_LEN:
            raise ValueError(f"Too short ({duration:.2f}s)")

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(
                inputs.input_values.to(DEVICE),
                attention_mask=inputs.attention_mask.to(DEVICE)
                if "attention_mask" in inputs else None
            )

        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(emb)
        valid_paths.append(path)

    except Exception as e:
        print(f"⚠️ Skipped {path} → {e}")

if len(embeddings) == 0:
    raise RuntimeError("❌ ZERO valid embeddings created")

# -------- BUILD FAISS --------
embeddings = np.vstack(embeddings).astype("float32")
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

with open(AUDIO_PATHS_FILE, "wb") as f:
    pickle.dump(valid_paths, f)

print("✅ FAISS index built")
print(f"📦 Total vectors: {index.ntotal}")

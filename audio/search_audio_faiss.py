import os
import sys
import pickle
import numpy as np
import torch
import faiss
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "audio_faiss.index")
AUDIO_PATHS_FILE = os.path.join(BASE_DIR, "audio_paths.pkl")

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Device: {DEVICE}")

# -------- LOAD MODEL --------
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# -------- LOAD FAISS --------
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"❌ Missing FAISS index: {FAISS_INDEX_PATH}")

if not os.path.exists(AUDIO_PATHS_FILE):
    raise FileNotFoundError(f"❌ Missing audio paths file")

index = faiss.read_index(FAISS_INDEX_PATH)

with open(AUDIO_PATHS_FILE, "rb") as f:
    audio_paths = pickle.load(f)

# -------- ENCODER --------
def encode_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

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

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().astype("float32")

# -------- SEARCH --------
if len(sys.argv) < 2:
    print("❌ Usage: python search_audio_faiss.py query.wav")
    sys.exit(1)

query_audio = sys.argv[1]
query_vec = encode_audio(query_audio)

k = 5
distances, indices = index.search(query_vec, k)

print("\n🎧 Top Matches:")
for rank, idx in enumerate(indices[0]):
    print(f"{rank+1}. {audio_paths[idx]} | Distance: {distances[0][rank]:.4f}")

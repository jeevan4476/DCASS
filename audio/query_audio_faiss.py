import torch
import faiss
import numpy as np
import librosa
from transformers import ClapProcessor, ClapModel

# ---------- CONFIG ----------
FAISS_INDEX_PATH = "audio.index"
EMBEDDINGS_PATH = "audio_embeddings.npy"
MODEL_NAME = "laion/clap-htsat-unfused"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", device)

# ---------- LOAD MODEL ----------
processor = ClapProcessor.from_pretrained(MODEL_NAME)
model = ClapModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------- LOAD FAISS ----------
index = faiss.read_index(FAISS_INDEX_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print("✅ FAISS index loaded")
print("📦 Index size:", index.ntotal)

# ---------- ENCODE QUERY ----------
def encode_audio(path):
    audio, sr = librosa.load(path, sr=48000)
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        emb = model.get_audio_features(**inputs)

    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")

# ---------- SEARCH ----------
def search(query_audio_path):
    query_emb = encode_audio(query_audio_path)
    distances, indices = index.search(query_emb, TOP_K)

    print("\n🔍 Top matches:")
    for rank, idx in enumerate(indices[0]):
        print(f"{rank+1}. Audio ID {idx} | Similarity {distances[0][rank]:.4f}")

# ---------- RUN ----------
if __name__ == "__main__":
    query_path = "query2.wav"  # <-- your test audio
    search(query_path)

# DCASS · GAN Steganography Demo

A Docker-based live demo of the **Deep Covert Adaptive Steganography System**.

## What It Shows

```
Covert Message
     │
     ▼ ① Semantic Chunking       "a dog running…" → ["a dog running", "a man riding"]
     │
     ▼ ② CLIP Encode → Images    each chunk → nearest image ID via FAISS
     │
     ▼ ③ GAN Schedule            TemporalPatternGenerator → delays + channel selection
     │
     ▼ ④ Steganographic TX       simulated sends with human-like timing jitter
     │
     ▼ ⑤ Decode                  image IDs → captions → reconstructed message
```

The GAN (GRU + Multi-head Attention) generates **human-like inter-transmission delays**
and **channel-hopping schedules** to evade Deep Packet Inspection.

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Clone / copy this folder
cd dcass-demo

# 2. Build and run
docker compose up --build

# 3. Open browser
open http://localhost:5000
```

### Option B — Docker CLI

```bash
docker build -t dcass-demo .
docker run -p 5000:5000 dcass-demo
```

### Option C — Local Python

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## Project Structure

```
dcass-demo/
├── app.py                        # Flask server + SSE stream
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── templates/
│   └── index.html                # Live dashboard UI
└── src/
    └── stealth/
        └── gan/
            ├── __init__.py
            └── generator.py      # TemporalPatternGenerator (GRU + Attention)
```

---

## Demo Scenarios

| Scenario | Message | Chunks |
|----------|---------|--------|
| A | a dog running through water and a man riding a bicycle | 2 |
| B | a cat sleeping on a sofa and children playing in a park | 3 |
| C | a mountain peak covered in snow and a river flowing through a valley | 2 |

---

## GAN Architecture

```
Latent Noise z (128-d)
      │
      ├── Time-of-day encoding (sin/cos → 32-d)
      │
      ▼
  Linear projection → LayerNorm → ReLU   [256-d]
      │
      ▼
  GRU (2-layer, hidden=256)
      │
      ▼
  Multi-head Self-Attention (8 heads)
      │
  Residual + LayerNorm
      │
  ┌───┴────────┬──────────────┐
  ▼            ▼              ▼
Delay Head  Channel Head  Confidence Head
(Softplus)  (Logits×3)    (Sigmoid)
```

- **Delay Head** — outputs positive inter-transmission delays (Softplus ensures > 0)
- **Channel Head** — selects which social platform to post to (Gumbel-Softmax)
- **Confidence Head** — generator's self-assessed human-likeness score

---

## Notes

- The prototype's CLIP + FAISS retrieval is **simulated** in the demo (pre-mapped IDs).
  Plug in your actual `image_embeddings.npy`, `caption_embeddings.npy`, and FAISS index
  to run full end-to-end retrieval.
- The GAN weights are **untrained random initialisation** for the demo.
  Swap in your trained checkpoint via `generator.load_state_dict(torch.load('ckpt.pt'))`.
- Delays are capped at 3 s in the SSE stream for demo pacing; actual values are shown.

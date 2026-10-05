# DCASS — Quickstart Guide

Everything you need to run the full system: backend API, frontend UI, GAN+RL training, and the end-to-end pipeline.

---

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- (Optional) Docker & Docker Compose for containerised runs

---

## 1. Python environment setup

```bash
# From the project root
cd /home/karash1/projects/DCASS

# Create a virtual environment (skip if you already have one)
python -m venv venv
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt

# CLIP must be installed separately
pip install git+https://github.com/openai/CLIP.git
```

---

## 2. Environment variables

```bash
# Copy the example and edit as needed
cp .env.example .env
```

Key variables (all optional — sensible defaults exist):

| Variable | Default | Purpose |
|---|---|---|
| `DCASS_DEVICE` | `cpu` | `cuda` if you have a GPU |
| `DCASS_CORS_ORIGINS` | `http://localhost:3000` | Frontend origin for CORS |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL the frontend calls |
| `DCASS_JWT_SECRET` | random at startup | Set a fixed value for persistent sessions |

---

## 3. Start the backend API

```bash
# Terminal 1
uvicorn src.api.server:app --reload --port 8000
```

On first start the backend will:
- Create `storage/auth.db` (SQLite) with the users table
- Seed three demo users: `alice/alice1234`, `bob/bob1234`, `charlie/charlie1234`
- Load the SemanticEngine (FAISS indices, codebook, etc.)

Verify it's running:

```bash
curl http://localhost:8000/api/status
```

---

## 4. Start the frontend

```bash
# Terminal 2
cd frontend
npm install      # first time only
npm run dev
```

Open **http://localhost:3000** in a browser.

---

## 5. Demo walkthrough (UI)

1. Go to http://localhost:3000 → click **Login**
2. Sign in as `alice` / `alice1234`
3. Click **Send** → pick `bob` as recipient → type a message → click **Send**
4. Open a second browser / incognito window → http://localhost:3000/login
5. Sign in as `bob` / `bob1234`
6. Click **Inbox** → you'll see alice's message → click **Decode** to reveal the plaintext

---

## 6. Demo walkthrough (CLI / curl)

```bash
# Login as alice
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"alice1234"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: $TOKEN"

# Encode a message
curl -s -X POST http://localhost:8000/api/encode \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Meet at the cafe at noon","use_ecc":true,"mode":"exact_vcp"}'

# Transmit (encode + schedule + send to bob)
curl -s -X POST http://localhost:8000/api/transmit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"media_ids":["id1","id2"],"recipient_username":"bob","mode":"auto","speed_multiplier":100}'

# Login as bob and check inbox
BOB_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"bob","password":"bob1234"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s http://localhost:8000/api/inbox \
  -H "Authorization: Bearer $BOB_TOKEN"
```

---

## 7. Run the pipeline (same-process round-trip)

This runs encode → GAN+RL schedule → transmit → reassemble → decode in one process:

```bash
python scripts/runtime/run_pipeline.py \
  --message "Meet at the cafe at noon" \
  --sender alice \
  --recipient bob \
  --mode auto \
  --speed 100
```

Or from Python:

```python
from src.stealth.pipeline import run_pipeline

result = run_pipeline(
    message="Meet at the cafe at noon",
    sender="alice",
    recipient="bob",
    mode="auto",
    speed_multiplier=100,
)
print(result.summary())
print(f"Match: {result.decoded_ok}")
```

---

## 8. GAN + RL training

### 8a. Train the GAN (WGAN-GP)

```bash
python scripts/stealth/train_gan.py \
  --epochs 20 \
  --batch-size 32 \
  --out storage/models/gan_generator.pt
```

Falls back to synthetic data if `data/human_traffic.json` doesn't exist.

### 8b. Evaluate the Warden

```bash
python scripts/stealth/eval_warden.py \
  --checkpoint storage/models/gan_generator.pt
```

Look for AUC trending toward 0.5 (generator fooling the critic).

### 8c. Train the RL agent (PPO against frozen Warden)

```bash
python scripts/stealth/train_rl.py \
  --warden-checkpoint storage/models/gan_generator.pt \
  --episodes 500 \
  --lambda-stealth 50 \
  --out storage/models/rl_agent.pt
```

### 8d. Verify end-to-end with trained models

```bash
python -c "
from src.stealth.stealth_scheduler import StealthScheduler
s = StealthScheduler(num_channels=3)
r = s.schedule(['m%d' % i for i in range(20)], mode='auto')
print('mode:', r['mode_used'])
print('delays:', len(r['delays']))
print('channels used:', len(set(r['channels'])))
"
```

Expected: `mode: rl`, 20 delays, multiple channels.

---

## 9. Run the full evaluation suite

```bash
python scripts/testing/evaluate_stealth.py \
  --gan-checkpoint storage/models/gan_generator.pt \
  --rl-checkpoint storage/models/rl_agent.pt
```

---

## 10. Run tests

```bash
# All stealth/pipeline tests
pytest tests/test_stealth/ -v

# Just the integration test (tiny GAN → RL → scheduler)
pytest tests/test_stealth/test_integration.py -v

# Just the pipeline round-trip tests
pytest tests/test_stealth/test_pipeline.py -v
```

---

## 11. Docker (alternative)

```bash
# Sender + receiver (static mode, no training needed)
docker compose up

# With the web UI
docker compose --profile web up

# Training pipeline
docker compose --profile training run dcass-train-gan
docker compose --profile training run dcass-train-rl

# Full rebuild
docker compose up --build

# Cleanup
docker compose down -v
```

---

## Project structure (key files)

```
src/
  api/
    server.py          # FastAPI backend (auth, encode, transmit, inbox)
    auth.py            # SQLite + bcrypt + JWT auth
  stealth/
    stealth_scheduler.py   # Scheduler: auto (RL → GAN → static)
    pipeline.py            # End-to-end round-trip module
    gan/trainer.py         # WGAN-GP trainer
    rl/environment.py      # PPO environment
    rl/ppo_agent.py        # PPO agent
  engine/
    semantic_engine.py     # Encode/decode engine (FAISS + codebook)

frontend/
  src/app/
    page.tsx           # Home page
    login/page.tsx     # Login
    register/page.tsx  # Register
    send/page.tsx      # Send a message
    inbox/page.tsx     # View + decode messages

scripts/
  stealth/
    train_gan.py       # GAN training CLI
    train_rl.py        # RL training CLI
    eval_warden.py     # Warden evaluation
  runtime/
    run_pipeline.py    # Pipeline CLI wrapper
  testing/
    evaluate_stealth.py  # Full evaluation suite

storage/
  models/              # Checkpoints (gan_generator.pt, rl_agent.pt)
  shared_channel/      # Per-recipient packet directories
  auth.db              # SQLite user database (auto-created)
```

# Getting Started with DCASS

> **Dynamic Context-Aware Semantic Steganography**  
> Branch: `feature/stealth-ai` | Python 3.10+ | Docker required for simulation

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Project Structure Overview](#3-project-structure-overview)
4. [Quick Start (No Docker)](#4-quick-start-no-docker)
5. [Quick Start (Docker)](#5-quick-start-docker)
6. [What Happens Under the Hood](#6-what-happens-under-the-hood)
7. [Configuration](#7-configuration)
8. [Environment Variables](#8-environment-variables)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required |
| Docker | 24.0+ | Required for simulation |
| Docker Compose | v2.x | Use `docker compose` (not `docker-compose`) |
| Git | any | — |
| CUDA (optional) | 11.8+ | CPU mode works fine |

> **Windows users**: Use PowerShell or WSL2. The project uses Unix-style paths inside containers.

---

## 2. Clone & Install

### 2.1 Clone the repository

```bash
git clone <repo-url>
cd dcass
git checkout feature/stealth-ai
```

### 2.2 Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 2.4 Install CLIP (required for image encoding)

```bash
pip install git+https://github.com/openai/CLIP.git
# or
make install-clip
```

### 2.5 Copy the example environment file

```bash
cp .env.example .env
```

Edit `.env` with any API keys you need (CoinGecko, etc. are optional).

---

## 3. Project Structure Overview

```text
dcass/
├── README.md                   ← Main project description
├── Dockerfile                  ← Container image (Python + PyTorch + FAISS)
├── docker-compose.yml          ← Alice/Bob simulation + training services
├── Makefile                    ← Convenience commands
├── requirements.txt            ← Python dependencies
│
├── document/                   ← All documentation (you are here)
│   ├── GETTING_STARTED.md      ← This file
│   ├── SCRIPTS.md              ← How to run each script
│   ├── DOCKER_SETUP.md         ← Full Docker reference
│   ├── IMPLEMENTATION_SUMMARY.md ← Technical implementation details
│   ├── DCASS_Implementation_Handout.md ← Complete system handout
│   └── diagrams/               ← Architecture & sequence diagrams
│
├── src/                        ← Main source code
│   ├── corpus/                 ← Data loading, embeddings, FAISS indices
│   ├── engine/                 ← Encode / decode core logic
│   ├── stealth/                ← GAN + RL stealth modules
│   │   ├── gan/                ← TemporalPatternGenerator + GANTrainer
│   │   ├── rl/                 ← PPOAgent + StealthEnvironment
│   │   └── stealth_scheduler.py ← Unified scheduler with fallback
│   ├── distribution/           ← Multi-channel dispatcher + noise
│   ├── analysis/               ← Warden (DPI detector) + benchmarks
│   └── cli/                    ← Command-line interface
│
├── scripts/                    ← Runnable scripts
│   ├── run_sender.py           ← Alice (sender): schedule + transmit
│   ├── run_receiver.py         ← Bob (receiver): watch + reassemble
│   ├── train_gan.py            ← Train GAN stealth scheduler
│   ├── train_rl.py             ← Train RL stealth policy (PPO)
│   ├── generate_traffic_data.py ← Synthetic human traffic data
│   └── docker_orchestrate.py   ← Docker pipeline CLI
│
├── data/                       ← Datasets and indices (mostly gitignored)
│   ├── behavioral/             ← Generated human traffic JSON
│   └── indices/                ← FAISS indices (build locally)
│
├── models/                     ← Trained model checkpoints (gitignored)
│   ├── gan/                    ← GAN generator checkpoints
│   └── rl/                     ← RL agent checkpoints
│
├── config/
│   ├── default.yaml            ← All system configuration
│   └── settings.py             ← Config singleton
│
└── tests/                      ← Test suite
```

---

## 4. Quick Start (No Docker)

This runs the pipeline **locally** without building containers. Useful for development.

### Step 1: Build the FAISS indices

Download the Flickr8k dataset and build the image index:

```bash
python scripts/data/download_flickr8k.py
python scripts/data/build_indices.py --modality image
```

> Building indices takes ~5–15 minutes depending on hardware.

### Step 2: Test the encoder

```bash
python -m src.cli.main encode "Meet me at the park at noon"
```

Expected output: a sequence of media IDs (image/text/audio) representing the message.

### Step 3: Test the full encode → distribute pipeline

```bash
python -m src.cli.main distribute "Meet me at the park at noon" casual
```

This encodes the message and dispatches media IDs through the local folder channel (outputs to `phase3_out/`).

### Step 4: Test the stealth scheduler (no checkpoints needed)

The scheduler automatically falls back to static mode when no trained model exists:

```bash
python scripts/runtime/run_sender.py \
  --mode auto \
  --shared-dir ./storage/shared_channel \
  --dry-run
```

You should see:
```
Auto mode: attempting RL scheduling …
[StealthScheduler] RL checkpoint not found — falling back to static
Auto mode: RL unavailable, attempting GAN scheduling …
[StealthScheduler] GAN checkpoint not found — falling back to static
Auto mode: no trained checkpoints found — falling back to STATIC
```

---

## 5. Quick Start (Docker)

### Step 1: Build the Docker images

```bash
docker compose build
# or
make docker-build
```

This builds a Python 3.10 image with PyTorch (CPU), FAISS, Gymnasium, and all DCASS dependencies.

### Step 2: Run Alice (Sender) + Bob (Receiver)

```bash
docker compose up
# or
make docker-send
```

**What happens**:
1. Bob (Receiver) starts, watches `/app/storage/shared_channel`
2. Alice (Sender) starts once Bob is healthy
3. Alice builds a 20-item media sequence
4. Alice uses `StealthScheduler(mode=auto)` → tries RL → tries GAN → falls back to static (since no checkpoints exist yet)
5. Alice writes packet JSON files to the shared volume
6. Bob picks them up, reassembles the sequence after a silence threshold

### Step 3: Watch the logs

```bash
docker compose logs -f dcass-alice    # Sender logs
docker compose logs -f dcass-bob      # Receiver logs
```

### Step 4: Inspect the packets

```bash
ls -la shared_channel/
cat shared_channel/_manifest.json
```

### Step 5: Stop everything

```bash
docker compose down
# or
make docker-clean
```

---

## 6. What Happens Under the Hood

### Dynamic → Static Fallback (The Key Feature)

When the sender runs in `auto` mode, it tries scheduling modes in this order:

```
auto mode
  ├── 1. Try RL  (looks for models/rl/ppo_agent_final.pt)
  │      ↓ not found → logs warning → try next
  ├── 2. Try GAN (looks for models/gan/final.pt)
  │      ↓ not found → logs warning → try next
  └── 3. Static  (NoiseController — always works, no model needed)
```

The `StealthScheduler` (`src/stealth/stealth_scheduler.py`) handles this cascade. Once you train models later (see [SCRIPTS.md](./SCRIPTS.md)), the sender will automatically pick them up.

### Packet Metadata Format

Each transmitted item is written as a JSON file to `shared_channel/`:

```json
{
  "media_id": "media_003",
  "channel_id": 1,
  "sequence_number": 3,
  "delay_seconds": 4.7,
  "timestamp": 1712345678.123,
  "mode_used": "static"
}
```

The `mode_used` field tells you whether static, GAN, or RL scheduling was used.

---

## 7. Configuration

All system parameters live in `config/default.yaml`. Key sections:

```yaml
# Stealth AI — enable when models are trained
stealth:
  gan:
    enabled: false      # Set true after training
  rl:
    enabled: false      # Set true after training

# Distribution timing
distribution:
  base_delay: 3         # Seconds between transmissions
  default_policy: round_robin

# Noise profiles
# Edit src/distribution/profiles.py for custom profiles
```

---

## 8. Environment Variables

Set in `docker-compose.yml` or `.env`:

| Variable | Default | Description |
|---|---|---|
| `DCASS_MODE` | `auto` | Sender mode: `auto` / `rl` / `gan` / `static` |
| `DCASS_PROFILE` | `casual` | Noise profile: `casual` / `stealth` / `bursty` |
| `DCASS_SEQ_LENGTH` | `20` | Number of items in the test sequence |
| `DCASS_BASE_DELAY` | `3.0` | Base inter-item delay (seconds) |
| `DCASS_SHARED_DIR` | `/app/storage/shared_channel` | Packet drop directory |
| `CUDA_VISIBLE_DEVICES` | `""` | Leave empty for CPU mode |
| `GAN_EPOCHS` | `50` | Epochs for GAN training service |
| `RL_EPISODES` | `1000` | Episodes for RL training service |

Override from the command line:

```bash
DCASS_MODE=static DCASS_SEQ_LENGTH=50 docker compose up
```

---

## 9. Troubleshooting

### `ModuleNotFoundError: faiss`

FAISS is only installed inside Docker. For local development:
```bash
pip install faiss-cpu
```

### Receiver never reassembles

The receiver waits for a **silence threshold** (10s by default) after the last packet. If you use `--dry-run` on the sender, packets arrive instantly but the receiver still waits. Either wait 10 seconds or reduce `--timeout`:

```bash
python scripts/runtime/run_receiver.py --watch ./storage/shared_channel --timeout 3
```

### Docker: `shared_channel` volume conflict

```bash
docker compose down --volumes
docker compose up
```

### Slow Docker build

The first build downloads PyTorch (CPU, ~500 MB). Subsequent builds use the cache and are fast.

### GAN / RL checkpoint not found (expected)

If you see this in logs, it's **correct behavior** — the system is falling back to static:
```
[StealthScheduler] RL checkpoint not found — falling back to static
```
Training scripts are in `scripts/training/train_gan.py` and `scripts/training/train_rl.py`. See [SCRIPTS.md](./SCRIPTS.md) for when to run them.

---

## Next Steps

| Goal | Guide |
|---|---|
| Run individual scripts | [SCRIPTS.md](./SCRIPTS.md) |
| Understand the Docker setup | [DOCKER_SETUP.md](./DOCKER_SETUP.md) |
| Read the technical implementation | [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) |
| Deep dive into every module | [DCASS_Implementation_Handout.md](./DCASS_Implementation_Handout.md) |
| Architecture diagrams | [diagrams/](./diagrams/) |

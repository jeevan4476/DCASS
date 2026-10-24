# DCASS Scripts Reference

> All scripts live in the `scripts/` directory. Run them from the **project root** (`dcass/`).  
> For Docker-based execution, see the Docker column in each section.

---

## Table of Contents

1. [Script Inventory](#1-script-inventory)
2. [Simulation Scripts (Alice & Bob)](#2-simulation-scripts-alice--bob)
   - [run_sender.py](#run_senderpy)
   - [run_receiver.py](#run_receiverpy)
   - [docker_orchestrate.py](#docker_orchestratepy)
3. [Training Scripts](#3-training-scripts)
   - [generate_traffic_dataset.py](#generate_traffic_datasetpy)
   - [train_gan.py](#train_ganpy)
   - [train_rl.py](#train_rlpy)
4. [Data Pipeline Scripts](#4-data-pipeline-scripts)
   - [download_flickr8k.py](#download_flickr8kpy)
   - [build_indices.py](#build_indicespy)
5. [Demo & Evaluation Scripts](#5-demo--evaluation-scripts)
   - [evaluate_stealth.py](#evaluate_stealthpy)
   - [test_stealth_system.py](#test_stealth_systempy)
6. [Project Timeline for Scripts](#6-project-timeline-for-scripts)

---

## 1. Script Inventory

| Script | Purpose | When to Run |
|---|---|---|
| `run_sender.py` | Alice — sends media sequence via StealthScheduler | Any time (auto fallback) |
| `run_receiver.py` | Bob — watches shared channel, reassembles packets | Any time alongside sender |
| `docker_orchestrate.py` | Docker pipeline CLI | When using Docker |
| `generate_traffic_dataset.py` | Synthetic human-like traffic JSON (`scripts/stealth/generate_traffic_dataset.py`) | Before GAN training (optional) |
| `train_gan.py` | Train GAN stealth scheduler | Later (Phase 2) |
| `train_rl.py` | Train RL stealth policy (PPO) | Later (Phase 2), after GAN |
| `download_flickr8k.py` | Download Flickr8k dataset | Once, before building indices |
| `build_indices.py` | Build FAISS vector indices | Once, after downloading data |
| `evaluate_stealth.py` | Evaluate warden detection scores | After training |
| `test_stealth_system.py` | Integration test for stealth pipeline | Anytime |

---

## 2. Simulation Scripts (Alice & Bob)

### `run_sender.py`

Alice — uses `StealthScheduler` to produce a timed schedule and writes packet metadata to the shared channel directory.

**Modes**:  *(static is the argparse default)*
- `static` — handcrafted `NoiseController`, no model needed *(default)*
- `auto` *(explicitly requested)* — cascades RL → GAN → static, always succeeds
- `rl` — RL policy, falls back to static if checkpoint missing
- `gan` — GAN generator, falls back to static if checkpoint missing
- `train` — trains the RL agent in-container (`--episodes`, only for this mode)

**Local usage**:
```bash
# Auto mode (dry run — no real delays, good for testing)
python scripts/runtime/run_sender.py \
  --mode auto \
  --shared-dir ./storage/shared_channel \
  --sequence-length 20 \
  --dry-run

# Static only
python scripts/runtime/run_sender.py \
  --mode static \
  --shared-dir ./storage/shared_channel \
  --profile casual \
  --base-delay 5.0

# With a trained RL checkpoint (Phase 2+)
python scripts/runtime/run_sender.py \
  --mode rl \
  --shared-dir ./storage/shared_channel \
  --rl-checkpoint storage/models/rl_agent.pt
```

**Docker usage**:
```bash
# Default: static mode (no checkpoints needed), 20 items
docker compose up dcass-sender dcass-receiver

# Override mode via env var
DCASS_MODE=auto docker compose up

# Override sequence length
DCASS_SEQ_LENGTH=50 docker compose up
```

**Full argument reference**:
```
--shared-dir     PATH    Drop directory for packets        [/app/shared_channel]
--mode           STR     static / auto / rl / gan / train  [static]
--base-delay     FLOAT   Base inter-item delay (seconds, static mode)  [3.0]
--num-channels   INT     Number of distribution channels   [3]
--profile        STR     Noise profile: casual/stealth/burst  [casual]
--gan-checkpoint PATH    Trained GAN .pt file              [None]
--rl-checkpoint  PATH    Trained RL .pt file               [None]
--episodes       INT     Training episodes (--mode train only)  [100]
--sequence-length INT    Media item count when --no-encode is set  [20]
--message        STR     Secret message to encode          ["Meet at the cafe at noon"]
--no-encode              Skip encoding, send dummy media_NNN IDs (sniff test only)
--dry-run                Skip real delays (immediate)
```

**Output**: JSON packet files in `--shared-dir`, plus a `_manifest.json` summarising the session.

---

### `run_receiver.py`

Bob — asynchronously watches a directory for incoming JSON packets, buffers them, and reassembles the media sequence after a silence threshold.

**Local usage**:
```bash
python scripts/runtime/run_receiver.py \
  --watch ./storage/shared_channel \
  --timeout 10 \
  --poll-interval 1.0
```

**Docker usage**:
```bash
docker compose up dcass-receiver
```

**Full argument reference**:
```
--watch          PATH    Directory to watch for packets    [/app/shared_channel]
--timeout        FLOAT   Silence threshold before reassembly (seconds)  [10.0]
--poll-interval  FLOAT   Polling frequency (seconds)       [1.0]
--no-decode              Sniff-only: log reassembled media IDs, skip decoding
--exit-after-decode      Exit after the first successful reassembly+decode
```

**How reassembly works**:
1. Polls `--watch` directory every `--poll-interval` seconds
2. Reads new `.json` packet files
3. Buffers `ReceivedPacket` objects
4. When no new packet arrives for `--timeout` seconds → reassembles by `sequence_number`
5. Passes ordered `media_id` list to decoder (stub in simulation mode)

> **Note**: Receiver runs indefinitely. Send `Ctrl+C` to stop.

---

### `docker_orchestrate.py`

CLI wrapper for running the full Docker pipeline.

```bash
# Default: sender + receiver with auto fallback
python scripts/utils/docker_orchestrate.py

# Or equivalently
python scripts/utils/docker_orchestrate.py --send-only

# Generate synthetic traffic data only
python scripts/utils/docker_orchestrate.py --gen-data

# Full pipeline: gen data → train GAN → train RL → send
python scripts/utils/docker_orchestrate.py --full-pipeline

# Force rebuild images first
python scripts/utils/docker_orchestrate.py --build --send-only

# Stop and remove containers
python scripts/utils/docker_orchestrate.py --cleanup
```

**Full argument reference**:
```
--full-pipeline    Gen data → train GAN → train RL → send
--gen-data         Generate synthetic traffic data only
--train            Gen data + GAN + RL training (no send)
--send-only        Run sender + receiver only (default)
--mode             STR    Sender mode (auto/rl/gan/static)  [auto]
--seq-length       INT    Sequence length                    [20]
--gan-epochs       INT    Epochs for GAN training            [50]
--rl-episodes      INT    Episodes for RL training           [1000]
--build            Force rebuild Docker images first
--cleanup          Stop and remove all containers
```

---

## 3. Training Scripts

> **These are deferred to later in the project timeline.**  
> Run them in order: generate data → train GAN → train RL.

### `generate_traffic_dataset.py`

Generates synthetic human-like social media posting sessions for GAN training. Uses circadian rhythm simulation and bursty behavior patterns.

**Local usage**:
```bash
python scripts/stealth/generate_traffic_dataset.py \
  --num-sessions 2000 \
  --num-channels 3 \
  --output storage/data/behavioral/human_traffic.json
```

The script itself is synthetic-data honest: it prints a note that this
corpus models browsing regimes, not recorded human behaviour. Generated
files are an **alternative** training corpus for the trainers' `--data`
flag — keep them out of `storage/data/traffic/`, which holds the recorded
corpus.

**Docker usage**:
```bash
docker compose --profile training run --rm dcass-gen-traffic
# or
make docker-gen-data
```

**Output format** (`storage/data/behavioral/human_traffic.json`):
```json
[
  {
    "delays": [5.2, 3.1, 12.4],
    "channels": [0, 1, 0],
    "time_of_day": 14
  }
]
```

**Arguments**:
```
--num-sessions   INT    Number of sessions to generate      [2000]
--num-channels   INT    Number of channels to simulate      [3]
--min-len        INT    Shortest session                    [15]
--max-len        INT    Longest session                     [80]
--seed           INT    Random seed for reproducibility     [42]
--output         PATH   Output JSON file path               [storage/data/behavioral/human_traffic.json]
```

---

### `train_gan.py`

Trains the GAN stealth scheduler: `TemporalPatternGenerator` (Generator) vs. `DeepPacketInspectionWarden` (Discriminator).

> **Data**: run `generate_traffic_dataset.py` first, or rely on the synthetic fallback when the default `--data` file (`storage/data/traffic/human_traffic.json`) is missing.

**Local usage**:
```bash
python scripts/stealth/train_gan.py \
  --data storage/data/traffic/human_traffic.json \
  --epochs 20 \
  --batch-size 32 \
  --checkpoint-dir checkpoints/gan
```

**Docker usage**:
```bash
docker compose --profile training run --rm dcass-train-gan
# or override epochs:
GAN_EPOCHS=100 docker compose --profile training run --rm dcass-train-gan
```

WGAN-GP is always on; the escape hatch is `--no-gradient-penalty` (training
is unstable without it). Per-epoch checkpoints accumulate in
`--checkpoint-dir`, but resuming from them is not supported by this CLI.

**Full argument reference**:
```
--data                  PATH    Human traffic JSON                  [storage/data/traffic/human_traffic.json]
--epochs                INT     Training epochs                     [20]
--batch-size            INT     Batch size                          [32]
--latent-dim            INT     Latent dimension                    [128]
--hidden-dim            INT     Hidden dimension                    [256]
--num-channels          INT     Channels to simulate                [3]
--max-seq-len           INT     Max session length                  [100]
--val-fraction          FLOAT   Held-out fraction for Phase D eval  [0.1]
--no-gradient-penalty           Disable WGAN-GP (escape hatch)
--lambda-gp             FLOAT   Gradient-penalty weight             [100.0]
--critic-offset-penalty FLOAT   Penalty on the critic score's offset  [0.0]
--critic-score-bound    FLOAT   Bound on the critic's score (0 = unbounded)  [0.0]
--generator-lr          FLOAT   Generator learning rate             [1e-4]
--warden-lr             FLOAT   Warden learning rate                [2e-4]
--warden-steps          INT     Warden updates per Generator step   [5]
--device                STR     cpu or cuda                         [cuda if available, else cpu]
--checkpoint-dir        PATH    Where to save per-epoch checkpoints [checkpoints/gan]
--out                   PATH    Final checkpoint path               [storage/models/gan_generator.pt]
```

**Outputs**:
- `checkpoints/gan/epoch_{N:03d}.pt` — per-epoch checkpoint (under `--checkpoint-dir`)
- `storage/models/gan_generator.pt` — final checkpoint (`--out`), the path `StealthScheduler` loads by default

**Structure of the final checkpoint** (`--out`):
```python
{
    "generator_state": ...,      # TemporalPatternGenerator weights
    "warden_state": ...,         # DeepPacketInspectionWarden weights — used by train_rl.py
    "generator_optimizer": ...,
    "warden_optimizer": ...,
    "epoch": 20,
    "global_step": ...,
    "config": { ... },
    "metrics_history": ...,
    "critic_bounded": ...,
    "critic_score_bound": ...,
}
```

---

### `train_rl.py`

Trains the PPO RL stealth policy agent inside `StealthEnvironment`.

> **Prerequisite**: run `train_gan.py` first — `--warden-checkpoint` is
> **required** and must point at the Phase-A checkpoint that contains
> `warden_state` (default: `storage/models/gan_generator.pt`).

**Local usage**:
```bash
python scripts/stealth/train_rl.py \
  --episodes 500 \
  --num-channels 3 \
  --out storage/models/rl_agent.pt \
  --warden-checkpoint storage/models/gan_generator.pt
```

**Docker usage**:
```bash
docker compose --profile training run --rm dcass-train-rl
# or override episodes:
RL_EPISODES=2000 docker compose --profile training run --rm dcass-train-rl
```

**Full argument reference**:
```
--episodes                  INT    Training episodes               [500]
--num-channels              INT    Number of channels              [3]
--device                    STR    cpu or cuda                     [cuda if available, else cpu]
--lambda-stealth            FLOAT  Stealth penalty weight (λ)      [50.0]
--lambda-warmup-fraction    FLOAT  Ramp-up fraction for λ          [0.2]
--min-seq-len               INT    Shortest media sequence         [30]
--max-seq-len               INT    Longest media sequence          [60]
--out                       PATH   Final agent checkpoint          [storage/models/rl_agent.pt]
--seed                      INT    Random seed                     [42]
--log-interval              INT    Print stats every N episodes    [10]
--warden-checkpoint         PATH   Phase-A Warden .pt (required)
```

**Output**:
- `storage/models/rl_agent.pt` — checkpoint (`--out`); pass it to
  `run_sender.py --rl-checkpoint` or `evaluate_stealth.py --rl-checkpoint`

---

## 4. Data Pipeline Scripts

### `download_flickr8k.py`

Downloads the Flickr8k image dataset.

```bash
python scripts/data/download_flickr8k.py
```

Files saved to `data/raw/flickr8k/`.

---

### `build_indices.py`

Builds FAISS vector indices from raw dataset files.

```bash
# Build all modalities
python scripts/data/build_indices.py

# Build only image index
python scripts/data/build_indices.py --modality image
make build-image-index

# Build only text index
python scripts/data/build_indices.py --modality text
make build-text-index
```

> Building indices requires CLIP and takes 5–20 minutes on CPU.

---

## 5. Demo & Evaluation Scripts

No `demo_dcass.py` exists; the demos live in `scripts/demo/`:

- `scripts/demo/mentor_live_demo.py` — five-stage live walkthrough (FAISS volume audit, VCP codebook, Reed–Solomon ECC, multi-modal encode/decode, steganalysis proof)
- `scripts/demo/demo_wgan_stealth.py` — interactive WGAN-GP timing-camouflage demo (metronome vs. generator, circadian stage, warden blind test)
- `scripts/demo/run_gan_demo.py` — GAN stealth demo written for a Docker context (`docker run dcass-demo`)

The end-to-end round trip that `encode → schedule → transmit → reassemble → decode` covers is run by `run_pipeline.py` (see QUICKSTART §7), not a demo script. Both standalone demos run directly from the project root.

### `evaluate_stealth.py`

Evaluates the stealth performance of transmission schedules using the Warden.

```bash
python scripts/testing/evaluate_stealth.py
```

### `test_stealth_system.py`

Integration test for the GAN + RL stealth pipeline (no checkpoints needed).

```bash
python scripts/testing/test_stealth_system.py
```

---

## 6. Project Timeline for Scripts

```
Phase 1 — Setup (NOW)
  ✅ docker compose up                       ← sender + receiver (static fallback)
  ✅ run_sender.py --mode auto --dry-run     ← verify dynamic→static cascade
  ✅ build_indices.py                        ← build FAISS after downloading data

Phase 2 — Training (LATER)
  [ ] generate_traffic_dataset.py           ← generate synthetic human traffic
  [ ] train_gan.py                          ← train GAN scheduler
  [ ] train_rl.py --warden-checkpoint ...  ← train RL agent

Phase 3 — Evaluation (FUTURE)
  [ ] evaluate_stealth.py                   ← measure warden scores
  [ ] docker compose up (mode=rl)           ← send with trained RL agent
```

---

## Makefile Quick Reference

```bash
make help               # Show all commands
make docker-build       # Build Docker images
make docker-send        # docker compose up (auto fallback)
make docker-gen-data    # Generate synthetic traffic data
make docker-train       # Gen data + GAN + RL training
make docker-pipeline    # Full pipeline end-to-end
make docker-clean       # Stop containers + clean shared_channel
make test               # Run pytest
make lint               # Ruff check
make format             # Black format
```

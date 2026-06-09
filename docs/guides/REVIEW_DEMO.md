# DCASS Review Demo — End-to-End (Static Scheduler)

> **Audience:** reviewers and team members running the first-review demo.
> **Scope:** full `message → encode → transmit → wire → receive → decode`
> pipeline using only the **static NoiseController scheduler**.
> GAN- and RL-based scheduling are deferred to a later review.

---

## 0. What you are looking at

DCASS encodes a secret message as a sequence of **unmodified** media items
retrieved from a semantic corpus, then distributes them with realistic
inter-transmission delays over multiple channels. Bob's decoder looks each
item up in the same corpus to reconstruct the semantic meaning.

For this review the `StealthScheduler` is pinned to `mode=static`, which
uses a handcrafted `NoiseController` plus activity-profile delays
(`casual` / `stealth` / `burst`). No model checkpoints are required.

### Pipeline at a glance

```
              Alice                                      Bob
  ┌────────────────────────────┐           ┌──────────────────────────────┐
  │  message                   │           │  watch shared_channel/       │
  │   → SemanticChunker        │           │   → reassemble by seq_num    │
  │   → UnifiedSemanticIndex   │           │   → SemanticDecoder          │
  │   → encode_result.media_ids│  shared   │   → reconstructed meaning    │
  │                            │ ─────────▶│                              │
  │  StealthScheduler(static)  │ channel/  │                              │
  │   → delays + channels      │  (JSON)   │                              │
  │   → PacketWriter (JSON)    │           │                              │
  └────────────────────────────┘           └──────────────────────────────┘
```

---

## 1. Prerequisites (once per machine)

| Requirement          | Version | Notes                                     |
| -------------------- | ------- | ----------------------------------------- |
| Python               | 3.10+   | For local / CLI demo                      |
| Node.js              | 20+     | For frontend demo                         |
| Docker + Compose     | 24.0+   | For sender/receiver simulation            |
| FAISS indices built  | —       | `storage/data/indices/{image,text,audio}.*` |

Install Python deps once:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Build the indices once (see `docs/guides/QUICK_START.md`
Option 1 Step 2 for full details):

```bash
python scripts/data/build_indices.py --modality image
python scripts/data/build_indices.py --modality text
# audio is optional for the review demo
```

Verify indices are present:

```bash
ls storage/data/indices/
# image.index  image_metadata.json  text.index  text_metadata.json  ...
```

---

## 2. Three ways to demo the pipeline

Pick one — they all exercise the same static-scheduler pipeline.

| # | Path                     | Best for                          | Time  |
| - | ------------------------ | --------------------------------- | ----- |
| A | **CLI `demo` command**   | Fast, one-shot correctness check  | < 10s |
| B | **Docker Alice ↔ Bob**   | Showing the transport layer live  | ~1m   |
| C | **Frontend UI**          | Visual walkthrough for reviewers  | ~1m   |

> The **recommended flow for the review is C → B → A**:
> show the reviewer the UI first, then drop into the Docker simulation to
> prove the sender/receiver architecture works, then use the CLI for any
> ad-hoc follow-up questions.

---

## Demo A — CLI one-shot (fastest)

Runs encode → decode locally with no networking. Good sanity check before
walking into the review room.

```bash
# End-to-end in one command
python -m src.cli.main demo "Meet at the cafe at noon"

# Or step through explicitly:
python -m src.cli.main encode "Meet at the cafe at noon"
# copy the comma-separated media_ids from the OUTPUT section
python -m src.cli.main decode "<paste_ids_here>"
```

Expected output: `STEP 4: COMPARISON` shows the original message and the
reconstructed semantic meaning side by side, plus a verification rate.

To also demo the **static distribution layer** (NoiseController + Scheduler
+ Dispatcher), use:

```bash
python -m src.cli.main distribute "Meet at the cafe at noon" \
    --profile casual --policy round_robin --seed 42
```

This prints the noise-filtered items, inter-item delays, and actually runs
the `Scheduler` against the `ConsoleChannel` so the reviewer can see the
timing behaviour.

---

## Demo B — Docker Alice ↔ Bob (transport)

This is the most impressive part of the review: two containers, a shared
channel directory, realistic inter-packet delays, and Bob decoding the
message back out.

### B.1 Start the simulation

```bash
docker compose build              # first run only
docker compose up                 # starts Alice (sender) + Bob (receiver)
```

What to expect in the logs:

```
dcass-alice  | DCASS Sender (Alice)
dcass-alice  | Mode: static
dcass-alice  | Encoding message into media sequence...
dcass-alice  | Encoder produced 7 media items: {'image': 4, 'text': 3}
dcass-alice  | TX #1  seq=0  media=flickr_00123  ch=0  delay=2.4s  [static]
dcass-alice  | TX #2  seq=1  media=wiki_00456    ch=1  delay=3.1s  [static]
...
dcass-bob    | [Receiver] Received packet: flickr_00123 (seq=0, channel=0)
dcass-bob    | [Receiver] Received packet: wiki_00456   (seq=1, channel=1)
...
dcass-bob    | [Receiver] Silence threshold reached. Reassembling 7 packets...
dcass-bob    | [Receiver] DECODED MESSAGE
dcass-bob    |  1. [OK ] [image] flickr_00123
dcass-bob    |        "A man meeting a friend at a coffee shop..."
dcass-bob    | ...
dcass-bob    |  Verification rate : 100.0%
dcass-bob    |  Reconstructed     : "A man meeting a friend at a coffee shop | ..."
```

### B.2 Change the message without rebuilding

```bash
DCASS_MESSAGE="The package is under the bridge at midnight" docker compose up
```

Other useful overrides:

| Env var              | Default          | Meaning                         |
| -------------------- | ---------------- | ------------------------------- |
| `DCASS_MESSAGE`      | `Meet at the cafe at noon` | Secret message to encode   |
| `DCASS_MODE`         | `static`         | Keep `static` for this review   |
| `DCASS_PROFILE`      | `casual`         | `casual` / `stealth` / `burst`  |
| `DCASS_BASE_DELAY`   | `3.0`            | Base inter-packet seconds       |
| `RECEIVER_TIMEOUT`   | `10`             | Silence seconds before reassemble |

### B.3 Stop

```bash
docker compose down
```

### B.4 Troubleshooting

| Symptom                                                 | Cause / fix                                                               |
| ------------------------------------------------------- | ------------------------------------------------------------------------- |
| Alice errors `Index file not found`                     | Indices missing under `storage/data/indices/` — run the builder scripts.  |
| Bob prints IDs but never decodes                        | `--no-decode` was passed, or indices not mounted — check container logs.  |
| `Permission denied: /app/shared_channel`                | Run `docker compose down -v` and retry; Docker volume perms stuck.        |
| All packets land in the same second on reassembly       | Normal in fast demo mode — use `DCASS_BASE_DELAY=5` to slow things down.  |

---

## Demo C — Frontend UI (visual)

### C.1 Start backend and frontend

```bash
# Terminal 1 — backend
python -m uvicorn src.api.server:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install        # first time only
npm run dev
```

Backend at `http://localhost:8000`, frontend at `http://localhost:3000`.

Alternatively, the Docker web profile starts both:

```bash
docker compose --profile web up
```

### C.2 Walkthrough

1. **Home** (`/`) — briefly show the navigation and architecture.
2. **Status** (`/status`) — confirm indices loaded, device, and that the
   stealth models section reports no GAN/RL checkpoints (that's the
   *expected* state for this review).
3. **Encode** (`/encode`):
   - Enter a message (e.g. `"The secret meeting is at midnight"`).
   - Leave **Diversity Mode = Best Match** and toggle `image` + `text`.
   - Click **🔐 Encode & Generate Sequence** — point at the semantic chunks
     and per-item similarity scores.
   - Click **📡 Transmit on Wire View**.
4. **Wire View** (`/wire`) — packets appear live with real inter-arrival
   timing. Point at the statistics panel (`mode_used: static`, avg delay,
   channels in use).
5. **Decode** (`/decode`):
   - Click **⇣ Import from Wire View** to pull the IDs Bob saw.
   - Click **🔓 Decode Sequence**.
   - Show the reconstructed meaning, per-item verification, and rate.

### C.3 Why the Decode page matters for the review

The Wire View alone only proves transport. The Decode page closes the loop
and demonstrates that the **same corpus-plus-index pair** recovers the
semantic content on the receiver side — which is the core claim of the
zero-modification steganography approach.

---

## 3. What's deliberately **not** in this review

| Feature              | Status                                      | Shown where                     |
| -------------------- | ------------------------------------------- | ------------------------------- |
| GAN scheduler        | Code present, not loaded                    | `src/stealth/gan/`              |
| RL (PPO) agent       | Code present, not loaded                    | `src/stealth/rl/`               |
| Traffic data gen     | Script present, not run                     | `scripts/training/`             |
| Warden evaluation    | Code present, not run                       | `src/analysis/adversarial/`     |
| Benchmarks suite     | Optional — runs via CLI                     | `python -m src.cli.main benchmark` |

`StealthScheduler` itself still supports `mode=gan` / `mode=rl` / `auto`;
setting `DCASS_MODE` at runtime (or passing `--mode` to the sender) will
switch it on once trained checkpoints land in `storage/models/`.
No rewire needed for the next review — just drop the checkpoints in and
flip the env var.

---

## 4. Quick reference

```bash
# CLI
python -m src.cli.main demo "<message>"
python -m src.cli.main encode "<message>"
python -m src.cli.main decode "<comma,separated,ids>"
python -m src.cli.main distribute "<message>" --profile casual

# Docker Alice ↔ Bob (static mode, real encoder + decoder)
docker compose up
DCASS_MESSAGE="<message>" docker compose up

# Frontend + backend
python -m uvicorn src.api.server:app --reload --port 8000
cd frontend && npm run dev
# or
docker compose --profile web up
```

---

**Last updated:** 2026-04-20 (review rewire — static scheduler only).

# DCASS Project - Complete Implementation Documentation

**Version:** 1.0  
**Date:** April 6, 2026  
**Status:** Production-Ready (Core Features)  
**Overall Completion:** 85%

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Implemented Features](#3-implemented-features)
4. [Technology Stack](#4-technology-stack)
5. [Directory Structure](#5-directory-structure)
6. [Component Reference](#6-component-reference)
7. [API Reference](#7-api-reference)
8. [CLI Reference](#8-cli-reference)
9. [Configuration](#9-configuration)
10. [System Requirements](#10-system-requirements)

---

## 1. Project Overview

### What is DCASS?

**DCASS (Dynamic Context-Aware Semantic Steganography)** is a research-oriented system that enables covert communication without modifying any carrier media. Unlike traditional steganography that embeds data into pixels or audio waveforms, DCASS encodes messages by:

1. **Selecting** semantically aligned, naturally occurring media (text, images, audio)
2. **Distributing** them using human-like behavioral patterns
3. **Never modifying** the media itself (zero-modification steganography)

### Key Innovation

Traditional steganography modifies media files (e.g., LSB embedding in images), which can be detected by statistical analysis. DCASS solves this by:

- Using genuine, publicly-available content
- Making the transmission pattern the covert channel
- Leveraging semantic similarity for message encoding
- Mimicking organic human social media behavior

### System Philosophy

**3-Tier Fallback Architecture:**
```
RL Agent (Optimal) → GAN Generator (Good) → Static NoiseController (Guaranteed)
```

This ensures the system **always works**, even without trained AI models.

---

## 2. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DCASS System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Corpus     │───▶│   Encoder    │───▶│   Stealth    │     │
│  │  Management  │    │              │    │  Scheduler   │     │
│  │              │    │  Chunker     │    │              │     │
│  │  - Flickr8k  │    │  - Semantic  │    │  - RL (PPO)  │     │
│  │  - Flickr30k │    │  - Synonym   │    │  - GAN       │     │
│  │  - Wikipedia │    │  - Context   │    │  - Static    │     │
│  │  - Audio     │    │              │    │              │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │            │
│         │                    │                    ▼            │
│         │                    │            ┌──────────────┐     │
│         │                    │            │ Distribution │     │
│         │                    │            │              │     │
│         │                    │            │  - Channels  │     │
│         │                    │            │  - Dispatch  │     │
│         │                    │            │  - Noise     │     │
│         │                    │            └──────────────┘     │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────┐      │
│  │            Unified Semantic Index (FAISS)            │      │
│  │  - Image embeddings (CLIP, 512-dim)                 │      │
│  │  - Text embeddings (Sentence-Transformers, 384-dim) │      │
│  │  - Audio embeddings (CLAP, 512-dim)                 │      │
│  │  - Score normalization across modalities            │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────┐                        ┌──────────────┐     │
│  │   Decoder    │◀───────────────────────│   Receiver   │     │
│  │              │                        │              │     │
│  │  - Lookup    │                        │  - Buffer    │     │
│  │  - Verify    │                        │  - Reassemble│     │
│  │  - Reconstruct│                        │  - Watch     │     │
│  └──────────────┘                        └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Encoding Flow (Alice)
```
User Message
    │
    ▼
Semantic Chunker ─────────────────┐
    │                             │
    ▼                             │
Query Embeddings                  │
    │                             │
    ▼                             │
FAISS Search                      │
    │                             │
    ▼                             │
Media Sequence [m1, m2, ..., mn]  │
    │                             │
    ▼                             │
Stealth Scheduler ◀───────────────┘
    │
    ▼
Timed Schedule [(m1, d1, c1), (m2, d2, c2), ...]
    │
    ▼
Distribution Channels
    │
    ▼
Transmitted
```

#### Decoding Flow (Bob)
```
Received Packets
    │
    ▼
Reassembly Buffer
    │
    ▼
Sorted Media Sequence [m1, m2, ..., mn]
    │
    ▼
Corpus Lookup (verify each ID)
    │
    ▼
Extract Semantic Content
    │
    ▼
Reconstructed Message
```

---

## 3. Implemented Features

### Phase 1: Corpus Management ✅ 100%

#### Multi-Modal Corpus Support
- **Image Datasets:**
  - Flickr8k: 8,091 images with captions
  - Flickr30k: 31,783 images with captions
  - Embedding: CLIP (ViT-B/32, 512-dim)

- **Text Datasets:**
  - Wikipedia articles (configurable count)
  - Embedding: Sentence-Transformers (all-MiniLM-L6-v2, 384-dim)

- **Audio Datasets:**
  - HuggingFace audio datasets
  - Embedding: CLAP (512-dim)

#### FAISS Indexing
- **UnifiedSemanticIndex** (`src/corpus/index/unified_index.py`)
  - Flat L2 indexing for all modalities
  - Score normalization across modalities
  - Lazy loading for memory efficiency
  - Metadata storage with JSON sidecar files

#### Corpus Loaders
| Loader | File | Lines |
|--------|------|-------|
| Flickr8kLoader | `loaders/flickr8k_loader.py` | 215 |
| Flickr30kLoader | `loaders/flickr30k_loader.py` | 201 |
| WikipediaLoader | `loaders/wikipedia_loader.py` | 189 |
| AudioLoader | `loaders/audio_loader.py` | 267 |

#### Embedders
| Embedder | Model | Dimension |
|----------|-------|-----------|
| CLIPEmbedder | openai/clip-vit-base-patch32 | 512 |
| SentenceTransformerEmbedder | all-MiniLM-L6-v2 | 384 |
| AudioEmbedder | CLAP | 512 |

**Files:** 15 files in `src/corpus/`

---

### Phase 2: Encoding/Decoding Engine ✅ 100%

#### Semantic Chunker
**File:** `src/engine/chunker.py` (417 lines)

**Features:**
- Smart message splitting (word, sentence, character modes)
- Synonym expansion using WordNet
- Context-aware chunking
- Configurable chunk sizes

**Example:**
```python
from src.engine.chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk("Meet at the cafe at noon", max_chunk_size=50)
# Returns: ["Meet at the cafe", "at noon"]
```

#### Semantic Encoder
**File:** `src/engine/encoder.py` (401 lines)

**Features:**
- 3 Diversity Modes:
  - `best`: Highest semantic similarity (best match)
  - `round_robin`: Balanced modality distribution
  - `balanced`: Mix of best and round-robin

- Multi-modal support (image, text, audio)
- Configurable results per chunk
- Duplicate avoidance

**Example:**
```python
from src.engine.encoder import SemanticEncoder

encoder = SemanticEncoder()
encoder.load()

result = encoder.encode(
    "Secret meeting at midnight",
    num_results=5,
    diversity="balanced"
)

for item in result.items:
    print(f"{item.modality}: {item.media_id} (score: {item.score:.3f})")
```

#### Semantic Decoder
**File:** `src/engine/decoder.py` (281 lines)

**Features:**
- Corpus verification (tamper detection)
- Multi-modal content extraction
- Semantic reconstruction
- Verification rate calculation

**Example:**
```python
from src.engine.decoder import SemanticDecoder

decoder = SemanticDecoder()
decoder.load()

media_ids = ["flickr8k_00123", "wiki_00456", "audio_00789"]
result = decoder.decode(media_ids)

print(result.reconstructed_meaning)
print(f"Verified: {result.verification_rate:.0%}")
```

**Files:** 6 files in `src/engine/`

---

### Phase 3: Stealth & Distribution ✅ 95%

#### A. Stealth Scheduler (CRITICAL COMPONENT) ✅ 100%

**File:** `src/stealth/stealth_scheduler.py` (202 lines)

**3-Tier Fallback:**
```python
StealthScheduler.schedule(mode="auto")
    ├─ Try RL (if checkpoint exists)
    ├─ Fallback to GAN (if checkpoint exists)
    └─ Fallback to Static (always available)
```

**Modes:**
- `auto`: Try RL → GAN → Static
- `rl`: RL agent with static fallback
- `gan`: GAN generator with static fallback
- `static`: NoiseController (always works)

**Example:**
```python
from src.stealth.stealth_scheduler import StealthScheduler

scheduler = StealthScheduler(num_channels=3, profile="casual")

schedule = scheduler.schedule(
    media_ids=["m1", "m2", "m3"],
    mode="auto",  # Will use static if no models trained
    base_delay=3.0
)

print(f"Mode used: {schedule['mode_used']}")
print(f"Delays: {schedule['delays']}")
print(f"Channels: {schedule['channels']}")
```

#### B. GAN Generator ✅ 100% (Code Complete)

**File:** `src/stealth/gan/generator.py` (379 lines)

**Architecture:**
```
Latent Noise (128-dim) + Time Embedding (32-dim)
    │
    ▼
Latent Projection (256-dim)
    │
    ▼
GRU (2 layers, 256-dim) [Autoregressive]
    │
    ▼
Multi-Head Self-Attention (8 heads)
    │
    ▼
Output Heads:
    ├─ Delay Head (Softplus) → Positive delays
    ├─ Channel Head (Logits) → Channel selection
    └─ Confidence Head (Sigmoid) → Confidence score [0, 1]
```

**Features:**
- Time-of-day awareness (cyclical encoding)
- Autoregressive sequence generation
- Gumbel-Softmax sampling for differentiable channel selection
- Confidence estimation

**Example:**
```python
import torch
from src.stealth.gan.generator import TemporalPatternGenerator

generator = TemporalPatternGenerator(num_channels=3)

schedule = generator.generate(
    batch_size=1,
    sequence_length=20,
    time_of_day=torch.tensor([14.0])  # 2 PM
)

print(f"Delays: {schedule.delays[0].tolist()}")
print(f"Channels: {schedule.sample_channels()[0].tolist()}")
```

#### C. RL Agent (PPO) ✅ 100% (Code Complete)

**File:** `src/stealth/rl/agent.py` (579 lines)

**Architecture:**
```
State (16-dim: queue, time, channels, history)
    │
    ▼
Shared MLP (256-dim)
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
Delay Head      Channel Head      Value Head
(Gaussian)      (Categorical)     (Critic)
    │                 │                 │
    ▼                 ▼                 ▼
Action: (delay, channel)          V(s)
```

**Features:**
- Proximal Policy Optimization (PPO)
- Actor-Critic architecture
- Continuous delay action (Gaussian distribution)
- Discrete channel action (Categorical distribution)
- Gradient clipping and normalization

**State Space:**
- Queue size (remaining items)
- Time of day (cyclical)
- Channel states (last transmission, rate limits)
- Recent transmission history (avg delay, variance, diversity)

**Reward Function:**
```
R = Throughput - λ * Warden_Score

where:
  Throughput = (items transmitted / time) * 60  (items/minute)
  Warden_Score = Bot probability from DPI Warden [0, 1]
  λ = Stealth penalty coefficient (default: 100.0)
```

#### D. RL Environment ✅ 100%

**File:** `src/stealth/rl/environment.py` (492 lines)

**Gym-Style Environment:**
- State: 16-dimensional vector
- Action: `{delay: float, channel: int}`
- Reward: Throughput - stealth penalty
- Done: Queue empty or timeout

**Features:**
- Channel rate limiting
- Warden evaluation window (last N transmissions)
- Episode timeout (default: 1 hour)
- Transmission history tracking

#### E. Adversarial Warden ✅ 100%

**File:** `src/analysis/adversarial/warden.py` (494 lines)

**Architecture:**
```
Delays + Channels
    │
    ▼
Embedding Layers
    │
    ▼
BiLSTM (2 layers, 256-dim)
    │
    ▼
Transformer Encoder (4 layers, 8 heads)
    │
    ▼
Pooling (mean + max)
    │
    ▼
MLP Classifier
    │
    ▼
Bot Probability [0, 1]
```

**Features:**
- Deep Packet Inspection (DPI) simulation
- BiLSTM for temporal patterns
- Transformer for long-range dependencies
- Confidence calibration

#### F. Distribution Layer ✅ 100%

**Components:**

| Component | File | Purpose |
|-----------|------|---------|
| NoiseController | `distribution/noise.py` | Mathematical jitter injection |
| Dispatcher | `distribution/dispatcher.py` | Multi-channel coordination |
| Scheduler | `distribution/scheduler.py` | Timing management |
| ChannelRegistry | `distribution/channel_registry.py` | Factory pattern for channels |
| Channels | `distribution/channels/` | Output targets |

**Channels Implemented:**
- `ConsoleChannel`: Print to stdout
- `LocalFolderChannel`: Write to filesystem
- `GitHubChannel`: Placeholder (TODO)
- `ImgurChannel`: Placeholder (TODO)

**Activity Profiles:**
```python
ACTIVITY_PROFILES = {
    "casual": {
        "noise_std": 2.0,
        "burst_probability": 0.1,
        "idle_probability": 0.05
    },
    "professional": {
        "noise_std": 1.0,
        "burst_probability": 0.05,
        "idle_probability": 0.1
    },
    "gamer": {
        "noise_std": 3.0,
        "burst_probability": 0.2,
        "idle_probability": 0.05
    },
    "night_owl": {
        "noise_std": 2.5,
        "burst_probability": 0.15,
        "idle_probability": 0.2
    }
}
```

**Files:** 9 files in `src/distribution/`

---

### Phase 4: API & CLI ✅ 100%

#### FastAPI Server

**File:** `src/api/server.py` (242 lines)

**Endpoints:**

```python
GET  /api/health
POST /api/encode
POST /api/decode
POST /api/search
GET  /api/status
GET  /api/benchmark/latest
```

**Features:**
- CORS enabled for all origins
- Lazy loading (encoder/decoder singletons)
- Error handling with HTTP status codes
- Request/response validation with Pydantic

**Example:**
```bash
# Start server
python scripts/runtime/start_server.py --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/encode \
  -H "Content-Type: application/json" \
  -d '{"message": "Secret message", "num_results": 5}'
```

#### CLI Interface

**File:** `src/cli/main.py` (907 lines!)

**Commands:**
```
dcass encode    - Encode a message
dcass decode    - Decode a media sequence
dcass demo      - Run demo
dcass status    - System status
dcass search    - Search corpus
dcass verify    - Verify media sequence
dcass distribute- Distribute media sequence
dcass benchmark - Run benchmarks
```

**Example:**
```bash
# Encode message
python src/cli/main.py encode "Meet at noon" --num-results 5

# Decode sequence
python src/cli/main.py decode --media-ids m1 m2 m3

# System status
python src/cli/main.py status

# Search corpus
python src/cli/main.py search "sunset" --modality image
```

---

### Phase 5: Docker Orchestration ✅ 100%

#### Dockerfile (Optimized)

**Features:**
- Multi-stage build (4 stages)
- CPU-only PyTorch (~1GB smaller)
- Layer caching optimization
- Non-root user support
- Health checks

**Stages:**
1. `base`: System dependencies
2. `builder`: Python packages
3. `production`: Minimal runtime (target)
4. `development`: Dev tools

**Size:** ~2.5GB (optimized from ~4GB+)

#### docker-compose.yml (Optimized)

**Services:**
1. `dcass-sender` (Alice)
2. `dcass-receiver` (Bob)
3. `dcass-gen-traffic` (training profile)
4. `dcass-train-gan` (training profile)
5. `dcass-train-rl` (training profile)
6. `tensorboard` (monitoring profile)

**Features:**
- YAML anchors for DRY config
- Resource limits (2G memory, 2 CPUs)
- Read-only volumes where appropriate
- Build cache optimization
- Named volumes for persistence

#### Scripts

| Script | File | Purpose | Lines |
|--------|------|---------|-------|
| Sender | `scripts/runtime/run_sender.py` | Alice's transmission logic | 383 |
| Receiver | `scripts/runtime/run_receiver.py` | Bob's reassembly logic | 296 |
| Orchestrator | `scripts/utils/docker_orchestrate.py` | Container management | 136 |

**Example:**
```bash
# Start simulation
docker compose up

# Train models
docker compose --profile training run dcass-gen-traffic
docker compose --profile training run dcass-train-gan
docker compose --profile training run dcass-train-rl

# Monitor
docker compose --profile monitoring up tensorboard
```

---

### Phase 6: Frontend ✅ 85%

#### Technology Stack
- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **HTTP Client:** Axios
- **State:** React Hooks

#### Pages

| Page | Route | Status | Purpose |
|------|-------|--------|---------|
| Home | `/` | ✅ Complete | Landing page, navigation |
| Status | `/status` | ✅ Complete | System health dashboard |
| Encode | `/encode` | ✅ Complete | Alice's encoding interface |
| Wire | `/wire` | ⚠️ 80% | Real-time packet feed |
| Decode | `/decode` | ❌ Pending | Bob's decoding interface |

#### Components

**File:** `frontend/src/components/ui.tsx`

```typescript
// UI Components
- Card
- StatCard
- Badge
- LoadingSpinner
```

**File:** `frontend/src/components/Navigation.tsx`

```typescript
// Navigation Component
- Unified nav bar
- Active route highlighting
- Responsive design
```

#### Features
- Dark mode UI (deep gray/black backgrounds)
- Neon accent colors (cyan primary, green success, red error)
- Responsive grid layouts
- Real-time status polling (10s interval)
- Error handling with user-friendly messages
- Loading states

**Files:** ~10 files in `frontend/src/`

---

## 4. Technology Stack

### Backend

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| ML Framework | PyTorch | 2.0+ |
| Embeddings | Sentence-Transformers | 2.2+ |
| Image Embeddings | CLIP | Latest |
| Audio Embeddings | CLAP | Latest |
| Vector DB | FAISS | 1.7.4+ |
| API Framework | FastAPI | Latest |
| CLI Framework | Typer | 0.9+ |
| HTTP Client | Requests | 2.31+ |
| NLP | NLTK | 3.8+ |

### Frontend

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | Next.js | 14 |
| Language | TypeScript | 5+ |
| Styling | Tailwind CSS | 3+ |
| HTTP Client | Axios | Latest |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker |
| Orchestration | docker-compose |
| Monitoring | TensorBoard |

---

## 5. Directory Structure

```
dcass/
├── audio/                      # Audio corpus data
├── config/                     # Configuration files
│   └── default_config.yaml
├── data/                       # Data storage
│   ├── audio/                  # Audio datasets
│   ├── behavioral/             # Traffic data for training
│   ├── benchmarks/             # Benchmark results
│   ├── cache/                  # Cache files
│   ├── indices/                # FAISS indices
│   │   ├── audio.index
│   │   ├── image.index
│   │   └── text.index
│   ├── processed/              # Processed data
│   └── raw/                    # Raw datasets
├── docs/                       # Technical documentation
│   ├── architecture.md
│   ├── class_diagram.md
│   ├── er_diagram.md
│   └── sequence_*.md
├── document/                   # User documentation
│   ├── DCASS_Implementation_Handout.md
│   ├── DOCKER_SETUP.md
│   ├── GETTING_STARTED.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── INCOMPLETE_TASKS_PRD.md
│   ├── PROJECT_COMPLETION_STATUS.md
│   ├── QUICK_START.md
│   ├── README.md
│   ├── SCRIPTS.md
│   └── UIUX_BASE_REQUIREMENTS.md
├── frontend/                   # Next.js frontend
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   │   ├── decode/        # ❌ Not implemented
│   │   │   ├── encode/        # ✅ Complete
│   │   │   ├── status/        # ✅ Complete
│   │   │   ├── wire/          # ⚠️ 80% complete
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── Navigation.tsx
│   │   │   └── ui.tsx
│   │   └── lib/
│   │       └── api.ts
│   ├── package.json
│   └── tsconfig.json
├── logs/                       # Log files
├── models/                     # Trained model checkpoints
│   ├── gan/                    # ❌ Empty (not trained)
│   │   └── final.pt
│   └── rl/                     # ❌ Empty (not trained)
│       └── ppo_agent_final.pt
├── scripts/                    # Runnable scripts (23 files)
│   ├── build_indices.py
│   ├── demo_dcass.py
│   ├── demo_encoder.py
│   ├── demo_full_loop.py
│   ├── docker_orchestrate.py
│   ├── download_flickr8k.py
│   ├── download_flickr30k.py
│   ├── download_wikipedia.py
│   ├── generate_traffic_data.py
│   ├── run_receiver.py
│   ├── run_sender.py
│   ├── start_server.py
│   ├── test_encoding.py
│   ├── test_stealth_system.py
│   ├── train_gan.py
│   └── train_rl.py
├── shared_channel/             # Docker simulation channel
├── src/                        # Main source code
│   ├── analysis/               # Analysis tools
│   │   └── adversarial/
│   │       └── warden.py       # DPI Warden (494 lines)
│   ├── api/                    # FastAPI server
│   │   └── server.py           # API endpoints (242 lines)
│   ├── cli/                    # CLI interface
│   │   └── main.py             # CLI commands (907 lines!)
│   ├── corpus/                 # Corpus management (15 files)
│   │   ├── embedders/
│   │   │   ├── audio_embedder.py
│   │   │   ├── clip_embedder.py
│   │   │   └── sentence_embedder.py
│   │   ├── index/
│   │   │   └── unified_index.py
│   │   └── loaders/
│   │       ├── audio_loader.py
│   │       ├── flickr8k_loader.py
│   │       ├── flickr30k_loader.py
│   │       └── wikipedia_loader.py
│   ├── distribution/           # Distribution layer (9 files)
│   │   ├── channels/
│   │   │   ├── console.py
│   │   │   ├── github.py       # Placeholder
│   │   │   ├── imgur.py        # Placeholder
│   │   │   └── local_folder.py
│   │   ├── channel_registry.py
│   │   ├── dispatcher.py
│   │   ├── noise.py            # NoiseController
│   │   ├── profiles.py
│   │   └── scheduler.py
│   ├── engine/                 # Encoding/decoding (6 files)
│   │   ├── chunker.py          # SemanticChunker (417 lines)
│   │   ├── decoder.py          # SemanticDecoder (281 lines)
│   │   └── encoder.py          # SemanticEncoder (401 lines)
│   └── stealth/                # Stealth components
│       ├── gan/
│       │   ├── generator.py    # GAN Generator (379 lines)
│       │   └── trainer.py      # GANTrainer (497 lines)
│       ├── rl/
│       │   ├── agent.py        # PPOAgent (579 lines)
│       │   └── environment.py  # StealthEnvironment (492 lines)
│       └── stealth_scheduler.py # StealthScheduler (202 lines)
├── tests/                      # Test files
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

**Total:** ~24,600+ lines of production code

---

## 6. Component Reference

### Core Components

#### UnifiedSemanticIndex

**Location:** `src/corpus/index/unified_index.py`

**Purpose:** Multi-modal FAISS index manager

**Methods:**
```python
load(modalities: list[Modality]) -> dict[str, bool]
search(query: str, modality: Modality, k: int) -> list[SearchResult]
get_by_id(media_id: str) -> Optional[MediaItem]
add_items(items: list[MediaItem], modality: Modality)
status() -> dict
```

**Example:**
```python
from src.corpus.index.unified_index import UnifiedSemanticIndex

index = UnifiedSemanticIndex(base_path="data/indices")
status = index.load(modalities=["image", "text"])

results = index.search("sunset beach", modality="image", k=5)
for result in results:
    print(f"{result.media_id}: {result.score:.3f}")
```

#### SemanticChunker

**Location:** `src/engine/chunker.py`

**Purpose:** Smart message splitting with synonym expansion

**Methods:**
```python
chunk(message: str, max_chunk_size: int = 50, mode: str = "smart") -> list[str]
expand_with_synonyms(text: str, max_synonyms: int = 3) -> list[str]
```

**Modes:**
- `smart`: Sentence-aware splitting
- `word`: Split by words
- `char`: Split by characters

#### SemanticEncoder

**Location:** `src/engine/encoder.py`

**Purpose:** Message → Media sequence encoding

**Methods:**
```python
load(modalities: list[Modality] = None) -> dict[str, bool]
encode(message: str, num_results: int = 3, diversity: str = "balanced") -> EncodingResult
```

**Diversity Modes:**
- `best`: Highest semantic similarity
- `round_robin`: Balanced modality distribution
- `balanced`: Mix of best and round-robin

#### SemanticDecoder

**Location:** `src/engine/decoder.py`

**Purpose:** Media sequence → Message reconstruction

**Methods:**
```python
load(modalities: list[Modality] = None) -> dict[str, bool]
decode(media_ids: list[str]) -> DecodingResult
decode_to_text(media_ids: list[str]) -> str
verify_sequence(media_ids: list[str]) -> tuple[bool, float]
```

#### StealthScheduler

**Location:** `src/stealth/stealth_scheduler.py`

**Purpose:** Unified scheduler with 3-tier fallback

**Methods:**
```python
schedule(
    media_ids: list[str],
    mode: Literal["static", "gan", "rl"] = "static",
    base_delay: float = 3.0,
    gan_checkpoint: Optional[Path] = None,
    rl_checkpoint: Optional[Path] = None
) -> dict
```

**Returns:**
```python
{
    "items": list[str],      # Media IDs (after noise)
    "delays": list[float],   # Inter-item delays (seconds)
    "channels": list[int],   # Channel index per item
    "mode_used": str         # Actual mode used
}
```

---

## 7. API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### GET /api/health
**Description:** Health check

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-06T12:00:00Z"
}
```

#### POST /api/encode
**Description:** Encode a message into media sequence

**Request:**
```json
{
  "message": "Secret meeting at noon",
  "num_results": 5,
  "diversity": "balanced",
  "modalities": ["image", "text"]
}
```

**Response:**
```json
{
  "message": "Secret meeting at noon",
  "items": [
    {
      "media_id": "flickr8k_00123",
      "modality": "image",
      "score": 0.856,
      "content": "People meeting at cafe"
    },
    ...
  ],
  "chunks": ["Secret meeting", "at noon"],
  "total_items": 10
}
```

#### POST /api/decode
**Description:** Decode media sequence back to message

**Request:**
```json
{
  "media_ids": ["flickr8k_00123", "wiki_00456", "audio_00789"]
}
```

**Response:**
```json
{
  "media_ids": ["flickr8k_00123", "wiki_00456", "audio_00789"],
  "decoded": [
    {
      "media_id": "flickr8k_00123",
      "modality": "image",
      "content": "People meeting at cafe",
      "verified": true
    },
    ...
  ],
  "reconstructed_meaning": "People meeting at cafe | Article about noon | ...",
  "verification_rate": 1.0
}
```

#### POST /api/search
**Description:** Search corpus by query

**Request:**
```json
{
  "query": "sunset beach",
  "modality": "image",
  "k": 10
}
```

**Response:**
```json
{
  "query": "sunset beach",
  "modality": "image",
  "results": [
    {
      "media_id": "flickr8k_00123",
      "score": 0.92,
      "content": "Beautiful sunset over ocean",
      "metadata": {...}
    },
    ...
  ],
  "count": 10
}
```

#### GET /api/status
**Description:** System status

**Response:**
```json
{
  "indices": {
    "image": {"loaded": true, "count": 8091},
    "text": {"loaded": true, "count": 5000},
    "audio": {"loaded": false, "count": 0}
  },
  "models": {
    "gan": {"exists": false, "path": "models/gan/final.pt"},
    "rl": {"exists": false, "path": "models/rl/ppo_agent_final.pt"}
  },
  "device": "cpu",
  "encoder_loaded": true,
  "decoder_loaded": true
}
```

#### GET /api/benchmark/latest
**Description:** Latest benchmark results

**Response:**
```json
{
  "timestamp": "2026-04-06T12:00:00Z",
  "encoding_time": 1.234,
  "decoding_time": 0.567,
  "throughput": 45.2
}
```

---

## 8. CLI Reference

### Installation

```bash
# Add to PATH or use full path
export PATH="$PWD:$PATH"

# Or use python -m
python -m src.cli.main <command>
```

### Commands

#### encode
```bash
python src/cli/main.py encode "Secret message" \
    --num-results 5 \
    --diversity balanced \
    --modalities image text \
    --output encoded.json
```

**Options:**
- `--num-results, -n`: Results per chunk (default: 3)
- `--diversity, -d`: Diversity mode (best|round_robin|balanced)
- `--modalities, -m`: Modalities to use (image|text|audio)
- `--output, -o`: Output file path

#### decode
```bash
python src/cli/main.py decode \
    --media-ids m1 m2 m3 \
    --verify
```

**Options:**
- `--media-ids`: Space-separated media IDs
- `--input, -i`: Input file with media IDs
- `--verify`: Verify corpus integrity

#### demo
```bash
python src/cli/main.py demo \
    --message "Test message" \
    --num-results 5
```

#### status
```bash
python src/cli/main.py status
```

**Output:**
```
System Status
═════════════
Indices:
  image: ✓ (8091 items)
  text:  ✓ (5000 items)
  audio: ✗ (not loaded)

Models:
  GAN:  ✗ (not trained)
  RL:   ✗ (not trained)

Encoder: ✓
Decoder: ✓
Device:  cpu
```

#### search
```bash
python src/cli/main.py search "sunset" \
    --modality image \
    --k 10
```

#### verify
```bash
python src/cli/main.py verify \
    --media-ids m1 m2 m3
```

#### distribute
```bash
python src/cli/main.py distribute \
    --media-ids m1 m2 m3 \
    --channels console folder \
    --base-delay 3.0 \
    --profile casual
```

#### benchmark
```bash
python src/cli/main.py benchmark \
    --num-messages 100 \
    --output benchmark_results.json
```

---

## 9. Configuration

### Environment Variables

```bash
# .env file
DCASS_MODE=auto                    # Scheduler mode (auto|rl|gan|static)
DCASS_PROFILE=casual               # Activity profile
DCASS_BASE_DELAY=3.0               # Base delay (seconds)
DCASS_SEQ_LENGTH=20                # Sequence length
DCASS_SHARED_DIR=./shared_channel  # Shared directory
NUM_CHANNELS=3                     # Number of channels
GAN_EPOCHS=50                      # GAN training epochs
RL_EPISODES=1000                   # RL training episodes
```

### Configuration File

**Location:** `config/default_config.yaml`

```yaml
corpus:
  base_path: "data"
  indices_path: "data/indices"
  modalities:
    - image
    - text
    - audio

encoder:
  num_results: 3
  diversity: "balanced"
  max_chunk_size: 50

decoder:
  verify: true

stealth:
  mode: "auto"
  profile: "casual"
  base_delay: 3.0

distribution:
  num_channels: 3
  channels:
    - console
    - local_folder
```

---

## 10. System Requirements

### Minimum Requirements

- **OS:** Linux, macOS, or Windows 10+
- **Python:** 3.10 or higher
- **RAM:** 8 GB
- **Storage:** 20 GB (for datasets)
- **CPU:** 4 cores

### Recommended Requirements

- **OS:** Linux (Ubuntu 20.04+)
- **Python:** 3.10
- **RAM:** 16 GB
- **Storage:** 50 GB SSD
- **CPU:** 8 cores
- **GPU:** Optional (CUDA-capable for faster training)

### Dependencies

See `requirements.txt` for full list.

**Key Dependencies:**
- PyTorch 2.0+
- FAISS 1.7.4+
- Sentence-Transformers 2.2+
- FastAPI
- Typer
- Rich

---

*Documentation generated: April 6, 2026*  
*Version: 1.0*  
*For questions, see README.md or contact the development team*

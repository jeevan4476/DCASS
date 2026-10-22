# DCASS: Dynamic Context-Aware Semantic Steganography

Zero-modification information hiding through multimodal media curation and behavioral traffic scheduling.

---

## Overview

DCASS is a research prototype for zero-modification semantic steganography. Conventional steganographic methods hide secret data by modifying media bitstreams, such as altering least significant bits (LSB) in images or perturbing transform coefficients in audio. These modifications leave statistical footprints detectable by modern steganalysis classifiers.

DCASS eliminates carrier distortion entirely. Instead of altering existing media, the system encodes messages into semantic vector sequences and retrieves unaltered public media (images, text passages, and audio clips) from a corpus. Receivers reconstruct the message by extracting semantic embeddings from the received carriers and decoding the trajectory using a shared, dynamically keyed codebook.

To attempt to protect against statistical traffic analysis, transmissions are scheduled using behavioral models:
- A Wasserstein GAN with Gradient Penalty (WGAN-GP) generates inter-arrival delays intended to resemble human activity. The shipped checkpoint's circadian adaptation is not demonstrated in the recorded benchmark.
- A Proximal Policy Optimization (PPO) reinforcement learning agent is designed to balance covert throughput against channel exposure and channel cool-down constraints.

---

## System Architecture

The architecture comprises four coordinated components:

1. **Corpus and Multimodal Indexing**
   - High-dimensional semantic embeddings via OpenAI CLIP (ViT-B/32) for images and text, and CLAP for audio.
   - Unified vector search with FAISS across 245,279 indexed corpus items.
   - Voronoi codebook partitioning with certified centroid distributions across 6 semantic families.

2. **Semantic Codec Engine**
   - **Dynamic Spherical Semantic Coding (DSSC):** Maps framed byte streams onto high-dimensional Voronoi centroids, achieving 13.50–13.66 coded bits per carrier (mean 13.60), including framing and Reed-Solomon parity. User-payload capacity is lower; see the benchmark table below.
   - **Exact Voronoi Codebook Partitioning (VCP):** Deterministic 1 byte per carrier baseline.
   - **Cryptographic Keying:** HMAC-SHA256 session key derivation with pseudo-random permutations to prevent static mapping attacks.
   - **Error Correction:** Reed-Solomon (RS) forward error correction with CRC-16 payload verification.

3. **Stealth and Traffic Scheduling**
   - **WGAN-GP Traffic Generator:** Produces non-deterministic packet delays intended to resemble human activity. The shipped checkpoint does not demonstrate human-like timing.
   - **PPO Reinforcement Learning Agent:** Schedules packets across multi-channel endpoints while respecting warden rate limits.
   - **Multi-Channel Dispatcher:** Simulates asynchronous delivery across independent communication platforms.

4. **Adversarial Evaluation Framework**
   - Content steganalysis testbed covering SRNet, ZhuNet, and YeNet (image-only; three gated detectors).
   - Distributional drift analysis using energy distance, MMD², Wasserstein-1 projections, and Kolmogorov-Smirnov tests.
   - Deep Packet Inspection (DPI) warden simulations with stateful threshold detection. The soft branch is a trained traffic detector whose threshold is calibrated on held-out human traffic, and it flags the WGAN and Hybrid schedules at 100% while the retrained PPO arm is not flagged; the human side of that comparison is a synthetic corpus, so the current traffic result is negative/limited, not demonstrated evasion.

---

## Empirical Benchmark Summary

These are recorded results, not targets. Each row states its scope and the reason it should or should not be read as a security claim. The full ledger is in [`docs/benchmarking/CLAIM_LEDGER.md`](docs/benchmarking/CLAIM_LEDGER.md).

| Evaluation Component | Recorded Result | Scope and Caveat |
|---|---|---|
| **Media Steganalysis** | AUC **0.4973–0.4993** for SRNet, ZhuNet, and YeNet; every 95% CI covers 0.5000 | Image-only, three gated detectors. This is a negative control for zero pixel modification, not proof of semantic or traffic security. StegExpose, Xu-Net, and classical SRM are not part of the recorded C1 claim. |
| **Selection Bias** | **0/1,536** KS coordinates reject after Benjamini-Hochberg control; image and text energy/MMD shifts are small but significant (p ≈ 0.004–0.012); audio p ≈ 0.57 | The uniform null is rejected (chi-square 3,253.4, p = 0.0005). The pipeline is consistent with its own mechanism null (p = 0.208), not with uniform sampling. |
| **Traffic Timing** | WGAN and Hybrid: **50/50** interceptions each (95% CI [92.9%, 100%]); Poisson 27/50 (54%, 95% CI [39.3%, 68.2%]); static 50/50; retrained PPO **0/50** (0%, 95% CI [0%, 7.1%]) | **Not evidence of evasion.** The soft branch is a trained traffic detector whose flag threshold is calibrated on held-out human traffic at a measured 4.60% false-positive rate, and the measurement uses a detector held out from RL training, so these are measurements. The PPO arm also respects the platform's per-channel cooldowns, which the baselines do not. The human baseline is the repository's own synthetic traffic corpus, not recorded human browsing. |
| **Codec Capacity** | DSSC coded **13.50–13.66 bpc** (mean 13.60) versus exact_vcp 8.00 coded bpc | "Coded" includes framing and Reed-Solomon parity. User-payload capacity is **7.00 bpc at 14 B** and **12.90 bpc at 500 B**. Carrier reduction is 40.7–41.4% multi-modal and 48.1–49.9% text-only. |
| **Concurrency** | Direct engine: 50 worker threads, 100 sessions, **100% decode success**, **0% observed swaps**, p50 **24,367 ms** | Direct `SemanticEngine.encode/decode` only. `/api/transmit` now admits concurrent sessions up to a bounded limit and answers HTTP 429 beyond it, so this is still not an API concurrency result. |
| **Reliability** | 100% clean-channel round trip; zero false accepts after decoder hardening; **0% exact recovery under ≥2% carrier dropout** | Dropout is unrecoverable in the current wire format. The pre-hardening false-accept artifact is archived. |

Full benchmark methodologies, LaTeX tables, and empirical figures are detailed in the [Benchmarking Report](docs/benchmarking/BENCHMARKING_REPORT.md).

---

## Repository Layout

```text
DCASS/
├── src/                     # Core Python modules
│   ├── api/                 # FastAPI backend, authentication, session routing
│   ├── engine/              # DSSC/VCP encoders, decoders, codebooks, and ECC
│   ├── stealth/             # WGAN-GP generator, PPO RL agent, stealth scheduler
│   ├── corpus/              # Multimodal feature extractors, loaders, and FAISS indices
│   ├── distribution/        # Multi-channel packet transport and channel simulators
│   └── analysis/            # Adversarial warden classifiers and benchmark harnesses
├── frontend/                # Next.js 14 web application (TypeScript, Tailwind CSS)
├── scripts/                 # Operational and benchmarking scripts
│   ├── runtime/             # Service runners and entry points
│   ├── stealth/             # Model training (WGAN-GP, PPO) and warden evaluation
│   ├── testing/             # System benchmark and validation suites
│   ├── data/                # Dataset download and FAISS index construction
│   └── demo/                # Standalone demonstration scripts
├── docs/                    # System documentation and academic papers
│   ├── benchmarking/        # Empirical results, methodologies, and paper drafts
│   ├── architecture/        # Mathematical specifications (DSSC, VCP, HMAC)
│   ├── research/            # Threat models and theoretical foundations
│   └── guides/              # Setup, operations, and development manuals
├── storage/                 # Data, models, and shared channel directory (gitignored)
│   ├── models/              # Checkpoints for WGAN-GP, RL agent, and Warden
│   └── benchmarks/          # Raw evaluation logs, generated plots, and LaTeX tables
├── tests/                   # Pytest test suites (unit, integration, pipeline)
├── demo_dcass.py            # Command-line demonstration entry point
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Multi-service simulation and UI configuration
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Project metadata and configuration
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher with npm
- CUDA-compatible GPU (recommended for FAISS and PyTorch, CPU fallback supported)
- Git

### Virtual Environment Setup

```bash
# Clone repository
git clone https://github.com/jeevan4476/DCASS.git
cd DCASS

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Environment Configuration

```bash
cp .env.example .env
```

Key environment variables:
- `DCASS_DEVICE`: `cuda` or `cpu` (default: auto-detected)
- `DCASS_CORS_ORIGINS`: `http://localhost:3000`
- `NEXT_PUBLIC_API_URL`: `http://localhost:8000`

---

## Quickstart

### 1. Run the End-to-End CLI Demonstration

Execute a complete encode, schedule, transmit, reassemble, and decode cycle:

```bash
python demo_dcass.py \
  --message "Confidential research verification" \
  --sender alice \
  --recipient bob \
  --codec dssc \
  --speed-multiplier 100
```

### 2. Start the Backend API

Launch the FastAPI backend service:

```bash
make server          # warmed engine, no auto-reload, port 8000
# or: make server-dev   (auto-reload for development)
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 3. Start the Web Dashboard

In a separate terminal, launch the Next.js interface:

```bash
make frontend        # production build, served on port 3001
# or: make frontend-dev   (hot-reload dev server on port 3000)
```

Open `http://localhost:3001` in your browser. Default test accounts are `alice` / `alice12345`, `bob` / `bob12345`, and `charlie` / `charlie12345`. These are local demo credentials only; the database may be reset.

`make stack` starts both services in the background (logs under `storage/run/`; stop with `make stop`, follow with `make logs`).

### 4. Run Test and Benchmark Suites

Verify pipeline functionality and run stealth benchmarks:

```bash
# Run unit and integration tests
pytest tests/ -v

# Run the full stealth and timing evaluation suite
python scripts/testing/evaluate_stealth.py
```

---

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:

### User and Operational Guides
- [System Quickstart](docs/QUICKSTART.md): Step-by-step setup for local and containerized deployments.
- [Getting Started Guide](docs/guides/GETTING_STARTED.md): Initial configuration and index verification.
- [Script Execution Reference](docs/guides/SCRIPT_EXECUTION_GUIDE.md): Running training, evaluation, and operational scripts.
- [Docker Deployment Guide](docs/guides/DOCKER_SETUP.md): Multi-container orchestration instructions.

### Research and Benchmarks
- [Full Benchmarking Report](docs/benchmarking/BENCHMARKING_REPORT.md): Complete empirical findings across all 5 benchmark components.
- [Academic Conference Paper Writeup](docs/benchmarking/BENCHMARK_PAPER_WRITEUP.md): Formal research paper structure and methodology.
- [Core Theoretical Concept](docs/research/IDEA.md): Mathematical principles of zero-modification curation.
- [Semantic Steganography Literature Review](docs/research/RESEARCH_PAPER_SEMANTIC_STEGANOGRAPHY.md): Prior work and theoretical positioning.

### Architecture and Specifications
- [DSSC Mathematical Specification](docs/architecture/DSSC_THEORY_AND_MATHEMATICAL_SPECIFICATION.md): Modality-conditioned coding theory.
- [VCP Mathematical Specification](docs/architecture/VCP_THEORY_AND_MATHEMATICAL_SPECIFICATION.md): Voronoi partitioning fundamentals.
- [Cryptographic Session Key Derivation](docs/architecture/HMAC_SHA256_SESSION_KEY_DERIVATION.md): Forward secrecy and keyed permutation proofs.
- [Complete Implementation Architecture](docs/guides/COMPLETE_IMPLEMENTATION_GUIDE.md): Internal data flows and module boundaries.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.

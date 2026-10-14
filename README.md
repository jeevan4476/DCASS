# DCASS: Dynamic Context-Aware Semantic Steganography

Zero-modification information hiding through multimodal media curation and behavioral traffic scheduling.

---

## Overview

DCASS is a research prototype for zero-modification semantic steganography. Conventional steganographic methods hide secret data by modifying media bitstreams, such as altering least significant bits (LSB) in images or perturbing transform coefficients in audio. These modifications leave statistical footprints detectable by modern steganalysis classifiers.

DCASS eliminates carrier distortion entirely. Instead of altering existing media, the system encodes messages into semantic vector sequences and retrieves unaltered public media (images, text passages, and audio clips) from a corpus. Receivers reconstruct the message by extracting semantic embeddings from the received carriers and decoding the trajectory using a shared, dynamically keyed codebook.

To protect against statistical traffic analysis, transmissions are scheduled using behavioral models:
- A Wasserstein GAN with Gradient Penalty (WGAN-GP) generates human-like inter-arrival delays and models circadian diurnal patterns.
- A Proximal Policy Optimization (PPO) reinforcement learning agent dynamically balances covert throughput against channel exposure and channel cool-down constraints.

---

## System Architecture

The architecture comprises four coordinated components:

1. **Corpus and Multimodal Indexing**
   - High-dimensional semantic embeddings via OpenAI CLIP (ViT-B/32) for images and text, and CLAP for audio.
   - Unified vector search with FAISS across 245,000+ corpus items.
   - Voronoi codebook partitioning with certified centroid distributions across 6 semantic families.

2. **Semantic Codec Engine**
   - **Dynamic Spherical Semantic Coding (DSSC):** Maps framed byte streams onto high-dimensional Voronoi centroids, achieving up to 13.67 bits per carrier.
   - **Exact Voronoi Codebook Partitioning (VCP):** Deterministic 1 byte per carrier baseline.
   - **Cryptographic Keying:** HMAC-SHA256 session key derivation with pseudo-random permutations to prevent static mapping attacks.
   - **Error Correction:** Reed-Solomon (RS) forward error correction with CRC-16 payload verification.

3. **Stealth and Traffic Scheduling**
   - **WGAN-GP Traffic Generator:** Produces non-deterministic packet delays matching human activity patterns.
   - **PPO Reinforcement Learning Agent:** Schedules packets across multi-channel endpoints while avoiding warden rate limits.
   - **Multi-Channel Dispatcher:** Simulates asynchronous delivery across independent communication platforms.

4. **Adversarial Evaluation Framework**
   - Content steganalysis testbed covering SRNet, YeNet, ZhuNet, and StegExpose.
   - Distributional drift analysis using Wasserstein-1 metrics and Kolmogorov-Smirnov statistical tests.
   - Deep Packet Inspection (DPI) warden simulations with stateful threshold detection.

---

## Empirical Benchmark Summary

Evaluations conducted on the full system demonstrate resistance to content-based steganalysis and high evasion against timing analysis:

| Evaluation Component | Metric | Baseline / Target | DCASS Result | Finding |
|---|---|---|---|---|
| **Media Steganalysis** | Detection AUC | 0.5000 (Random Guess) | **0.5000** | Zero carrier distortion yields perfect media indistinguishability across SRNet, YeNet, ZhuNet, and StegExpose. |
| **Selection Bias** | KS Test Rejection Rate | < 5.0% (alpha = 0.05) | **2.6%** | Carrier selection distributions are statistically indistinguishable from uniform sampling across 783 dimensions. |
| **Traffic Timing Evasion** | Interception Rate | 0.0% (Full Evasion) | **0.0%** (WGAN / Hybrid) | Zero detection against trained DPI Warden; static schedules are intercepted at 100%. |
| **Codec Efficiency** | Bits per Carrier | 8.00 bpc (VCP) | **13.67 bpc** (DSSC) | 70.9% increase in throughput; 40% to 66.7% reduction in required carrier counts. |
| **Concurrency and Scaling** | Cross-Talk Rate | 0.0% | **0.000%** | Strict session isolation confirmed across 50 concurrent client threads at 100% request success. |

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
uvicorn src.api.server:app --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 3. Start the Web Dashboard

In a separate terminal, launch the Next.js interface:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. Default test accounts include `alice` (password: `alice123`) and `bob` (password: `bob123`).

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

This project is licensed under the MIT License. See the LICENSE file for details.

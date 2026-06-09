# DCASS  
**Dynamic Context-Aware Semantic Steganography via Zero-Modification Media Curation**

---

## 📌 Overview

DCASS is a **research-oriented system** for *semantic steganography* that enables covert communication **without modifying any carrier media**.  
Instead of embedding data into pixels, audio samples, or bitstreams, DCASS encodes messages by **curating semantically aligned, naturally occurring media** (text, images, audio) and distributing them using **human-like behavioral patterns**.

This project explores the intersection of:
- Semantic communication
- Multi-modal embeddings
- AI-driven stealth (GANs & Reinforcement Learning)
- Traffic analysis evasion

The system is designed as a **proof-of-concept prototype** accompanied by a **research paper**.

---

## 🎯 Key Idea

> *If a message can be represented by meaning rather than bits, then existing content can act as a carrier without ever being altered.*

DCASS:
- Encodes messages into **semantic vector sequences**
- Retrieves **existing media** from a large corpus
- Uses **dynamic context keys** to prevent static mappings
- Distributes content using **behaviorally realistic schedules**
- Achieves stealth against both **content-based steganalysis** and **traffic analysis**

---

## ✨ Core Features

- **Zero-Modification Steganography**  
  No changes to carrier media → resistant to classical steganalysis

- **Multi-Modal Support**  
  Text, image, and audio-based semantic encoding

- **Unified Vector Search**  
  FAISS-based high-performance similarity search

- **Dynamic Context Awareness**  
  Time, public data, and contextual keys affect encoding

- **AI-Based Stealth**
  - GAN-based human behavior scheduler
  - Reinforcement Learning agent for adaptive stealth

- **Adversarial Evaluation**
  Traffic analysis, stealth metrics, and benchmarking

- **CLI-Based Prototype**
  Fully controllable via command line

---

## 🧠 System Architecture

DCASS is organized into **four logical layers**:

1. **Corpus & Indexing**
   - Large-scale text, image, and audio datasets
   - Semantic embeddings (Sentence-Transformers, CLIP, CLAP)
   - Unified FAISS vector index

2. **Encoding / Decoding Engine**
   - Semantic chunking
   - Message ↔ vector sequence transformation
   - Dynamic context key derivation
   - Error correction mechanisms

3. **Stealth & Distribution**
   - GAN-based behavioral scheduler
   - RL-based policy agent
   - Multi-channel content distribution

4. **Analysis & Testing**
   - Performance benchmarks
   - Adversarial traffic analysis
   - Stealth and accuracy metrics

---

## 🧰 Technology Stack

| Component | Technology |
|---------|------------|
| Language | Python |
| ML Framework | PyTorch |
| Embeddings | Sentence-Transformers, CLIP, CLAP |
| Vector DB | FAISS |
| GAN | Custom PyTorch implementation |
| RL | Stable-Baselines3 / RLlib |
| CLI | Typer / Click |
| Data Processing | NumPy, Pandas, Librosa |

---

## 📖 Documentation

All documentation lives in the [`docs/`](./docs/) folder:

### User Guides
| Guide | Description |
|---|---|
| [Getting Started](./docs/guides/GETTING_STARTED.md) | Install, configure, and run for the first time |
| [Quick Start](./docs/guides/QUICK_START.md) | Fast track to running DCASS |
| [Script Execution Guide](./docs/guides/SCRIPT_EXECUTION_GUIDE.md) | Complete Docker & local execution reference |
| [Scripts Reference](./docs/guides/SCRIPTS.md) | How to run every script with examples |
| [Docker Setup](./docs/guides/DOCKER_SETUP.md) | Full Docker reference |

### Project Documentation
| Document | Description |
|---|---|
| [Complete Implementation Guide](./docs/guides/COMPLETE_IMPLEMENTATION_GUIDE.md) | Full feature documentation |
| [Implementation Summary](./docs/project/IMPLEMENTATION_SUMMARY.md) | GAN + RL technical details |
| [Project Status](./docs/project/PROJECT_COMPLETION_STATUS.md) | Current completion status |
| [Team Handoff Guide](./docs/guides/TEAM_HANDOFF_GUIDE.md) | Comprehensive handoff document |
| [Full Handout](./docs/project/DCASS_Implementation_Handout.md) | Complete module walkthrough |

### Migration & Refactoring
| Document | Description |
|---|---|
| [Refactoring Guide](./docs/guides/REFACTORING_MIGRATION_GUIDE.md) | Directory structure changes & migration |

---

## 📁 Project Structure

```text
dcass/
├── README.md                # This file
│
├── docs/                    # All documentation
│   ├── guides/              # User-facing guides
│   ├── project/             # Project documentation
│   ├── diagrams/            # Architecture diagrams
│   └── research/            # Research artifacts
│
├── src/                     # Core system code
│   ├── corpus/              # Dataset loading, preprocessing, embeddings, FAISS
│   ├── engine/              # Encoding / decoding logic
│   ├── stealth/             # GAN scheduler and RL agent
│   ├── distribution/        # Multi-channel dispatcher
│   ├── analysis/            # Benchmarks and adversarial testing
│   ├── api/                 # FastAPI backend server
│   └── cli/                 # Command-line interface
│
├── scripts/                 # Organized executable scripts
│   ├── data/                # Data preparation & corpus building
│   ├── audio/               # Audio-specific workflows
│   ├── training/            # Model training (GAN & RL)
│   ├── runtime/             # Core execution (sender/receiver/server)
│   ├── demos/               # Demo scripts
│   ├── testing/             # Testing & evaluation
│   └── utils/               # Utility scripts
│
├── storage/                 # Runtime data (gitignored)
│   ├── data/                # Downloaded datasets & indices
│   ├── models/              # Trained GAN and RL checkpoints
│   ├── logs/                # Application logs
│   ├── checkpoints/         # Training checkpoints
│   └── shared_channel/      # Alice-Bob communication
│
├── frontend/                # Next.js web interface
├── tests/                   # Unit, integration, adversarial tests
├── config/                  # Configuration files
└── tools/                   # Legacy/one-off scripts

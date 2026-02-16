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

## 📁 Project Structure

```text
dcass/
├── src/
│   ├── corpus/          # Dataset loading, preprocessing, embeddings, FAISS
│   ├── engine/          # Encoding / decoding logic
│   ├── stealth/         # GAN scheduler and RL agent
│   ├── distribution/   # Multi-channel dispatcher
│   ├── analysis/        # Benchmarks and adversarial testing
│   └── cli/             # Command-line interface
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── models/              # Trained GAN and RL models
├── tests/               # Unit, integration, adversarial tests
├── notebooks/           # Research experiments
├── docs/                # Architecture & research paper
└── scripts/             # Utility scripts

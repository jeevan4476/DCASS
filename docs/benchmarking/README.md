# DCASS Scientific Benchmarking & Verification Suite
## Master Architecture, Theoretical Foundations & Experimental Methodology

> **Target Conferences:** USENIX Security, ACM CCS, IEEE Transactions on Information Forensics and Security (TIFS), NeurIPS.  
> **System:** Distributed Covert Communication via Adversarial Semantic Steganography (DCASS).  
> **Core Steganographic Paradigm:** Selection-Based Coverless Dynamic Semantic State-Space Coding (DSSC) with Multi-Modal GAN/RL Temporal Obfuscation.

---

## 1. Executive Summary & Purpose

The DCASS Benchmarking Suite provides the complete empirical and mathematical verification required to prove the core scientific and security claims of the DCASS architecture. 

In traditional digital steganography (LSB matching, S-UNIWARD, WOW, HILL), messages are hidden by imperceptibly altering the pixel, sample, or character values of a cover medium. Because modifications are made to physical media, high-order statistical detectors (SRNet, Xu-Net, Ye-Net, SRM) can detect high-frequency residual energy and intercept communications.

DCASS operates under a fundamentally different information-theoretic paradigm: **Coverless Dynamic Semantic State-Space Coding (DSSC)**. DCASS **never modifies any media files**. Instead:
1. Media items (images, audio clips, text documents) are drawn untouched from public corpus repositories ($\mathcal{C}_{\text{public}}$).
2. The payload is encoded strictly into the **semantic selection index** of carriers across high-dimensional Voronoi topological clusters, modulated by a private, session-keyed pseudo-random permutation.
3. Media items are transmitted across disparate communication channels governed by an adversarial GAN / RL temporal scheduler.

This benchmarking suite provides rigorous, reproducible, un-mocked experimental proof across all layers of the DCASS stack.

---

## 2. Theoretical Foundations

### 2.1 Information-Theoretic Security ($D_{KL} = 0.0$)
In Christian Cachin's seminal 1998 information-theoretic model of steganography (*"An Information-Theoretic Model for Steganography"*, Information and Computation), a steganographic system is defined as $\epsilon$-secure against a passive adversary if the relative entropy (Kullback-Leibler divergence) between the cover distribution $P_{\text{cover}}$ and the steganographic carrier distribution $P_{\text{stego}}$ satisfies:

$$D_{KL}(P_{\text{cover}} \parallel P_{\text{stego}}) = \sum_{x \in \mathcal{X}} P_{\text{cover}}(x) \log_2 \left( \frac{P_{\text{cover}}(x)}{P_{\text{stego}}(x)} \right) \le \epsilon$$

* If $\epsilon = 0$, the system is **perfectly secure**. The adversary's probability of error $P_E$ collapses to random guessing:
  $$P_E = \frac{1}{2}(P_{FA} + P_{MD}) = \frac{1}{2} = 0.5000$$
* In DCASS, because the carrier images, audio, and text transmitted over the wire are drawn directly from the public corpus without modifying a single bit, pixel, or audio sample:
  $$P_{\text{stego}}(x) \equiv P_{\text{cover}}(x) \quad \forall x \in \mathcal{X}$$
  $$\therefore D_{KL}(P_{\text{cover}} \parallel P_{\text{stego}}) \equiv 0.0000 \text{ bits}$$

### 2.2 The Coverless Threat Model (The Real Adversary)
Because pixel-level modification is zero, traditional steganalysts fail by mathematical definition. However, an academic security reviewer evaluates a coverless system against a **Semantic / Statistical Warden**:
1. **Pixel/Acoustic Layer:** Does the transmitted media contain physical residual perturbations?
2. **Semantic Distribution Layer (Selection Bias):** Does the sequence of chosen carriers distort the background corpus distribution (e.g. over-sampling specific semantic clusters)?
3. **Temporal Traffic Layer:** Does the inter-arrival cadence of packets reveal automated, non-human covert transmission?
4. **Transport Reliability Layer:** Does the channel survive network erasures, packet dropouts, and adversarial probing?

---

## 3. Architecture of the Benchmarking Suite

The benchmarking suite is modularized into 6 comprehensive, independent components:

```
docs/benchmarking/
├── README.md                              # Master Overview: Theoretical Foundations & CLI Runner
├── 00_METHODOLOGY_AND_RATIONALE.md        # Scientific Rationale: Why Option B, CSM & Competent Adversaries
├── 01_MEDIA_LAYER_STEGANALYSIS.md         # Component 1: 5-Detector Deep Image & Acoustic Steganalysis
├── 02_SEMANTIC_SELECTION_BIAS.md          # Component 2: CLIP/CLAP Manifold Drift & KS/Wasserstein Tests
├── 03_TRAFFIC_TIMING_EVASION.md           # Component 3: GAN/RL Scheduling vs DPI Warden & Traffic Analysis
├── 04_CODEC_PERFORMANCE_AND_ECC.md        # Component 4: DSSC vs VCP Capacity, Throughput & Adaptive RS-ECC
├── 05_CONCURRENCY_AND_API_STRESS.md       # Component 5: Multi-threaded Concurrency, Locks & API Stress
└── 06_PUBLICATION_ARTIFACTS_AND_LATEX.md  # Component 6: Automated LaTeX Tables & Publication Figures
```

### Component Overview Matrix

| # | Component Document | Focus Area | Key Models / Tools Evaluated | Target Scientific Metric |
| :---: | :--- | :--- | :--- | :--- |
| **0** | [`00_METHODOLOGY_AND_RATIONALE.md`](00_METHODOLOGY_AND_RATIONALE.md) | Scientific Rationale | Option B vs Option A, CSM Analysis, Positive Controls | Peer-review validity, 0 confounding variables |
| **1** | [`01_MEDIA_LAYER_STEGANALYSIS.md`](../01_MEDIA_LAYER_STEGANALYSIS.md) | Pixel & Acoustic Media | SRM (30-filter), Xu-Net, Ye-Net, Zhu-Net, SRNet | ROC AUC $= 0.500$, $D_{KL} = 0.000$, $P_E = 0.500$ |
| **2** | [`02_SEMANTIC_SELECTION_BIAS.md`](../02_SEMANTIC_SELECTION_BIAS.md) | Distributional Drift | CLIP (ViT-B/32), CLAP (HTSAT), 256 Voronoi clusters | Wasserstein-1 ($W_1$), KS-test ($p > 0.05$), Entropy $H(C)$ |
| **3** | [`03_TRAFFIC_TIMING_EVASION.md`](../03_TRAFFIC_TIMING_EVASION.md) | Network Traffic Cadence | DPI Warden, WGAN Generator, PPO RL Agent | Interception Rate $P_{\text{detect}} < 5\%$, Delay Entropy $H_T$ |
| **4** | [`04_CODEC_PERFORMANCE_AND_ECC.md`](../04_CODEC_PERFORMANCE_AND_ECC.md) | Codec & Cryptography | DSSC Codec, RS-ECC, HMAC-DRBG Permutation | Latency $< 50\text{ms}$, Capacity $\sim 12\text{ bpc}$, 15% Loss Recovery |
| **5** | [`05_CONCURRENCY_AND_API_STRESS.md`](../05_CONCURRENCY_AND_API_STRESS.md) | System Concurrency | FastAPI Engine, Threading Locks, Isolated States | 50 Concurrent Sessions, 0% Race Conditions, $p_{99} < 100\text{ms}$ |
| **6** | [`06_PUBLICATION_ARTIFACTS_AND_LATEX.md`](../06_PUBLICATION_ARTIFACTS_AND_LATEX.md) | Paper Export | Automated LaTeX Tabular, Matplotlib High-DPI Curves | Formatted LaTeX tables for USENIX / ACM CCS papers |

---

## 4. Execution Workflow

All benchmarks are executable via a unified CLI script located in `scripts/analysis/`:

```bash
# 1. Run the complete end-to-end benchmark suite:
./.venv/bin/python scripts/analysis/run_master_benchmark.py --all

# 2. Run an individual benchmark component:
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 1   # Media Steganalysis
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 2   # Semantic Distribution
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 3   # Traffic Evasion
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 4   # Codec Performance
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 5   # API Concurrency
./.venv/bin/python scripts/analysis/run_master_benchmark.py --component 6   # Export LaTeX & Plots

# 3. Quick verification run (reduced sample size for fast sanity check):
./.venv/bin/python scripts/analysis/run_master_benchmark.py --quick
```

---

## 5. Academic Citation Standards & Reproducibility

Every component in this suite is implemented with:
1. **Deterministic Random Seeds:** All stochastic evaluations lock seeds (`seed=42`) using `torch.manual_seed(42)` and `np.random.default_rng(42)`.
2. **Zero Synthetic Mocking:** All tests operate on real media files from `storage/data/raw/flickr30k/flickr30k/images/` and real vector indices from `storage/data/indices/`.
3. **Standard Scientific Packages:** Calculations utilize `scikit-learn` (`sklearn.metrics.roc_auc_score`), `scipy.stats` (`entropy`, `ks_2samp`, `wasserstein_distance`), and `PyTorch` (v2.14).

# DCASS Modular Engineering & Theoretical Specification Index


## Complete Module Reference Suite

This directory contains deep technical, mathematical, and intuitive specifications for every subsystem built in DCASS.

```
===================================================================================================
                                  DCASS CORE MODULE DIRECTORY
===================================================================================================

 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 1: REED-SOLOMON ERROR CORRECTION CODE (GF(2^8))                                        │
 │ File: docs/modules/01_REED_SOLOMON_ECC_MODULE.md                                              │
 │ Code: src/engine/ecc.py (RSErrorCorrection)                                                   │
 │ Math: Galois Field GF(2^8), Generator G(x), Syndrome S_i, Berlekamp-Massey, Chien/Forney      │
 │ Purpose: Absorbs continuous vector quantization drift noise -> 0% Bit Error Rate (BER)       │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 2: SPHERICAL K-MEANS VORONOI CODEBOOK PARTITIONING (VCP)                               │
 │ File: docs/modules/02_VORONOI_CODEBOOK_PARTITIONING.md                                        │
 │ Code: src/corpus/cluster/voronoi_codebook.py (VoronoiCodebook)                                │
 │ Math: Unit Hypersphere S^511, Centroid Norm ||c_m||_2 = 1.0, Soft-Margin Buffer delta >= 0.05 │
 │ Purpose: Divides continuous 512d space into 256 non-overlapping discrete byte cells (0x00..FF)│
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 3: UNIFIED MULTI-MODAL FAISS 512D VECTOR INDEXING                                      │
 │ File: docs/modules/03_UNIFIED_MULTIMODAL_INDEXING.md                                          │
 │ Code: src/corpus/index/unified_index.py (UnifiedIndex, MediaItem)                             │
 │ Math: CLIP ViT-B/32 (Images/Text) + CLAP HTSAT (Audio), L2-Norm, FAISS IndexFlatIP            │
 │ Purpose: Unified 153,281-vector corpus with exact local disk file path resolution             │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 4: SEMANTIC CHUNKING & DUAL-CONSTRAINT CARRIER SELECTION                               │
 │ File: docs/modules/04_SEMANTIC_CHUNKING_AND_CARRIER_SELECTION.md                              │
 │ Code: src/engine/chunker.py, src/engine/encoder.py, src/engine/decoder.py                     │
 │ Math: argmax_{x in V(c_m)} cos(v_chunk, x) (Dual-Constraint: Symbol Exactness + Meaning)      │
 │ Purpose: Selects authentic media that mirrors secret topic while embedding cryptographic byte  │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 5: WGAN-GP TEMPORAL TRAFFIC MIMICRY GENERATOR                                          │
 │ File: docs/modules/05_WGAN_GP_TRAFFIC_MIMICRY.md                                              │
 │ Code: src/stealth/gan/generator.py, src/stealth/gan/trainer.py, src/analysis/adversarial/      │
 │ Math: Wasserstein-1 Distance, 1-Lipschitz Gradient Penalty (lambda=10.0), Causal Gated TCN   │
 │ Purpose: Generates human social-media posting burstiness, defeating Deep Packet Inspection    │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 6: INFORMATION-THEORETIC SECURITY & ZERO-MODIFICATION STEGANALYSIS DEFENSE             │
 │ File: docs/modules/06_SECURITY_AND_STEGANALYSIS_DEFENSE.md                                    │
 │ Code: Zero-Modification Carrier Transmission Architecture                                     │
 │ Math: Relative Entropy D_KL(P_cover || P_stego) = 0.000 bits, Cachin Epsilon=0, ROC AUC=0.500 │
 │ Purpose: Renders deep convolutional steganalysts (SRNet, Zhu-Net, Ye-Net) mathematically blind│
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MODULE 7: PPO REINFORCEMENT LEARNING CLOSED-LOOP STEALTH SCHEDULER                            │
 │ File: docs/modules/07_PPO_REINFORCEMENT_LEARNING_SCHEDULER.md                                 │
 │ Code: src/stealth/rl/agent.py (PPOAgent, ActorCritic), src/stealth/rl/environment.py          │
 │ Math: Multi-Objective R_t, Action Masking, GAE-lambda, Path Entropy H(p)=1.57/1.58 bits      │
 │ Purpose: Closed-loop multi-platform channel hopping with live cooldown and backpressure evad  │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
===================================================================================================
```

---

## Module Index Overview

| Module | Core File | Key Concept | Primary Formula |
| :---: | :--- | :--- | :--- |
| **01** | [`01_REED_SOLOMON_ECC_MODULE.md`](./01_REED_SOLOMON_ECC_MODULE.md) | RS-ECC $GF(2^8)$ Block Code | $C(x) = M(x) \cdot x^{2t} + P(x)$ |
| **02** | [`02_VORONOI_CODEBOOK_PARTITIONING.md`](./02_VORONOI_CODEBOOK_PARTITIONING.md) | Spherical K-Means $\mathbb{S}^{511}$ | $\mathbf{c}_m = \frac{\sum x_i}{\|\sum x_i\|_2}, \; \delta \ge 0.05$ |
| **03** | [`03_UNIFIED_MULTIMODAL_INDEXING.md`](./03_UNIFIED_MULTIMODAL_INDEXING.md) | Multi-Modal FAISS 512d Index | $\text{sim}(\mathbf{u}, \mathbf{v}) = \langle \mathbf{u}, \mathbf{v} \rangle$ |
| **04** | [`04_SEMANTIC_CHUNKING_AND_CARRIER_SELECTION.md`](./04_SEMANTIC_CHUNKING_AND_CARRIER_SELECTION.md) | Dual-Constraint Optimization | $x^* = \arg\max_{x \in \mathcal{V}(c_m)} \cos(v_{\text{chunk}}, x)$ |
| **05** | [`05_WGAN_GP_TRAFFIC_MIMICRY.md`](./05_WGAN_GP_TRAFFIC_MIMICRY.md) | WGAN-GP Traffic Generator | $\min_G \max_D \mathbb{E}[D(x)] - \mathbb{E}[D(\tilde{x})] - L_{\text{GP}}$ |
| **06** | [`06_SECURITY_AND_STEGANALYSIS_DEFENSE.md`](./06_SECURITY_AND_STEGANALYSIS_DEFENSE.md) | $D_{\text{KL}} = 0.0$ Steganalysis Proof | $D_{\text{KL}}(P_{\text{cover}} \parallel P_{\text{stego}}) = 0.000$ |
| **07** | [`07_PPO_REINFORCEMENT_LEARNING_SCHEDULER.md`](./07_PPO_REINFORCEMENT_LEARNING_SCHEDULER.md) | PPO Closed-Loop Controller | $R_t = \text{Throughput} - \lambda P_{\text{Warden}} + \beta H(\mathbf{p})$ |


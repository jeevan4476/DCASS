# DCASS Benchmarking Technical Report
## Distributed Covert Adaptive Semantic Steganography — Complete Empirical Verification Suite

**Version:** 1.0  
**Date:** October 2026  
**Classification:** Internal Research — Claims Reconciled to Recorded Artifacts
**System:** DCASS (Distributed Covert Adaptive Semantic Steganography)  
**Core Paradigm:** Coverless Dynamic Semantic State-Space Coding (DSSC) with Multi-Modal GAN/RL Temporal Obfuscation

---

## Executive Summary

This report records what the current artifacts support across the six components. It is not a claim that every design goal was met. The headline table separates measured facts from qualified or negative results.

### Recorded Results

| Metric | Recorded Result | Status | Scope |
|--------|-----------------|--------|-------|
| Coded carrier capacity | 13.50–13.66 bits/carrier, mean 13.60 | PASS (coded) | Includes CRC/RS framing and parity; not user-payload capacity |
| User-payload capacity | 7.00 bpc at 14 B; 12.90 bpc at 500 B | Measured (derived) | Payload bytes divided by recorded carrier count |
| Carrier reduction | 40.7–41.4% multi-modal; 48.1–49.9% text-only | PASS | Relative to the all-modality exact_vcp reference count |
| Media steganalysis | AUC 0.4973–0.4993; every 95% CI covers 0.5000 | Negative control | Image-only; SRNet, ZhuNet, YeNet; not a semantic or traffic security proof |
| Semantic drift | 0/1,536 KS BH discoveries; small image/text energy shifts; audio p≈0.57 | Mixed | Uniform null rejected; mechanism null not rejected (p=0.208) |
| Traffic evasion | 100% interception for the WGAN and Hybrid schedules, 54% for Poisson, 0% for the retrained PPO arm, under a held-out calibrated detector | **NOT DEMONSTRATED** | The PPO arm evades a detector it never trained against and respects the platform cooldowns, but the human baseline is a synthetic corpus and the detector is narrow |
| Round-trip accuracy | 100% on recorded clean-channel trials | PASS | Dropout is unrecoverable in the current wire format |
| Direct-engine concurrency | 50 threads, 100 sessions, 100% decode, 0% observed swaps | PASS (direct engine) | API transmit admits concurrent sessions up to a bounded limit and returns HTTP 429 beyond it |

### Critical Bug Fix: HMAC Framing Overhead

A design issue was found and corrected: the original DSSC implementation appended a 32-byte HMAC-SHA256 tag during payload framing, consuming 66.7% of transmission bandwidth for small messages. The HMAC-routed family selection and session-permuted state spaces are intended to provide cryptographic separation; the framing change was validated by the recorded round-trip and wrong-key tests, not assumed.

**Fix Applied:** Switched to lightweight CRC-16 framing (5-byte header), reducing framed overhead from 40 to 10 bytes (75% reduction). The recorded coded mean is 13.60 bits/carrier (70.0% above the 8-bit coded baseline). User-payload capacity is lower and is reported separately in Section 6.1.

### Integrity Fix: Framed Payloads Under ECC

The rebuilt Component 4c campaign found a second correctness issue. In dropout trials, a deletion could move the frame marker away from byte zero while Reed-Solomon still returned a codeword. The legacy raw fallback then accepted the result as an unframed message and reported success with wrong plaintext. The pre-hardening run recorded 41 such accepts across the 320 dropout trials at 2 through 5 percent deletion, 8.8 to 21.2 percent acceptance at those rates.

The decoder now requires framed payloads whenever ECC is enabled, and legacy raw payloads remain decodable only when ECC is off. The current 4c artifact records zero false acceptances, and exact recovery is unchanged. The pre-hardening artifact and table are archived in `storage/benchmarks/archive/`.

---

## 1. System Architecture Overview

### 1.1 Core Design Principles

DCASS implements **zero-modification semantic steganography**: messages are encoded by selecting existing carrier media from a public corpus based on semantic alignment, without modifying any carrier bytes. This defeats classical steganalysis (LSB, SRM, CNN-based) which detects pixel/statistical modifications.

### 1.2 Multi-Modal Corpus

| Modality | Source | Items | Embedding | Dimensions |
|----------|--------|-------|-----------|------------|
| Image | Flickr30k | 31,783 | CLIP ViT-B/32 | 512 |
| Text | Wikipedia | 200,000 | CLIP ViT-B/32 | 512 |
| Audio | `libretta` (via CLAP) | 13,496 | CLAP HTSAT-base | 512 |
| **Total** | — | **245,279** | — | 512 |

### 1.3 Voronoi Codebook Partitioning (VCP)

- 256 centroids on the unit hypersphere S⁵¹¹
- Modality-stratified spherical k-means with δ = 0.05 margin
- Certified binding between codebook and FAISS indices (SHA-256 fingerprints)
- 256 clusters → direct byte mapping (0x00–0xFF)

### 1.4 Semantic Families (2-2-2 Split)

| Modality | Families | Cluster Range | Items/Family | Bits/Carrier |
|----------|----------|---------------|--------------|--------------|
| Image | nature_outdoor, urban_architecture | 0–84 | 15.5K | 13.0–13.8 |
| Text | people_interaction, objects_indoor | 85–169 | 89K–107K | 16.0 |
| Audio | technology_science, sound_atmosphere | 170–255 | 6.5K–7K | 12.0–12.6 |

### 1.5 Encoding Pipeline (DSSC)

```
Message → CRC-16 Framing → RS-ECC → Semantic Chunking
    → Family Selection (HMAC) → Modality Round-Robin
    → State-Space Permutation (NumPy DRBG)
    → Symbol → Carrier Selection → Output
```

### 1.6 Stealth Scheduler

| Mode | Architecture | Recorded Warden Interception (Component 3) |
|------|--------------|-------------------------------------------|
| Static | NoiseController (mathematical) | 100% |
| GAN | TemporalPatternGenerator (WGAN-GP) | 100% interception under the calibrated composite warden; not an evasion result |
| RL | PPO Agent (Multi-channel) | 100% (50/50) |

The scheduler shapes transmission timing only. It does not change the codec, so the recorded coded capacity is 13.60 bits per carrier in every mode. User-payload capacity is lower; see Section 6.1.

---

## 2. Benchmarking Methodology

### 2.1 Six-Component Framework

| Component | Focus | Script |
|-----------|-------|--------|
| 1 | Media-Layer Steganalysis | benchmark_1_media_steganalysis.py |
| 2 | Semantic Distribution & Selection-Bias | benchmark_2_semantic_distribution.py |
| 3 | Traffic Timing & Traffic Analysis Evasion | benchmark_3_traffic_evasion.py |
| 4 | Codec Capacity, Throughput & Robustness | benchmark_4a_capacity.py, benchmark_4b_robustness.py, benchmark_4c_ecc_failure_analysis.py |
| 5 | Codec Concurrency & System Stress | benchmark_5_system_concurrency.py |
| 6 | Publication Artifacts | Written directly by each benchmark script |

### 2.2 Positive Control Protocol

Steganalysis detectors are validated against spatial ±1 grey-level twins at 1.0 bpp on Flickr30k, with a payload sweep at 0.4, 0.2, and 0.1 bpp:
- Threshold: 0.90 accuracy required on held-out twins, applied per fold, all folds must pass
- DCASS security metric: measured AUC ≈ 0.50 on DCASS carriers with balanced accuracy 0.50 within its interval

### 2.3 Statistical Rigor

- Sample sizes: 1,000 carriers requested per detector fold, of which 922, 933 and 944 are scored after the duplicate and train/val overlap exclusions, image modality only; the Component 2 semantic campaign uses 5,000 balanced embeddings per group, Component 2b counts 10,800 carriers, and Component 3 scores 50 warden trials per scheduler against a 5,000-draw surrogate reference
- Confidence: 95% intervals via bootstrap, at the per-metric resample count the component records (2,000 for the per-fold detector AUC, 1,000 for the fold-averaged AUC and balanced accuracy); the other components use Wilson intervals or permutation p-values
- Multiple testing: Benjamini-Hochberg FDR correction (α = 0.05)
- Reproducibility: seeds are fixed where recorded (SEED = 42) and HMAC-DRBG permutations are deterministic. C5 is a fresh run whose artifact records the commit, the timestamp, the device, and the package versions.

### 2.4 Output Artifacts

| Type | Location | Format |
|------|----------|--------|
| Raw Results | storage/benchmarks/*.json | Machine-readable |
| LaTeX Tables | storage/benchmarks/tables/table*.tex | Publication-ready |
| Plots | storage/benchmarks/plots/fig_*.png | 200 DPI, publication quality |
| Logs | storage/benchmarks/logs/ | Debug/trace |

---

## 3. Component 1: Media-Layer Steganalysis (Detailed)

### 3.1 Objective

Evaluate whether DCASS carriers are statistically indistinguishable from natural media under state-of-the-art steganalysis detectors.

### 3.2 Detectors Implemented

| Detector | Architecture | Reference | Control accuracy | Gate |
|----------|--------------|-----------|------------------|------|
| SRNet | End-to-end, 12 layers (Type1×2→Type2×5→Type3×4→Type4) | Boroumand et al., TIFS 2018 | 93.67% [92.88, 94.39] | pass |
| ZhuNet | Separable conv + SPP | Zhu et al., TIFS 2020 | 95.05% [94.33, 95.68] | pass |
| YeNet | 5 conv blocks, TLU, 30 SRM kernels | Ye et al., TIFS 2017 | 93.97% [93.19, 94.67] | pass |
| SRM + FLDA | 34,671-d co-occurrence features, random-subspace FLDA | Fridrich & Kodovsky, TIFS 2012 | 50.15% pooled | fail, disclosure foil |
| XuNet | CNN, ABS + tanh, KV high-pass front end | Xu et al., SPL 2016 | 82.03% pooled | fail, disclosure foil |

The official result rests on the three gated detectors, and the recorded `component_1_summary.json` and the three fold artifacts carry exactly those three. The two foils were executed in a separate five-detector campaign whose summary is archived at `storage/run/comp1_variants/five/component_1_summary.json`; they are excluded from the official summary under the gate rule, and disclosed with their caveats in `BENCHMARK_DEEP_EXPLANATION.md`, Section 1.7.

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | RTX 4050 Laptop (6GB VRAM) |
| Batch Size | 8, gradient accumulation ×4 (effective 32) |
| Precision | FP32 |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 20 per fold, best-validation-accuracy checkpoint |
| Protocol | 3 image-level folds over a 2,000-image pool; twins never straddle splits; 1.0 bpp control plus 0.4/0.2/0.1 sweep |
| Carriers | 1,000 requested per fold; fold-split overlap excluded, 922–944 scored per fold |

### 3.4 Detailed Results

#### Raw Data from component_1_summary.json

```json
{
  "schema": "component1-summary-v2",
  "protocol": {
    "images": 2000,
    "folds": 3,
    "epochs": 20,
    "carriers": 1000,
    "sweep_rates": [1.0, 0.4, 0.2, 0.1],
    "positive_control_gate": 0.9,
    "seed": 42
  },
  "detectors": [
    {
      "name": "sr_net",
      "reference": "Boroumand et al., TIFS 2018",
      "control": {"accuracy": 0.93675, "accuracy_ci": [0.9288, 0.9439],
                  "auc_mean": 0.9844, "n": 4000, "gate_pass_all": true},
      "dcass": {"auc_mean": 0.4973, "auc_ci": [0.4830, 0.5143],
                "balanced_mean": 0.5025, "balanced_ci": [0.4938, 0.5112],
                "cover_rate_mean": 0.8831, "carrier_n_mean": 933.0,
                "kl_mean": 0.2310, "kl_p_min": 0.238, "kl_p_max": 0.423},
      "sweep": {"1": {"accuracy": 0.93675}, "0.4": {"accuracy": 0.72125},
                "0.2": {"accuracy": 0.566}, "0.1": {"accuracy": 0.5235}}
    }
  ],
  "all_gates_passed": true
}
```

The excerpt is truncated: `zhu_net` and `ye_net` follow the same schema, and the full per-fold rows, split manifests, carrier provenance, and per-sample posteriors live in `storage/benchmarks/component_1_summary.json` and `component_1_fold0.json` through `component_1_fold2.json`.

### 3.5 Key Finding: DCASS Security Metric

The three gated detectors are statistically indistinguishable from natural media at the score level. Measured DCASS AUCs are 0.4973 [0.4830, 0.5143], 0.4993 [0.4820, 0.5146], and 0.4993 [0.4826, 0.5160], every 95 percent interval covering 0.5000, with balanced accuracy 0.5025, 0.5025, and 0.4988. The plug-in $D_{KL}$ values of 0.18 to 0.23 bits are calibrated by label permutation with $p$ ranges 0.238–0.423, 0.250–0.937, and 0.385–0.849, so they reflect estimator sparsity, not covert signal.

**Note on the accuracy versus AUC paradox:** the carrier cover rate is 88.31–92.34 percent, a one-sided rate at the 0.5 threshold. The balanced accuracy and the threshold-free AUC both sit at one half within their intervals, so the one-sided rate is threshold bookkeeping, not discriminative power.

### 3.6 LaTeX Table Output

```latex
% Generated by scripts/analysis/benchmark_1_media_steganalysis.py
\begin{table*}[t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Detector & Reference & PC acc (\%), 95\% CI & PC AUC & DCASS AUC, 95\% CI & Cover rate (\%) & $D_{\mathrm{KL}}$ [perm $p$] \\
\midrule
sr\_net & Boroumand et al., TIFS 2018 & 93.67 [92.88, 94.39] & 0.9844 & 0.4973 [0.4830, 0.5143] & 88.31 & 0.23 [0.238, 0.423] \\
zhu\_net & Zhu et al., TIFS 2020 & 95.05 [94.33, 95.68] & 0.9860 & 0.4993 [0.4820, 0.5146] & 92.14 & 0.18 [0.250, 0.937] \\
ye\_net & Ye et al., TIFS 2017 & 93.97 [93.19, 94.67] & 0.9853 & 0.4993 [0.4826, 0.5160] & 92.34 & 0.23 [0.385, 0.849] \\
\bottomrule
\end{tabular}}
\caption{Media-layer steganalysis resistance on an image-level 3-fold protocol. PC acc: pooled positive-control accuracy on held-out 1.0 bpp LSB twins with a Wilson 95\% CI. PC AUC and DCASS AUC: fold-mean AUC; the DCASS AUC is measured from recorded posteriors with a 95\% stratified bootstrap CI over folds. Cover rate: fraction of DCASS carriers decided cover at threshold 0.5. $D_{\mathrm{KL}}$: legacy plug-in statistic with its label-permutation $p$ range across folds. (n = 3 folds x 2000 images, 3 gated detectors; scope = image-only content-residual negative control; commit = 536808a8f8c1046e3802cf9ef094ef54d93a9908)}
\label{tab:steganalysis}
\end{table*}
```

---

## 4. Component 2: Semantic Distribution & Selection-Bias (Detailed)

### 4.1 Objective

Quantify whether DCASS carrier selection introduces statistical bias detectable by a "Semantic Warden" analyzing the sequence of carriers.

### 4.2 Metrics

| Metric | Definition | Security Interpretation |
|--------|------------|------------------------|
| Energy distance | Weighted cross-sample minus within-sample mean distance, permutation calibrated | At zero = exchangeable with the reference |
| MMD² (RBF kernel) | Kernel mean embedding distance, median-heuristic bandwidth, permutation calibrated | At zero = no kernel-mean shift |
| Wasserstein-1 (W₁) on PC1 | One-dimensional projection transport cost, secondary | Lower = projection shift smaller |
| Kolmogorov-Smirnov (KS) family | Per-dimension CDF difference, BH-controlled with a family-wise permutation p | Any BH discovery or family rejection = detectable coordinate shift |
| Voronoi Cluster Entropy | Shannon entropy over 256 clusters, on slots and on unique media | Higher = no semantic hotspots |
| Modality Balance | Carrier proportions across text/image/audio | Round-robin compliance |

### 4.3 Detailed Results

The recorded campaign runs 10 draws of 50 messages each, 500 messages and 27,850 carrier slots, with fresh session keys per message. Each reference sample holds 5,000 embeddings per modality drawn from corpus items that no draw ever transmitted, so reference and carrier sets are disjoint.

#### 4.3.1 Headline Multivariate Tests

Energy distance and RBF-kernel MMD² on the pooled 5,000 vs 5,000 balanced samples, calibrated by 1,000 label permutations on a shared distance matrix. The unique-media column repeats the energy test after collapsing repeats:

| Modality | Energy Distance | p | MMD² | p | Energy (Unique Media) | p |
|----------|-----------------|---|------|---|----------------------|---|
| Image | 0.00011 | 0.009 | 6.10e-05 | 0.012 | 0.00007 | 0.035 |
| Text | 0.00010 | 0.004 | 8.36e-05 | 0.004 | 0.00016 | 0.001 |
| Audio | -0.00003 | 0.574 | -3.28e-05 | 0.642 | -0.00004 | 0.683 |

**Interpretation:** Audio is indistinguishable from a uniform corpus draw; small negative values are ordinary sampling noise for an unbiased statistic. Image and text carry small but resolved shifts, about one ten-thousandth of the embedding scale, and they persist on unique media, so the signal is distributional rather than a repeat artifact. The shifts match the family-restricted selection pools by design; Section 6 of the deep explanation shows the transmitted stream is statistically consistent with the mechanism's own structural prediction, so the detectable component is the public mechanism rather than key-dependent randomness.

#### 4.3.2 Wasserstein-1 on PC1 (Secondary)

W₁ is measured on the first principal coordinate of a per-modality PCA fit on the pooled sample and reported with the share of pooled variance that coordinate carries. It is a one-dimensional projection, not a multivariate transport cost, so values are not comparable across modalities:

| Modality | W₁ (PC1) | PC1 Variance Share | Projection KS | p |
|----------|----------|--------------------|---------------|---|
| Image | 0.00810 | 5.74% | 0.0284 | 0.011 |
| Text | 0.00612 | 6.20% | 0.0253 | 0.030 |
| Audio | 0.00278 | 27.03% | 0.0106 | 0.857 |

#### 4.3.3 KS Family (Per-Dimension Results from component_2_results.json)

Each modality screens all 512 embedding dimensions, 1,536 tests in total, with Benjamini-Hochberg control at q = 0.05 and a family-wise permutation test of the maximum ECDF gap over 500 relabelings:

| Modality | Dimensions | Mean D_KS | Median D_KS | Max D_KS | BH Discoveries | Family-wise p | FW Null 95th |
|----------|------------|-----------|-------------|----------|----------------|---------------|--------------|
| Image | 512 | 0.0170 | 0.0159 | 0.0351 | 0 | 0.315 | 0.0391 |
| Text | 512 | 0.0163 | 0.0155 | 0.0363 | 0 | 0.148 | 0.0388 |
| Audio | 512 | 0.0159 | 0.0155 | 0.0362 | 0 | 0.092 | 0.0378 |

**Interpretation:** Zero coordinates reject after multiplicity control anywhere. The per-coordinate null center at these sample sizes is 0.0155 and the observed means land directly on it; the single largest gap, 0.0363, is a maximum over 512 correlated coordinates whose permutation null has a 95th percentile near 0.039, so it is not evidence of a coordinate shift.

#### 4.3.4 Voronoi Cluster Entropy and Repeats

| Metric | Value | Max | Ratio |
|--------|-------|-----|-------|
| Cluster Entropy, slots (H) | 7.8090 bits | 8.000 | 97.6% |
| Cluster Entropy, unique media | 7.7814 bits | 8.000 | 97.3% |
| Repeat rate | 27,850 slots on 23,890 unique media | reuse factor 1.1658 | max 6 uses |

The mixture identity closes exactly: weighted per-modality slot entropies of 6.2242 bits plus the modality mixing term of 1.5847 bits equal 7.8090 bits, a residual below two millionths of a bit. The 0.191 bit deficit from maximum is the structural price of modality routing, not a hotspot artifact.

#### 4.3.5 Modality Balance (Round-Robin Verification)

| Modality | Carriers | Proportion | Corpus | Status |
|----------|----------|------------|--------|--------|
| Image | 9,250 | 33.21% | 12.96% | PASS |
| Text | 9,500 | 34.11% | 81.54% | PASS |
| Audio | 9,100 | 32.68% | 5.50% | PASS |

Round-robin scheduler operating correctly; deviations from a perfect third stay under 0.8 points and trace to chunk-level packaging rather than sampling noise.

### 4.4 Visualizations Generated

| Plot | File | Description |
|------|------|-------------|
| Manifold t-SNE | fig_manifold_tsne.png | 3-panel disjoint reference vs carriers, 1,500 points per group, 1-σ ellipses |
| Distributional Stats | fig2_distributional_stats.png | Energy, MMD², and W₁ panels with per-draw dots and permutation p-values |
| KS Heatmap | fig_ks_heatmap.png | D_KS across 512 dimensions per modality with BH and family-wise annotations |

### 4.5 LaTeX Table Output

The runner generates `storage/benchmarks/tables/table2_distributional_drift.tex`:

```latex
\begin{table}[t]
\centering
\caption{Component 2 semantic distribution check (schema v2). Energy distance and MMD$^2$ are permutation calibrated against a disjoint reference sample; $W_1$ is a one-dimensional PC1 projection, not a multivariate transport cost. KS rejects count Benjamini--Hochberg discoveries over 512 dimensions; FW $p$ is the family-wise permutation p of the maximum ECDF-gap statistic. $H_{\mathrm{uniq}}$ is cluster entropy on unique media.}
\label{tab:component2-distributional-drift}
\small
\begin{tabular}{lccccccc}
\toprule
Modality & Energy $d$ [$p$] & MMD$^2$ [$p$] & $W_1$ PC1 & KS BH & FW $p$ & $H_{\mathrm{uniq}}$ & Share \\
\midrule
Image & 0.00011 [0.009] & 6.10e-05 [0.012] & 0.00810 & 0/512 & 0.315 & 6.320 & 33.21\% \\
Text & 0.00010 [0.004] & 8.36e-05 [0.004] & 0.00612 & 0/512 & 0.148 & 6.037 & 34.11\% \\
Audio & -0.00003 [0.574] & -3.28e-05 [0.642] & 0.00278 & 0/512 & 0.092 & 6.320 & 32.68\% \\
\bottomrule
\end{tabular}
\end{table}
```

### 4.6 Component 2b: Transmitted Cluster Usage

A dedicated 200-message campaign tests the 256-bin cluster histogram against two nulls: uniform usage and a structured replay of the pipeline's own mechanism. Against uniformity the histogram rejects decisively, chi-square 3,253.4 with permutation p 0.0005, because the corpus itself is not uniform; against the mechanism null it is unremarkable, chi-square 268.6 on 253 degrees of freedom, permutation p 0.208, largest standardized deviation +2.62 against a permutation ceiling of 4.08. The 10,800 carriers draw on 10,152 unique media, and the entropy, 7.7935 bits on slots and 7.7873 on unique media, sits inside the mechanism null interval [7.7837, 7.8098]. Full detail is in Section 6 of the deep explanation.

---

## 5. Component 3: Traffic Timing & Traffic Analysis Interaction (Detailed)

### 5.1 Threat Model

An inline DPI Warden monitors inter-arrival times (IAT) and channel routing to detect covert traffic:

- Periodic Rhythm Anomaly: Fixed intervals → spectral spikes
- Burst Rate Anomaly: Rapid flushes → volume thresholds
- Channel Concentration: Single-channel hammering → cooling violations

### 5.2 Schedulers Evaluated

| Scheduler | Architecture | Delay Generation | Channel Routing |
|-----------|--------------|------------------|-----------------|
| Static Cadence | Fixed clock | Δt = 3.0s ∀ i | Round-robin |
| Stochastic Poisson | Exponential | Δt ~ Exp(λ=1/3) | Uniform random |
| DCASS Temporal WGAN | 100.0 | 0.0 | 3.6586 | 0.5595 | 1.06e-06 | 7.50 |
| DCASS Hybrid Engine | 100.0 | 0.0 | 3.6584 | 0.5553 | 7.75e-05 | 7.44 |
| DCASS PPO RL Agent | PPO Actor-Critic | π(a|s) optimizing latency vs DPI | Action masking |

### 5.3 Warden Architecture

DeepPacketInspectionWarden (BiLSTM + Transformer Encoder):
- IAT Statistics: μ, σ², CV, Skewness, Kurtosis
- Delay Shannon Entropy: H_T = -Σ p_b log₂(p_b)
- Sliding Window Burst Density (W=5.0s)
- Output: unbounded anomaly head plus sigmoid bot probability

Interception decisions in the benchmark come from the composite `evaluate_dpi_warden` function, not from the neural module alone:
- Periodic rules: CV < 0.20 or log-bin entropy < 1.0 bits, fixed score 0.98
- Burst rule: 6 or more packets in a sliding 5.0s window, score 0.65 to 0.95
- Single-channel hammer rule: fixed score 1.0
- Soft branch for all other traffic: the bot probability of a trained traffic detector, judged against a threshold calibrated on held-out human traffic
- Rate-limit violations are counted as a diagnostic field and are not scored
- The KS distance to the human reference is recorded per trial and does not enter the score
- The shipped WGAN-GP warden checkpoint in `storage/models/warden.pt` is a constant function (mean anomaly 0.62493 for every input) and is not used by the benchmark; a purpose-built detector supplies the neural term, and the one used for the measurement is the held-out `storage/models/traffic_detector_eval.pt`

### 5.4 Detailed Results (Recorded Composite Warden)

| Scheduler | Interception Rate (%) | Not Flagged Rate (%) | Delay Entropy (bits) | KS Distance | KS p-value | Mean Latency (s) |
|-----------|----------------------|------------------|---------------------|-------------|------------|------------------|
| Static Cadence | 100.0 | 0.0 | 0.0000 | 0.5330 | 1.20e-13 | 3.00 |
| Stochastic Poisson | 54.0 | 46.0 | 4.2335 | 0.2211 | 4.88e-02 | 3.04 |
| DCASS Temporal WGAN | 100.0 | 0.0 | 3.6586 | 0.5595 | 1.06e-06 | 7.50 |
| DCASS Hybrid Engine | 100.0 | 0.0 | 3.6584 | 0.5553 | 7.75e-05 | 7.44 |
| DCASS PPO RL Agent | 0.0 | 100.0 | 3.5809 | 0.2538 | 1.24e-02 | 5.20 |

The soft branch's flag threshold is 0.0181, calibrated on 457 human bursts at a 5% target false-positive rate; the measured false-positive rate on that set is 4.60%.

#### Diurnal Conditioning Check (WGAN, identical mapping)

| Time | Raw Output Mean | Mean Latency (s) | Std Dev (s) | Delay Entropy (bits) | KS Distance | Interception Rate (%) |
|------|-----------------|------------------|-------------|----------------------|-------------|----------------------|
| Daytime Hour (14:00) | 0.501596 | 3.89 | 3.21 | 2.7290 | 0.4303 | 100.0 |
| Night Hour (03:00) | 0.508955 | 14.93 | 4.15 | 4.4098 | 0.8357 | 100.0 |

An earlier revision of this report listed 3.19 s daytime against 13.07 s night dormancy. Those numbers came from per-regime affine constants inside the benchmark script rather than the model and are withdrawn; see `BENCHMARK_DEEP_EXPLANATION.md` section 3.4. The human surrogate baseline records 1.1735 bits under the linear-bin descriptive estimator and 4.2309 bits under the log-bin estimator used by the warden's periodic rule.

### 5.5 Key Observations

- **Static cadence fails completely** — 100% interception through the coefficient-of-variation rule, zero entropy
- **Poisson records 54% interception (46% not flagged)** — 27 of 50 trials fire the burst rule at scores 0.71 to 0.89, and the other 23 reach the soft branch at a bot probability of about 0.004, well below the 0.0181 threshold, because the detector reads exponential inter-arrivals as human-like
- **DCASS GAN and Hybrid record 100% interception** — their soft-branch bot probabilities span 0.3300 to 0.9945 and 0.3976 to 0.9945 against a threshold of 0.0181, so every trial of both is flagged. The earlier 0/50 for these rows was the soft branch's arithmetic and is withdrawn
- **The retrained RL agent records 0% interception** — all 50 trials score 0.0040 to 0.0047, below the threshold and below the human population's 95th percentile of 0.0133. It is also the only row that respects the platform's per-channel cooldowns: 15 violations across 50 trials against 2,450 for the static baseline. The shipped agent it replaces was trained with an identically zero stealth penalty, because the warden it trained against returns `bot_probability = 0.0` for every input, and it was flagged at 100%. See [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md)
- **No measurable diurnal adaptation** — hour-14 and hour-3 means are 3.89 s and 14.93 s under one identical mapping; the sign agrees with a dormancy policy but the training data has no hour-conditioned structure to learn it from. The earlier 3.19 s / 13.07 s split was an artifact of per-regime analysis constants
- **Framework validated** — the surrogate reference is flagged at 12% (burst rule) and the detector's own held-out corpus sessions at 2%; the soft branch itself flags none of the 44 surrogate draws that reach it. The threshold is calibrated on 457 human bursts at a measured 4.60% false-positive rate
- **The neural term now separates traffic** — the trained detector's bot probability is 0.004 to 0.006 on human traffic from an independent generator and 0.509 and 0.481 on the DCASS generators (`storage/benchmarks/traffic_detector_generalization.json`). The superseded WGAN-GP checkpoint was a constant 0.62493 on every input. The root cause of that collapse is documented in [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md)
- **The PPO arm is a stealth policy now, and was not before** — it is retrained against a working detector, and the retraining required fixing four defects in the RL path: the agent optimized delays the network never saw, the deployment transform collapsed 98.3% of its choices onto one value, the environment's cooldown test was more permissive than the benchmark's, and a rate-limit violation was priced below a soft failure. See [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md) §2
- **The scope of the result is bounded by the reference traffic** — the human side of the calibration is the repository's own traffic corpus, whose time-of-day distribution is uniform and whose delay values are bounded, so it is a synthetic surrogate rather than recorded human browsing. The measurement is separability from that corpus, not detectability to a real network observer

### 5.6 LaTeX Table Output

The generated `table3_traffic_evasion.tex` now labels its second column **Not Flagged Rate**, and the caption states that this is not an evasion rate.

```latex
% Component 3: Traffic Timing & DPI Evasion Benchmark Results
\begin{table*}[htbp]
\centering
\caption{Traffic Timing: Warden Interception Rates and Timing Fidelity. The second column is the not-flagged rate, not an evasion rate. (n = 5 schedulers x 50 trials x 50 carriers; scope = recorded composite warden; the soft branch is a trained traffic detector with a threshold calibrated on held-out human traffic, so a not-flagged rate is a measurement and not an artifact; commit = 9b02563f956c5bbfb39201fea39965be1348b5b4)}
\label{tab:traffic_evasion}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Scheduler & Interception Rate (\%) & Not Flagged Rate (\%) & Delay Entropy (bits) & KS Distance & KS $p$-value & Mean Latency (s) & Cooldown Violations & Triggering Rule (trials) \\
\midrule
Static Cadence & 100.0 & 0.0 & 0.0000 & 0.5330 & 1.20e-13 & 3.00 & 2450 & periodic cv 50/50 \\
Stochastic Poisson & 54.0 & 46.0 & 4.2335 & 0.2211 & 4.88e-02 & 3.04 & 1276 & burst 27/50, soft 23/50 \\
DCASS Temporal WGAN & 100.0 & 0.0 & 3.6586 & 0.5595 & 1.06e-06 & 7.50 & 982 & soft 50/50 \\
DCASS Hybrid Engine & 100.0 & 0.0 & 3.6584 & 0.5553 & 7.75e-05 & 7.44 & 13 & soft 50/50 \\
DCASS PPO RL Agent & 0.0 & 100.0 & 3.5809 & 0.2538 & 1.24e-02 & 5.20 & 15 & soft 50/50 \\
\midrule
\multicolumn{9}{l}{\textbf{Diurnal Conditioning Check (WGAN, identical mapping):}} \\
~~\textendash~Daytime Hour (14:00) & 100.0 & 0.0 & 2.7290 & 0.4303 & -- & 3.89 & -- & soft 25/25 \\
~~\textendash~Night Hour (03:00) & 100.0 & 0.0 & 4.4098 & 0.8357 & -- & 14.93 & -- & soft 24/25, periodic cv 1/25 \\
\midrule
Human Baseline & -- & -- & 1.1735 & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}}
\end{table*}
```

### 5.7 Component 3b: Warden Score Distributions

Every interception verdict is a threshold on a stored score distribution, and the distribution shapes carry information the rates alone hide. Static Cadence is a point mass at 0.98. Stochastic Poisson is bimodal: 23 soft trials at about 0.004 and 27 burst trials at 0.71 to 0.89, so its mean of 0.408 sits between two modes. The WGAN and Hybrid occupy a point mass at 0.994, and the retrained PPO arm a point mass at 0.004, below the threshold of 0.0181. Component 3b also reports an earlier PPO run at 44.07 s mean latency, but that run is a historical artifact: it was scored by the superseded WGAN-GP critic and by a scheduler that did not yet clip the RL delays, so its scores are on a different scale and its latency is not reproducible with the current code. It is judged against the 0.50 threshold that was in force when it was recorded and is not a second operating point of the current pipeline. The quantiles are archived in `storage/benchmarks/tables/table3b_warden_score_quantiles.tex` and drawn in `storage/benchmarks/plots/fig_3b_warden_scores.png`. Full detail is in Section 3 of the deep explanation.

---

## 6. Component 4: Codec Capacity, Throughput & Robustness (Detailed)

### 6.1 Capacity Benchmark (Component 4a): DSSC vs exact_vcp

#### 6.1.1 DSSC vs exact_vcp Comparison (Matched Framing + ECC)

| Mode | Payload (B) | Carriers | Coded Bits/Carrier | User Bits/Carrier | Reduction (%) | Encode (ms, ref.) | Decode (ms, ref.) |
|------|-------------|----------|--------------------|-------------------|---------------|-------------|-------------|
| Mode | Payload (B) | Carriers | Coded Bits/Carrier | User Bits/Carrier | Reduction (%) | Encode (ms, ref.) | Decode (ms, ref.) |
|------|-------------|----------|--------------------|-------------------|---------------|-------------|-------------|
| DSSC (multi-modal) | 14 | 16 | 13.50 | 7.00 | 40.7% | 106.0 | 181.3 |
| DSSC (text-only) | 14 | 14 | 15.43 | 8.00 | 48.1% | 89.3 | 243.3 |
| exact_vcp (multi-modal) | 14 | 27 | 8.00 | 4.15 | -- | 1784.6 | 2.2 |
| DSSC (multi-modal) | 50 | 37 | 13.62 | 10.81 | 41.3% | 113.6 | 280.4 |
| DSSC (text-only) | 50 | 32 | 15.75 | 12.50 | 49.2% | 111.2 | 442.1 |
| exact_vcp (multi-modal) | 50 | 63 | 8.00 | 6.35 | -- | 5232.6 | 5.0 |
| DSSC (multi-modal) | 150 | 96 | 13.58 | 12.50 | 41.1% | 119.0 | 542.7 |
| DSSC (text-only) | 150 | 82 | 15.90 | 14.63 | 49.7% | 153.6 | 1009.4 |
| exact_vcp (multi-modal) | 150 | 163 | 8.00 | 7.36 | -- | 14861.7 | 12.3 |
| DSSC (multi-modal) | 300 | 188 | 13.66 | 12.77 | 41.4% | 167.9 | 1013.1 |
| DSSC (text-only) | 300 | 161 | 15.95 | 14.91 | 49.8% | 214.5 | 1931.4 |
| exact_vcp (multi-modal) | 300 | 321 | 8.00 | 7.48 | -- | 29529.9 | 23.8 |
| DSSC (multi-modal) | 500 | 310 | 13.65 | 12.90 | 41.4% | 209.0 | 1580.0 |
| DSSC (text-only) | 500 | 265 | 15.97 | 15.09 | 49.9% | 296.9 | 3066.4 |
| exact_vcp (multi-modal) | 500 | 529 | 8.00 | 7.56 | -- | 51703.8 | 40.7 |

**Coded versus user capacity.** "Coded bits/carrier" is the recorded field: `8 * codeword bytes / carrier count`, so it counts the 5-byte CRC frame and the 8 parity bytes. "User bits/carrier" is derived as `8 * payload bytes / carrier count` and is the payload actually available to the sender. The DSSC multi-modal advantage holds on both columns, and the gap narrows as the payload grows.

**Key Result:** Multi-modal DSSC achieves 13.50–13.66 coded bits/carrier (mean 13.60) with 40.7–41.4% carrier reduction. Text-only DSSC achieves 15.43–15.97 coded bits/carrier and uses 48.1–49.9% fewer carriers than the one-per-codeword-byte exact_vcp reference. exact_vcp is capped at 8.0 coded bits/carrier. Its encode latency column measures the shipped reference path, which scores semantic candidates for every byte; it is not a codec-only lower bound. The exact_vcp row is the all-modality reference mapper, not a strict text-only baseline, because the mapper falls back to image or audio when a text cluster lacks a requested symbol.

### 6.2 Permutation Benchmark

| Candidate Pool (N) | Mean (ms) | p50 (ms) | Determinism | Key Sensitivity (ρ) | Throughput (perm/s) |
|-------------------|-----------|----------|-------------|---------------------|---------------------|
| 1,000 | 0.0575 | 0.0576 | PASS | +0.043505 | 17,378 |
| 5,000 | 0.1673 | 0.1749 | PASS | -0.016776 | 5,978 |
| 10,000 | 0.1575 | 0.1292 | PASS | +0.004498 | 6,350 |
| 20,000 | 0.3013 | 0.3411 | PASS | -0.002327 | 3,319 |
| 35,000 | 0.4056 | 0.3611 | PASS | -0.001982 | 2,466 |

**0.4056 ms per 35,000-candidate permutation**, about 616x faster than the roughly 250 ms pure-Python HMAC loop it replaced. Key-sensitivity correlation stays within ±0.044 at every size with no trend. The 20,000 row has a p50 slightly above its mean and the 35,000 row a p50 slightly below, so dispersion is small; the p50 column is the better central estimate under scheduler noise. These are wall-clock measurements on a shared machine and shift between runs; the key-sensitivity column does not.

### 6.3 ECC Recovery Under Carrier Substitution

| Substitution Rate (%) | Recovery Rate (%) | 95% CI | Mean BER (%) | Mean Fixed Errors | End-to-End Latency (ms) |
|------------------|-------------------|--------|--------------|-------------------|------------------------|
| 0.0% | 100.0% | [83.9, 100.0] | 0.00% | 0.0 B | 585.2 |
| 5.0% | 70.0% | [48.1, 85.5] | 30.00% | 2.7 B | 554.9 |
| 10.0% | 0.0% | [0.0, 16.1] | 100.00% | 0.0 B | 533.2 |
| 15.0% | 0.0% | [0.0, 16.1] | 100.00% | 0.0 B | 529.9 |
| 20.0% | 0.0% | [0.0, 16.1] | 100.00% | 0.0 B | 511.9 |
| 25.0% | 0.0% | [0.0, 16.1] | 100.00% | 0.0 B | 444.0 |

**Substitution is tolerated to the algebraic limit.** Recovery is 100.0% at 0%, 70.0% at 5%, and 0% from 10% through 25% in this campaign. The 5% interval [48.1, 85.5] is wide because this arm uses 20 trials per rate; the 80-trial Component 4c estimate is 61.2% [50.3, 71.2]. Section 6.5 decomposes the cliff and shows it is the damaged-byte count crossing t = 4, not a codec defect; dropout, the harder channel, is covered there as well.

### 6.4 Key Separation (Cryptographic)

| Test Type | Trials | Accepted | Exact | FAR | FAR 95% Upper |
|-----------|--------|----------|-------|-----|---------------|
| Random Keys | 100 | 0 | 0 | 0.000000 | 3.70% |
| 1-Bit Flipped Keys | 256 | 0 | 0 | 0.000000 | 1.48% |

**Zero false acceptances in 356 unauthorized attempts** (100 random keys, 256 single-bit flips). A false acceptance is any CRC-passing decode under an unauthorized key, whether or not the returned plaintext matches the original. The observed zero counts do not resolve the CRC-16 bound: the Wilson 95 percent upper bounds are 3.70 percent for the 100-key arm and 1.48 percent for the 256-bit-flip arm. The per-attempt acceptance bound remains theoretical, about 2⁻¹⁶ for the CRC path; the 2⁻²⁵⁶ bound applies to the opt-in HMAC frame variant, which this campaign does not use.

### 6.5 ECC Failure Anatomy (Component 4c)

Component 4c repeats the sweep with 80 trials per rate, a bit-exact instrumentation mirror that counts damaged codeword bytes per trial, and Wilson intervals, so the shape of the Section 6.3 curve is explained rather than merely recorded.

| Substitution Rate | Exact | Accepted | 95% CI | Mean Damaged Bytes |
|-------------------|-------|----------|--------|--------------------|
| 0% | 80/80 | 80/80 | [95.4, 100.0] | 0.0 |
| 2% | 80/80 | 80/80 | [95.4, 100.0] | 2.2 |
| 3% | 80/80 | 80/80 | [95.4, 100.0] | 2.2 |
| 4% | 80/80 | 80/80 | [95.4, 100.0] | 2.1 |
| 5% | 49/80 | 49/80 | [50.3, 71.2] | 4.3 |
| 6% | 52/80 | 52/80 | [54.1, 74.5] | 4.4 |
| 8% | 0/80 | 0/80 | [0.0, 4.6] | 6.6 |
| 10% | 0/80 | 0/80 | [0.0, 4.6] | 8.6 |

**Substitution:** one corrupted carrier damages about 2.2 codeword bytes, so recovery collapses as the damage crosses the t = 4 byte budget: 100% through 4%, 61.2% at 5%, 65.0% at 6%, and 0% from 8%. Across all 640 substitution trials, exact recovery happened if and only if the measured damage stayed at or below t, with zero counterexamples. The separate acceptance count, which includes any CRC-passing decode even with the wrong plaintext, accepted no trial above the limit in the substitution arm. The substitution condition replaces carriers with random corpus IDs; same-symbol replacements are no-ops and the artifact records the symbol-change share.

**Dropout:** deleting carriers is the harder channel. A deletion early in the list shifts every later position and damages about 32.7 codeword bytes, so recovery is 0% at every recorded rate with at least one deletion. A few trials at 2, 3, and 4 percent have at most four damaged bytes yet still fail because the received list is short rather than shifted; damaged bytes are not a sufficient statistic for dropout. The erasure-bound replay measures the latent headroom: if positions were known, the recorded damage would stay correctable through 6% at 100%, 98.8% at 8%, and 51.2% at 10%, because the erasure budget is 8 bytes rather than 4. The wire format carries no per-carrier sequence markers, so this headroom is not reachable end to end today.

The first rebuilt 4c run, before the decoder fix, recorded 41 CRC-passing wrong-plaintext decodes across the 320 dropout trials at 2 through 5 percent deletion: 8.8% at 2 percent, 8.8% at 3 percent, 12.5% at 4 percent, and 21.2% at 5 percent, while exact recovery was zero at every one of those rates. A deletion moved the frame marker away from byte zero, Reed-Solomon still returned a codeword, and the legacy raw fallback accepted the unframed bytes as a successful decode. The decoder now requires framed payloads when ECC is enabled, and legacy raw payloads remain decodable only when ECC is off. The pre-hardening artifact is archived at `storage/benchmarks/archive/component_4c_ecc_failure_results_pre_hardening.json`; the current artifact records zero false acceptances, and exact recovery is unchanged.

### 6.6 Verified Round-Trip Results

Round-trip accuracy is enforced by assertion rather than summarized in a separate table. The 4a campaign asserts exact message equality on all 75 clean-channel encode-decode pairs across five payload sizes and three modes, and the clean-channel rows of 4b and 4c recover 20/20 and 80/80 respectively. Every recorded clean-channel round trip returned the original payload exactly.

### 6.7 LaTeX Table Outputs

#### Table 4a: Codec Capacity

```latex
% Component 4a: Codec Capacity - DSSC vs exact_vcp
\begin{table*}[htbp]
\centering
\caption{Codec capacity: DSSC vs exact\_vcp with matched framing and Reed-Solomon ECC. Latency columns are reference-implementation wall-clock measurements, not codec-only lower bounds. (n = 5 payload sizes x 3 modes; scope = coded bits/carrier including framing and RS parity; user-payload density is lower; commit = 536808a8f8c1046e3802cf9ef094ef54d93a9908)}
\label{tab:codec_capacity}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Mode & Payload (B) & Carriers & Bits/Carrier & Reduction (\%) & Encode (ms, ref.) & Decode (ms, ref.) \\
\midrule
DSSC (multi-modal) & 14 & 16 & 13.50 & 40.7\% & 106.0 & 181.3 \\
DSSC (text-only) & 14 & 14 & 15.43 & 48.1\% & 89.3 & 243.3 \\
exact\_vcp (multi-modal) & 14 & 27 & 8.00 & -- & 1784.6 & 2.2 \\
DSSC (multi-modal) & 50 & 37 & 13.62 & 41.3\% & 113.6 & 280.4 \\
DSSC (text-only) & 50 & 32 & 15.75 & 49.2\% & 111.2 & 442.1 \\
exact\_vcp (multi-modal) & 50 & 63 & 8.00 & -- & 5232.6 & 5.0 \\
DSSC (multi-modal) & 150 & 96 & 13.58 & 41.1\% & 119.0 & 542.7 \\
DSSC (text-only) & 150 & 82 & 15.90 & 49.7\% & 153.6 & 1009.4 \\
exact\_vcp (multi-modal) & 150 & 163 & 8.00 & -- & 14861.7 & 12.3 \\
DSSC (multi-modal) & 300 & 188 & 13.66 & 41.4\% & 167.9 & 1013.1 \\
DSSC (text-only) & 300 & 161 & 15.95 & 49.8\% & 214.5 & 1931.4 \\
exact\_vcp (multi-modal) & 300 & 321 & 8.00 & -- & 29529.9 & 23.8 \\
DSSC (multi-modal) & 500 & 310 & 13.65 & 41.4\% & 209.0 & 1580.0 \\
DSSC (text-only) & 500 & 265 & 15.97 & 49.9\% & 296.9 & 3066.4 \\
exact\_vcp (multi-modal) & 500 & 529 & 8.00 & -- & 51703.8 & 40.7 \\
\bottomrule
\end{tabular}}
\end{table*}
```

#### Table 4b: Codec Robustness

```latex
% Component 4b: Codec Robustness, Permutation Scaling & Key Separation
\begin{table*}[htbp]
\centering
\caption{DSSC codec robustness: permutation scaling, RS-ECC resilience, and key separation. Intervals are Wilson 95 percent. (n = 5 permutation pool sizes, 6 substitution rates, 356 wrong-key trials; scope = clean-channel codec; substitution recovery and key separation; dropout is not covered here; commit = 536808a8f8c1046e3802cf9ef094ef54d93a9908)}
\label{tab:codec_robustness}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\multicolumn{6}{c}{\textbf{Sub-Millisecond HMAC-DRBG Permutation Scaling}} \\
Candidate Pool ($N$) & Mean (ms) & p50 (ms) & Determinism & Key Sensitivity ($\rho$) & Throughput (perm/s) \\
\midrule
1,000 & 0.0575 & 0.0576 & \checkmark & +0.043505 & 17,378 \\
5,000 & 0.1673 & 0.1749 & \checkmark & -0.016776 & 5,978 \\
10,000 & 0.1575 & 0.1292 & \checkmark & +0.004498 & 6,350 \\
20,000 & 0.3013 & 0.3411 & \checkmark & -0.002327 & 3,319 \\
35,000 & 0.4056 & 0.3611 & \checkmark & -0.001982 & 2,466 \\
\midrule
\multicolumn{6}{c}{\textbf{Algebraic RS-ECC Resilience Under Carrier Substitution}} \\
Substitution Rate (\%) & Recovery (\%) & 95\% CI & Mean BER (\%) & Mean Fixed Errors & Latency (ms) \\
\midrule
0.0\% & 100.0\% & [83.9, 100.0] & 0.00\% & 0.0 B & 585.2 \\
5.0\% & 70.0\% & [48.1, 85.5] & 30.00\% & 2.7 B & 554.9 \\
10.0\% & 0.0\% & [0.0, 16.1] & 100.00\% & 0.0 B & 533.2 \\
15.0\% & 0.0\% & [0.0, 16.1] & 100.00\% & 0.0 B & 529.9 \\
20.0\% & 0.0\% & [0.0, 16.1] & 100.00\% & 0.0 B & 511.9 \\
25.0\% & 0.0\% & [0.0, 16.1] & 100.00\% & 0.0 B & 444.0 \\
\midrule
\multicolumn{6}{c}{\textbf{Cryptographic Key Separation \& Unforgeability}} \\
Test Type & Trials & Accepted & Exact & FAR & FAR 95\% Upper \\
\midrule
Random Keys & 100 & 0 & 0 & 0.000000 & 0.036995 \\
1-Bit Flipped Keys & 256 & 0 & 0 & 0.000000 & 0.014784 \\
\bottomrule
\end{tabular}}
\end{table*}
```

#### Table 4c: ECC Failure Anatomy

```latex
% Component 4c: ECC Failure-Mode Decomposition and Channel Loss Anatomy
\begin{table*}[htbp]
\centering
\caption{ECC failure anatomy: substitution versus dropout versus algebraic limits. Codeword is RS(63,55) correcting 4 byte errors or 8 known erasures. Exact is plaintext recovery; accepted is any CRC-passing decode. No CRC-passing wrong plaintext occurred in the recorded trials after the framed-payload hardening. (n = 8 substitution and 6 dropout conditions, 40 repeats per algebraic point; scope = post-hardening artifact; framed payloads required when ECC is enabled; commit = 536808a8f8c1046e3802cf9ef094ef54d93a9908)}
\label{tab:ecc_failure_modes}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Condition & Loss & Exact & Accepted & 95\% CI & Damaged bytes & Per affected carrier \\
\midrule
Substitution & 0\% & 100.0\% & 100.0\% & [95.4, 100.0] & 0.0 & 0.00 \\
Substitution & 2\% & 100.0\% & 100.0\% & [95.4, 100.0] & 2.2 & 2.17 \\
Substitution & 3\% & 100.0\% & 100.0\% & [95.4, 100.0] & 2.2 & 2.20 \\
Substitution & 4\% & 100.0\% & 100.0\% & [95.4, 100.0] & 2.1 & 2.10 \\
Substitution & 5\% & 61.2\% & 61.2\% & [50.3, 71.2] & 4.3 & 2.17 \\
Substitution & 6\% & 65.0\% & 65.0\% & [54.1, 74.5] & 4.4 & 2.19 \\
Substitution & 8\% & 0.0\% & 0.0\% & [0.0, 4.6] & 6.6 & 2.19 \\
Substitution & 10\% & 0.0\% & 0.0\% & [0.0, 4.6] & 8.6 & 2.16 \\
Dropout & 0\% & 100.0\% & 100.0\% & [95.4, 100.0] & 0.0 & 0.00 \\
Dropout & 1\% & 100.0\% & 100.0\% & [95.4, 100.0] & 0.0 & 0.00 \\
Dropout & 2\% & 0.0\% & 0.0\% & [0.0, 4.6] & 32.7 & 32.67 \\
Dropout & 3\% & 0.0\% & 0.0\% & [0.0, 4.6] & 32.9 & 32.91 \\
Dropout & 4\% & 0.0\% & 0.0\% & [0.0, 4.6] & 33.7 & 33.67 \\
Dropout & 5\% & 0.0\% & 0.0\% & [0.0, 4.6] & 44.5 & 22.26 \\
\midrule
\multicolumn{7}{c}{\textbf{Algebraic layer, known positions}} \\
Injected bytes & Error success & Erasure success & & & & \\
0 & 100\% & 100\% & & & & \\
1 & 100\% & 100\% & & & & \\
2 & 100\% & 100\% & & & & \\
3 & 100\% & 100\% & & & & \\
4 & 100\% & 100\% & & & & \\
5 & 0\% & 100\% & & & & \\
6 & 0\% & 100\% & & & & \\
7 & 0\% & 100\% & & & & \\
8 & 0\% & 100\% & & & & \\
9 & 0\% & 0\% & & & & \\
10 & 0\% & 0\% & & & & \\
11 & 0\% & 0\% & & & & \\
12 & 0\% & 0\% & & & & \\
\bottomrule
\end{tabular}}
\end{table*}
```

The 4c table reports the post-hardening result. The pre-hardening artifact in `storage/benchmarks/archive/` records 41 CRC-passing wrong-plaintext decodes across the 320 dropout trials at 2 through 5 percent deletion; the decoder now requires framed payloads when ECC is enabled.

---

## 7. Component 5: Codec Latency, Throughput & Direct-Engine Concurrency (Detailed)

### 7.1 End-to-End Codec Latency & Throughput

| Payload Size (Bytes) | Carrier Count (N) | Encode Time (μ ± σ, ms) | Decode Time (μ ± σ, ms) | Total Latency (ms) | Encode Rate (KB/s) | Carrier Speed (carriers/s) |
|---------------------|-------------------|-------------------------|-------------------------|---------------------|-------------------|----------------------------|
| 32 B | 27 | 107.1 ± 10.6 | 222.6 ± 7.6 | 329.7 | 0.29 | 81.9 |
| 64 B | 46 | 111.0 ± 10.3 | 321.7 ± 16.2 | 432.7 | 0.56 | 106.3 |
| 128 B | 83 | 123.2 ± 12.7 | 484.7 ± 16.1 | 607.9 | 1.01 | 136.5 |
| 256 B | 163 | 158.5 ± 13.5 | 856.6 ± 11.9 | 1015.1 | 1.58 | 160.6 |
| 512 B | 317 | 206.3 ± 10.2 | 1590.0 ± 19.1 | 1796.2 | 2.42 | 176.5 |

#### Stage Breakdown (Encode)

| Payload | ECC Framing (ms) | Permutation (ms) | FAISS Search (ms) |
|---------|------------------|------------------|-------------------|
| 32 B | 0.02 | 0.07 | 107.12 |
| 64 B | 0.04 | 0.19 | 109.40 |
| 128 B | 0.06 | 0.07 | 114.16 |
| 256 B | 0.12 | 0.08 | 147.34 |
| 512 B | 0.26 | 0.07 | 193.28 |

#### Stage Breakdown (Decode)

| Payload | Centroid Lookup (ms) | Inv Permutation (ms) | ECC Syndrome (ms) |
|---------|---------------------|---------------------|-------------------|
| 32 B | 242.10 | 0.07 | 0.04 |
| 64 B | 342.02 | 0.19 | 0.06 |
| 128 B | 468.09 | 0.07 | 0.09 |
| 256 B | 861.23 | 0.08 | 0.16 |
| 512 B | 1642.00 | 0.07 | 0.29 |

**Dominant cost:** FAISS search (encode) and centroid lookup (decode), both scale with carrier count. The FAISS and centroid rows are residual attributions, computed as full encode or decode time minus ECC and permutation. They are not direct instrumentation of the FAISS call. Permutation overhead stays near 0.07 ms per message in this sweep; the largest recorded pool in Component 4b, 35,000 candidates, costs 0.4056 ms per permutation.

### 7.2 Direct-Engine Concurrency, Latency Percentiles & Payload Isolation

These rows measure `SemanticEngine.encode` and `.decode` directly. They do not call FastAPI, `/api/transmit`, `StealthScheduler`, or the API session map. Latency starts inside the worker after the task begins, so it excludes queue wait, HTTP, scheduler, and transmit delays. Throughput is completed sessions per second, where each session is one encode and one decode.

| Worker Threads | Total Sessions Evaluated | Throughput (sessions/s) | p50 Latency (ms) | p90 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Success Rate (%) | Observed Swap Rate (%) |
|-------------------|-------------------------|-------------------|------------------|------------------|------------------|------------------|------------------|---------------------|
| 10 | 100 | 2.00 | 4976.0 | 5502.0 | 5706.0 | 5978.9 | 100.0% | 0.000% |
| 20 | 100 | 1.95 | 10089.0 | 11220.0 | 11455.0 | 11730.2 | 100.0% | 0.000% |
| 50 | 100 | 1.89 | 24367.0 | 26989.0 | 27598.0 | 27948.6 | 100.0% | 0.000% |

The direct-engine test observed zero swapped marker payloads at all concurrency levels. The success column checks exact plaintext reconstruction, so corrupted payloads without the marker count as failures rather than swaps. The API transmit path is separate: the server now admits concurrent transmissions up to a bounded limit and answers HTTP 429 beyond it, and this benchmark does not exercise that path.

### 7.3 LaTeX Table Output

```latex
% Component 5: Computational Performance & System Scalability
\begin{table*}[htbp]
\centering
\caption{\textbf{Codec Latency, Encoding Throughput, and Direct-Engine Thread Concurrency.} Empirical benchmark of the DCASS Dynamic Semantic State-Space Coding (DSSC) pipeline across payload sizes and parallel worker-thread workloads. The concurrency rows are direct SemanticEngine encode/decode service times, not FastAPI or transmit latencies. (n = 5 payload sizes, 3 thread levels x 100 sessions each; scope = direct SemanticEngine encode/decode only; the API transmit path is not exercised; commit = 536808a8f8c1046e3802cf9ef094ef54d93a9908)}
\label{tab:system_performance}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\multicolumn{7}{l}{\textbf{Part A: End-to-End Codec Latency \& Throughput Across Payload Sizes}} \\
\midrule
Payload Size & Carrier Count & Encode Time & Decode Time & Total Latency & Encode Rate & Carrier Speed \\
(Bytes) & ($N$) & ($\mu \pm \sigma$, ms) & ($\mu \pm \sigma$, ms) & (ms) & (KB/s) & (carriers/s) \\
\midrule
32 B & 27 & 107.1 $\pm$ 10.6 & 222.6 $\pm$ 7.6 & 329.7 & 0.29 & 81.9 \\
64 B & 46 & 111.0 $\pm$ 10.3 & 321.7 $\pm$ 16.2 & 432.7 & 0.56 & 106.3 \\
128 B & 83 & 123.2 $\pm$ 12.7 & 484.7 $\pm$ 16.1 & 607.9 & 1.01 & 136.5 \\
256 B & 163 & 158.5 $\pm$ 13.5 & 856.6 $\pm$ 11.9 & 1015.1 & 1.58 & 160.6 \\
512 B & 317 & 206.3 $\pm$ 10.2 & 1590.0 $\pm$ 19.1 & 1796.2 & 2.42 & 176.5 \\
\bottomrule
\end{tabular}}

\vspace{0.6em}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\multicolumn{9}{l}{\textbf{Part B: Direct-Engine Thread Concurrency, Latency Percentiles \& Payload Isolation}} \\
\midrule
Worker & Total Sessions & Throughput & $p_{50}$ Latency & $p_{90}$ Latency & $p_{95}$ Latency & $p_{99}$ Latency & Success Rate & Observed Swap \\
Threads & Evaluated & (sessions/s) & (ms) & (ms) & (ms) & (ms) & (\%) & Rate (\%) \\
\midrule
10 & 100 & 2.00 & 4975.6 & 5502.2 & 5705.7 & 5978.9 & \textbf{100.0\%} & \textbf{0.000\%} \\
20 & 100 & 1.95 & 10089.1 & 11220.0 & 11454.7 & 11730.2 & \textbf{100.0\%} & \textbf{0.000\%} \\
50 & 100 & 1.89 & 24367.5 & 26988.9 & 27597.6 & 27948.6 & \textbf{100.0\%} & \textbf{0.000\%} \\
\bottomrule
\end{tabular}}
\end{table*}
```

---

## 8. Component 6: Publication Artifacts

This section is generated by `scripts/analysis/generate_publication_inventory.py`
from the filesystem, so it cannot drift from what is actually on disk. The
full mapping, including the source script and claim ID for every artifact,
is in [`06_PUBLICATION_ARTIFACTS_AND_LATEX.md`](06_PUBLICATION_ARTIFACTS_AND_LATEX.md).

| Table | File | Source script | Claim IDs | Description |
|-------|------|---------------|-----------|-------------|
| Table 1 | `storage/benchmarks/tables/table1_steganalysis.tex` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01`, `C1-SCOPE-02` | Media-layer stegananalysis resistance on the image-level 3-fold protocol |
| Table 2 | `storage/benchmarks/tables/table2_distributional_drift.tex` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01`, `C2-KS-02` | Per-modality distributional drift against a disjoint corpus reference |
| Table 2b | `storage/benchmarks/tables/table2b_cluster_uniformity.tex` | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-UNIFORM-03`, `C2-MECH-04` | Transmitted cluster usage against the pipeline's own mechanism null |
| Table 3 | `storage/benchmarks/tables/table3_traffic_evasion.tex` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06`, `C3-CI-03`, `C3-WARDEN-02` | Warden interception rates and timing fidelity; the second column is the not-flagged rate |
| Table 3b | `storage/benchmarks/tables/table3b_warden_score_quantiles.tex` | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02`, `C3-STRUCT-03` | Stored warden critic score quantiles per scheduler |
| Table 4a | `storage/benchmarks/tables/table4a_capacity.tex` | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01`, `C4-USER-02`, `C4-REDUCTION-03` | DSSC against exact_vcp coded capacity and carrier reduction |
| Table 4b | `storage/benchmarks/tables/table4b_codec_robustness.tex` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04` | Permutation scaling, RS-ECC substitution resilience, and key separation |
| Table 4c | `storage/benchmarks/tables/table4c_ecc_failure_modes.tex` | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | ECC failure anatomy: substitution, dropout, and the algebraic limits |
| Table 5 | `storage/benchmarks/tables/table5_system_performance.tex` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01`, `C5-API-02` | Direct-engine codec latency, throughput, and thread concurrency |

### 8.2 Generated Plots

| Plot | File | Source script | Claim IDs | Description |
|------|------|---------------|-----------|-------------|
| Payload Sweep (vector) | `storage/benchmarks/plots/fig1_payload_sweep.pdf` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Vector version of the payload sweep figure |
| Payload Sweep | `storage/benchmarks/plots/fig1_payload_sweep.png` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Detection accuracy against payload rate from 1.0 down to 0.1 bpp |
| ROC Curves (vector) | `storage/benchmarks/plots/fig1_roc_curves.pdf` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Vector version of the ROC and accuracy figure |
| ROC Curves | `storage/benchmarks/plots/fig1_roc_curves.png` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Detection accuracy and ROC AUC, positive control against DCASS carriers |
| Distributional Stats | `storage/benchmarks/plots/fig2_distributional_stats.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Energy, MMD squared, and W1 panels with per-draw dots |
| Concurrency Scaling | `storage/benchmarks/plots/fig5_concurrency_scaling.png` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Throughput and latency percentiles under direct-engine worker-thread load |
| Latency Breakdown | `storage/benchmarks/plots/fig5_latency_breakdown.png` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Residual latency breakdown across payload sizes |
| Cluster Usage | `storage/benchmarks/plots/fig_2b_cluster_usage.png` | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-MECH-04` | Transmitted cluster usage against the mechanism null |
| Warden Scores | `storage/benchmarks/plots/fig_3b_warden_scores.png` | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02` | Stored warden score distributions with both PPO operating points |
| Capacity Comparison | `storage/benchmarks/plots/fig_4a_capacity_comparison.png` | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01` | DSSC against exact_vcp coded capacity |
| ECC Recovery | `storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-ECC-07` | Recovery rate against carrier substitution |
| Key Separation | `storage/benchmarks/plots/fig_4b_key_separation.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04` | Wrong-key rejection and single-bit avalanche correlations |
| Permutation Scaling | `storage/benchmarks/plots/fig_4b_permutation_scaling.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-CODED-01` | HMAC-DRBG latency and throughput against candidate pool size |
| ECC Failure Modes | `storage/benchmarks/plots/fig_4c_ecc_failure_modes.png` | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Substitution and dropout recovery with the erasure-bound replay |
| Circadian Diurnal | `storage/benchmarks/plots/fig_circadian_diurnal.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-CI-03` | Diurnal conditioning check at hour 14 against hour 03 under one identical mapping |
| Inter-arrival CDF | `storage/benchmarks/plots/fig_interarrival_cdf.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Scheduler inter-arrival distributions against the synthetic surrogate |
| Interception Rates | `storage/benchmarks/plots/fig_interception_rates.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Warden interception and not-flagged rates per scheduler |
| KS Heatmap | `storage/benchmarks/plots/fig_ks_heatmap.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-KS-02` | KS statistic across 512 dimensions with BH and family-wise annotations |
| Manifold t-SNE | `storage/benchmarks/plots/fig_manifold_tsne.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Disjoint reference against carrier embedding visualization |

### 8.3 Result Artifacts

| File | Component | Kind | Source script | Claim IDs | Description |
|------|-----------|------|---------------|-----------|-------------|
| `storage/benchmarks/component_1_fold0.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 0 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_fold1.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 1 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_fold2.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 2 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_summary.json` | Component 1 | results | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01`, `C1-SCOPE-02` | Aggregated three-fold steganalysis results and the protocol record |
| `storage/benchmarks/component_1_summary_manifest.json` | Component 1 | run manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_2_results.json` | Component 2 | results | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01`, `C2-KS-02` | Per-modality distributional drift against a disjoint corpus reference |
| `storage/benchmarks/component_2_results_manifest.json` | Component 2 | run manifest | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_2b_cluster_usage_results.json` | Component 2b | results | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-UNIFORM-03`, `C2-MECH-04` | Transmitted cluster histogram against the uniform and mechanism nulls |
| `storage/benchmarks/component_2b_cluster_usage_results_manifest.json` | Component 2b | run manifest | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-MECH-04` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_3_results.json` | Component 3 | results | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06`, `C3-CI-03` | Scheduler interception rates, timing statistics, and the warden self-test |
| `storage/benchmarks/component_3_results_baseline.json` | Component 3 | historical run | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Earlier component 3 run, scored by the superseded warden, holding the historical PPO operating point |
| `storage/benchmarks/component_3_results_manifest.json` | Component 3 | run manifest | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-RERUN-05` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_3b_warden_score_summary.json` | Component 3b | results | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02` | Quantiles of the stored per-trial warden scores |
| `storage/benchmarks/component_3b_warden_score_summary_manifest.json` | Component 3b | run manifest | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-RERUN-05` | Command, git state, seeds, and input and output hashes |
| `storage/benchmarks/component_4a_capacity_results.json` | Component 4a | results | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01`, `C4-REDUCTION-03` | DSSC against exact_vcp coded capacity and carrier reduction |
| `storage/benchmarks/component_4a_capacity_results_manifest.json` | Component 4a | run manifest | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_4b_robustness_results.json` | Component 4b | results | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04`, `C4-ECC-07` | Permutation scaling, substitution resilience, and key separation |
| `storage/benchmarks/component_4b_robustness_results_manifest.json` | Component 4b | run manifest | `scripts/analysis/benchmark_4b_robustness.py` | `C4-ECC-07` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_4c_ecc_failure_results.json` | Component 4c | results | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Post-hardening ECC failure anatomy |
| `storage/benchmarks/component_4c_ecc_failure_results_manifest.json` | Component 4c | run manifest | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_5_results.json` | Component 5 | results | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01`, `C5-API-02` | Direct-engine latency, throughput, and thread concurrency |
| `storage/benchmarks/component_5_results_manifest.json` | Component 5 | run manifest | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/critic_input_gradient_probe.json` | Component 3 | results | `scripts/analysis/probe_critic_input_gradient.py` | `C3-GEN-07` | The critic's input gradient for the shipped post-LN stack and for a pre-LN variant, at initialization and after training, which puts the architectural hypothesis out |
| `storage/benchmarks/critic_sensitivity_bound10.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-epoch bounded-critic arm at score bound 10, epoch by epoch: after one over-Lipschitz epoch the critic scores real and generated traffic identically from epoch 3 through epoch 6 |
| `storage/benchmarks/critic_sensitivity_bound100.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-epoch bounded-critic arm at score bound 100, epoch by epoch: three epochs of 38-point swings, then the same dead state from epoch 4 through the final epoch, which is collapsed |
| `storage/benchmarks/critic_sensitivity_final_run.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-07` | The first repaired run's critic, epoch by epoch, in which the critic still collapses |
| `storage/benchmarks/critic_sensitivity_lambda1000.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The ten-epoch lambda_gp=1000 probe, epoch by epoch: the constraint holds in every epoch, so the shipped weight of 100 is the smallest measured sufficient weight, not the only one |
| `storage/benchmarks/critic_sensitivity_repaired_run.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The shipped run's critic, epoch by epoch: its input gradient, the penalty it implies, and the score gap between real and generated traffic |
| `storage/benchmarks/critic_sensitivity_stronger_critic.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-critic-step arm at the old lambda_gp of 10, epoch by epoch: the critic's score gap swings across 33 points and prefers the generated traffic in 22 of 30 epochs, and the generator's delay median sweeps 0.9 to 10.7 seconds |
| `storage/benchmarks/generator_detector_adversary_probe.json` | Component 3 | results | `scripts/analysis/probe_generator_detector_adversary.py` | `C3-GEN-07` | The generator trained directly against the detector's bot probability for 40 epochs, which shows a bounded adversary saturates instead of discriminating |
| `storage/benchmarks/generator_path_defects.json` | Component 3 | results | `scripts/analysis/measure_generator_path_defects.py` | `C3-GEN-07` | The unit mismatch between the critic's two training sides, and the delay mapping's conditioning of the generator's output |
| `storage/benchmarks/generator_path_repair.json` | Component 3 | results | `scripts/analysis/verify_generator_path.py` | `C3-GEN-08` | The repaired generator path: the delay mapping is no longer saturated, the gradient reaches the delay head, and the critic's input gradient survives training |
| `storage/benchmarks/rl_stealth_reward_probe.json` | Component 3 | results | `scripts/analysis/measure_rl_stealth_reward.py` | `C3-TRAFFIC-06` | The reward the RL environment assigns to bot-like and human-like traffic under the shipped WGAN-GP critic and under a trained detector |
| `storage/benchmarks/soft_branch_design_comparison.json` | Component 3 | results | `scripts/analysis/compare_soft_branch_designs.py` | `C3-DETECT-06` | Each candidate soft-branch score thresholded at a 5 percent false-positive rate on held-out human traffic, and the resulting separation |
| `storage/benchmarks/sr_net_corrected_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |
| `storage/benchmarks/traffic_detector_eval_results.json` | Component 3 | results | `scripts/analysis/train_traffic_detector.py` | `C3-DETECT-06` | The held-out evaluator's training summary, held-out AUC, and threshold calibration |
| `storage/benchmarks/traffic_detector_generalization.json` | Component 3 | results | `scripts/analysis/check_detector_generalization.py` | `C3-DETECT-06` | The traffic detector's bot probability on six traffic populations, including a human generator it has never seen |
| `storage/benchmarks/traffic_detector_results.json` | Component 3 | results | `scripts/analysis/train_traffic_detector.py` | `C3-DETECT-06` | The traffic detector's training summary, held-out AUC, and threshold calibration |
| `storage/benchmarks/traffic_ks_null_band.json` | Component 3 | results | `scripts/analysis/measure_ks_null.py` | `C3-DETECT-06` | The per-trial KS distances a burst drawn from the human reference itself produces, which is the band a matching sampler lands in |
| `storage/benchmarks/warden_diagnostic.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden.py` | `C3-WARDEN-02` | Warden outputs against six deliberately different delay sequences |
| `storage/benchmarks/warden_diagnostic_components.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden_components.py` | `C3-WARDEN-02`, `C3-STRUCT-03` | Component swap isolating the transformer collapse and the dead anomaly head |
| `storage/benchmarks/warden_diagnostic_layers.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden_layers.py` | `C3-WARDEN-02` | Per-layer signal spread showing where the input sensitivity dies |
| `storage/benchmarks/wgan_training_data_probe.json` | Component 3 | results | `scripts/analysis/probe_wgan_training_data.py` | `C3-GEN-07` | The same WGAN-GP training replayed against the repository's synthetic fallback and against the storage traffic corpus, six epochs each, which rules the training data out as the cause of the collapse |
| `storage/benchmarks/ye_net_improved_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |
| `storage/benchmarks/zhu_net_improved_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |

### 8.4 Inventory and compile checks

Three checks keep this inventory honest:

| Check | What it verifies | Command |
|-------|------------------|---------|
| Inventory | Every table, plot, and artifact on disk appears here, and nothing here is missing from disk | `.venv/bin/python scripts/analysis/check_artifact_inventory.py` |
| LaTeX compile | Every table compiles under the root-relative `\input` paths the paper uses | `.venv/bin/python scripts/analysis/check_latex_compile.py` |
| Table widths | Which tables overflow a single-column text block | `.venv/bin/python scripts/analysis/check_latex_table_widths.py` |

The compile check needs a LaTeX engine. With `tectonic` on the path it
reports the result directly; without one it reports that it could not run
rather than claiming a pass.

The LaTeX integration guide, with the full `\input` and `\includegraphics`
blocks and the table-width notes, is in
[`06_PUBLICATION_ARTIFACTS_AND_LATEX.md`](06_PUBLICATION_ARTIFACTS_AND_LATEX.md).

## 9. Hardware & Software Environment

### 9.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| CPU | AMD Ryzen 7 / Intel Core i7 (8+ cores) |
| RAM | 32 GB DDR5 |
| Storage | NVMe SSD |

### 9.2 Software

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.14+ |
| CUDA | 12.1+ |
| NumPy | 1.26+ |
| FAISS | 1.8+ |
| ONNX Runtime | 1.18+ |

### 9.3 Memory Management

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 10. Conclusions & Future Work

### 10.1 Conclusions

The recorded evidence supports a narrower set of conclusions than the original headline table implied:

1. **Media-Layer Detectability (Negative Control):** DCASS carriers are indistinguishable from natural media at the score level under the three gated image detectors: measured AUCs of 0.4973 to 0.4993 with every 95 percent interval covering 0.5000, and balanced accuracy of 0.50 within an interval. This is expected because the carriers are bit-identical corpus files. It removes the pixel/sample embedding signal; it does not address selection or timing.

2. **Semantic Distribution:** energy distance and MMD² see image and text carriers about one ten-thousandth away from a uniform corpus draw at permutation p 0.009 and 0.004, which is the public family-restriction mechanism, with nothing resolved for audio at p 0.57; zero of 1,536 KS coordinates reject after Benjamini-Hochberg control; cluster entropy reaches 97.6% of maximum on slots and 97.3% on unique media, with the deficit fully accounted for by the modality mixture identity. The measurable component is consistent with the pipeline's public mechanism; this is not uniform indistinguishability.

3. **Traffic Analysis Evasion: Not Demonstrated, and Now Measured as Detected:** the recorded composite warden flags the static, WGAN and Hybrid schedules at 100% and Poisson at 54%, while the retrained PPO arm is not flagged at 0%. The earlier 0/50 for the WGAN and Hybrid rows was the soft branch's arithmetic rather than a measurement: the branch was `clip(0.10 * anomaly + 0.35 * ks, 0, 0.45)` against a 0.50 threshold, so its maximum was 0.45 and it could not flag any traffic for any input. It is now a trained traffic detector's bot probability against a threshold of 0.0181, calibrated on 457 held-out human bursts at a measured 4.60% false-positive rate. No diurnal adaptation is measurable. A separate diagnostic ([`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md)) establishes why the shipped WGAN-GP warden could not serve as a detector, and that the RL arm had been trained with a zero stealth penalty, which is what the retraining fixed. The interception rates are faithfully reported, but none of them is a statement about detectability in the field: the human side of the comparison is the repository's own synthetic traffic corpus, not recorded human browsing.

4. **Codec Capacity (Coded):** DSSC achieves 13.50–13.66 coded bits/carrier (mean 13.60, a 70.0% improvement over the 8-bit coded baseline) with 40.7–41.4% carrier reduction. Text-only mode reaches 15.43–15.97 coded bits/carrier with up to 49.9% reduction. These values include framing and RS parity; user-payload capacity is 7.00–12.90 bpc.

5. **Robustness:** Zero false acceptances across 356 wrong-key attempts, with the per-attempt bound set by the CRC-16 frame scope rather than 2⁻²⁵⁶. Substitution recovery is exact to the algebraic limit (61.2% at 5%, zero beyond the damage budget), while carrier dropout remains unrecoverable in the current format and now fails closed instead of accepting wrong plaintext. One 35,000-candidate permutation completes in 0.4056 ms.

6. **Scalability:** Direct-engine codec: 100% decode success at 50 worker threads with 0.000% observed marker-payload swaps. Throughput saturates near 1.9 sessions per second and p99 rises to 27.9 s; encoding grows sublinearly from 107.1 ms at 32 B to 206.3 ms at 512 B. The FastAPI transmit path admits concurrent sessions up to a bounded limit but is not covered by this benchmark.

### 10.2 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| RS-ECC substitution recovery collapses past the t = 4 damage budget | 61.2% recovery at 5% substitution, 0% from 8% | Raise parity along the adaptive ladder; documented as the algebraic limit |
| Carrier dropout is unrecoverable | One deletion damages ~32.7 codeword bytes and zero trials recover; the decoder now fails closed under ECC | Per-carrier sequence markers to convert deletions into known erasures (8-byte budget) |
| Legacy raw fallback accepted wrong plaintext under dropout | 41/320 pre-hardening trials at 2 through 5 percent deletion returned success with wrong plaintext | Fixed by requiring framed payloads when ECC is enabled; archived pre-hardening artifact |
| Decode latency 2–9x encode | Asymmetric pipeline | Parallelize decode, cache state spaces |
| Audio modality smaller corpus | 12-12.6 bits/carrier | Expand CLAP index with more datasets |
| RL agent high latency (15s mean) | Slow transmission | Hybrid GAN+RL scheduling |
| API transmit admits concurrent sessions up to a bound | A `/api/transmit` beyond `MAX_CONCURRENT_TRANSMISSIONS` returns HTTP 429; concurrent API throughput is not measured | A queueing redesign or a measured API concurrency campaign; the current benchmark is scoped to the direct engine |
| Packet JSON and manifest are plaintext | A filesystem observer can read channel, sequence, delay, and the message | Encrypt and authenticate packet bodies and manifests before treating the store as confidential |
| Component 5 latency is in-task direct-engine service time | It excludes queue wait, HTTP, scheduler, and transmit delay; p99 rests on 100 sessions | Add a client-observed API latency benchmark and a larger tail sample |

### 10.3 Future Work

1. **Erasure-aware framing** with per-carrier sequence markers for carrier dropout resilience
2. **Adversarial steganalysis training** with distillation from SRNet/EfficientNet
3. **Real-world traffic corpus** for Warden pretraining (WhatsApp, Telegram, Signal traces)
4. **Hardware acceleration** for VCP mapping (FAISS GPU, ONNX Runtime)
5. **Formal security proof** under Cachin's information-theoretic model
6. **Side-channel analysis** (cache timing, power, EM leakage)

---

## Appendix: Reproduction Commands

```bash
# Component 1: Media-Layer Steganalysis
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py

# Component 2: Semantic Distribution
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_2_semantic_distribution.py

# Component 3: Traffic Evasion
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py

# Component 4a: Capacity
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_4a_capacity.py

# Component 4b: Robustness
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_4b_robustness.py

# Component 4c: ECC Failure Anatomy
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py

# Component 5: Codec Concurrency
PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_5_system_concurrency.py

# Publication artifacts (tables, plots, JSON) are written directly by each
# benchmark script above; there is no separate artifact exporter.
```

---

*Report generated from empirical benchmarking data in storage/benchmarks/*. Results are reproducible with recorded seeds; environment provenance varies by component, and Component 5 carries a full provenance block.*

# DCASS Research Paper Write-Up Plan: Complete Benchmarking Suite
## Master Blueprint & Guide for Writing Section V (Experimental Evaluation & Security Analysis)

> **Target Venues:** IEEE Transactions on Information Forensics and Security (TIFS), ACM Conference on Computer and Communications Security (CCS), USENIX Security, IEEE TDSC  
> **Purpose:** This document provides the complete, step-by-step structural plan and technical reference guide for writing the experimental write-up of all 5 benchmark components. Use this guide to draft your final manuscript content, integrate LaTeX tables, embed figures, and address peer-reviewer critiques.

---

## 1. Global Document Organization & Manuscript Flow

When writing the experimental section of your research paper, follow this standard 5-part architecture corresponding to the 5 benchmark components:

```
Section V: EXPERIMENTAL EVALUATION AND SECURITY ANALYSIS
├── V-A. Media-Layer Content Steganalysis Resistance (Component 1)
│   ├── Threat Model: Passive Deep Residual Steganalysts
│   ├── Option B Empirical Protocol (Positive Controls vs Zero-Modification)
│   └── Verification: SRNet, ZhuNet, and YeNet Indistinguishability, measured AUC 0.497–0.499 with intervals covering 0.5000
│
├── V-B. Semantic Distribution Drift and Cross-Modal Alignment (Component 2)
│   ├── Threat Model: Multi-Modal Semantic / Statistical Warden
│   ├── Headline Calibrated Tests: Energy Distance & MMD²; KS Family and W₁ Projection as Secondary Screens
│   ├── Cluster Entropy, Repeat Structure, and the Modality Mixture Identity
│   └── Mechanism Consistency (Component 2b) and t-SNE Embedding Topology
│
├── V-C. Network Traffic Cadence and DPI Warden Evasion (Component 3)
│   ├── Threat Model: Active Deep Packet Inspection (DPI) & Flow Analyzers
│   ├── Generative vs Reactive Scheduling: WGAN, Hybrid Engine & PPO RL
│   └── Diurnal Conditioning Check (Hour 14:00 vs Hour 03:00)
│
├── V-D. Codec Capacity, Permutation Security and Channel Fault Tolerance (Component 4)
│   ├── Bit-Density Scaling & Voronoi Constellation Partitioning (13.60 bpc mean vs 8.00 exact_vcp)
│   ├── Transport Robustness: Reed-Solomon Recovery over Galois Field GF(2^8)
│   └── Cryptographic Avalanche & Observed FAR (0 of 356 wrong-key attempts)
│
└── V-E. Direct-Engine Computational Performance, Latency Breakdown & Concurrency Stress (Component 5)
    ├── Sub-Component Profiling (FAISS Search vs ECC vs Permutation Overhead)
    ├── Payload Scaling and Direct-Engine Speed (32B to 512B)
    └── Multi-Threaded Payload-Isolation Check and API Transmit Scope
```

---

## 2. Standard Writing Template for Each Component

For **every** component write-up, follow this 7-step formula:

1. **Adversary Threat Model:** Define what information the adversary observes and what capabilities they possess.
2. **Terminology & Metric Formulations:** Formally define every equation, symbol, and statistical test.
3. **Hypothesis & Expected Outcome:** State what the theory predicts (e.g., $\text{AUC} = 0.5000$, $D_{\mathrm{KS}} < 0.05$, $\text{FAR} = 0.00$).
4. **Empirical Evidence (Walkthrough of Data):** Walk the reader through the exact rows of the LaTeX table and curves of the figure.
5. **Physical & Statistical Causality ("Why it acts like this"):** Explain the underlying computer science mechanism behind the observed numbers.
6. **Comparison Against Baselines:** Explicitly compare DCASS against legacy methods (LSB, Poisson, Static metronome, pure VCP).
7. **Preempting Reviewer Questions:** Address anticipated skeptical questions from reviewers.

---

## 3. Component-by-Component Writing Blueprint

---

### Component 1: Media-Layer Content Steganalysis Blindness

#### A. Artifacts to Reference
* **LaTeX Table:** `storage/benchmarks/tables/table1_steganalysis.tex`
* **Plot / Curves:** `storage/benchmarks/plots/fig1_roc_curves.png` and `fig1_roc_curves.pdf`, plus `fig1_payload_sweep.png` and `fig1_payload_sweep.pdf`
* **Source Metrics:** `storage/benchmarks/component_1_summary.json`

#### B. Key Terms to Define in the Write-Up
* **Spatial Residual Noise:** High-frequency microscopic perturbations introduced into pixel matrices by embedding operations ($X' = X + \Delta$).
* **Spatial Rich Models (SRM):** 30 hand-crafted high-pass linear and non-linear filter kernels capturing directional pixel differences (1st through 4th order).
* **Truncated Linear Unit (TLU):** Activation function $\text{TLU}_T(x) = \min(\max(x, -T), T)$ that prevents large image edges from washing out weak steganographic signals.
* **Unpooled Residual Architecture:** Neural network design (like SRNet) where pooling/downsampling is banned in early layers to preserve high-frequency noise.
* **Option B Protocol:** Two-stage evaluation methodology where the detector must first pass a 0.90 accuracy gate on a known positive control, per fold, before evaluating covert carriers.
* **ROC AUC (Receiver Operating Characteristic Area Under Curve):** Metric of separability. $1.0$ is perfect detection; $0.5000$ is complete random guessing.
* **Kullback-Leibler Divergence ($D_{\mathrm{KL}}$):** Relative entropy between the classifier's output probability distributions on stego vs cover; in this benchmark it is a permutation-calibrated plug-in statistic, so quote it with its $p$ value.

#### C. Exact Numerical Data to Quote
* **SRNet (Residual Gold Standard):** Positive control accuracy $93.67\%$ [92.88, 94.39], AUC $0.9844$. DCASS carrier cover rate $88.31\%$, balanced accuracy $0.5025$ [0.4938, 0.5112], **measured AUC $= 0.4973$** [0.4830, 0.5143], $D_{\mathrm{KL}} = 0.23\text{ bits}$ with permutation $p$ 0.238–0.423.
* **ZhuNet (Separable Convolutions + SPP):** Positive control accuracy $95.05\%$ [94.33, 95.68], AUC $0.9860$. DCASS carrier cover rate $92.14\%$, balanced accuracy $0.5025$ [0.4956, 0.5101], **measured AUC $= 0.4993$** [0.4820, 0.5146], $D_{\mathrm{KL}} = 0.18\text{ bits}$ with permutation $p$ 0.250–0.937.
* **YeNet (SRM Filter Bank + TLU):** Positive control accuracy $93.97\%$ [93.19, 94.67], AUC $0.9853$. DCASS carrier cover rate $92.34\%$, balanced accuracy $0.4988$ [0.4914, 0.5069], **measured AUC $= 0.4993$** [0.4826, 0.5160], $D_{\mathrm{KL}} = 0.23\text{ bits}$ with permutation $p$ 0.385–0.849.
* **Disclosure foils:** SRM plus FLDA and XuNet fail the 0.90 gate in all folds and are excluded from the official result. Disclose them in a footnote on record, including the fresh SRM DCASS AUC 0.5059 [0.4883, 0.5217] with per-fold permutation $p$ 0.816, 0.154, and 0.987, the earlier 0.5548 reading that did not reproduce, and the gate rule: only detectors passing the control gate testify about carriers.

#### D. "Why It Acts Like This" (Physical Explanation)
* *Why traditional stego gets caught (93–95 percent control accuracy):* LSB modifications introduce high-frequency parity noise that violates local natural image dependencies, lighting up the SRM filters and unpooled residual blocks.
* *Why DCASS collapses the measured AUC to 0.50:* DCASS utilizes **selection-based coverless steganography** — it **never modifies a single pixel, bit, or DCT coefficient**. The carrier is drawn untouched from public corpora. Because the physical media matrix is identical to a natural cover, there are zero physical residuals, removing the content-residual signal these detectors were trained to find. This does not cover selection patterns or traffic timing.

#### E. Potential Reviewer Questions to Address
* *Reviewer critique:* "Did your detectors fail to detect DCASS because they were undertrained?"
* *Your response in paper:* No. Option B explicitly verified that the three models passed the per-fold 0.90 gate on $\pm 1$ LSB positive controls at 93.7–95.1% accuracy and 0.984–0.986 AUC. The blindness on DCASS is caused strictly by the absence of media modification.

---

### Component 2: Semantic Selection Bias & Distributional Drift

#### A. Artifacts to Reference
* **LaTeX Tables:** `storage/benchmarks/tables/table2_distributional_drift.tex`, `storage/benchmarks/tables/table2b_cluster_uniformity.tex`
* **Plots & Projections:** `storage/benchmarks/plots/fig2_distributional_stats.png`, `fig_ks_heatmap.png`, `fig_manifold_tsne.png`, `fig_2b_cluster_usage.png`
* **Source Metrics:** `storage/benchmarks/component_2_results.json`, `storage/benchmarks/component_2b_cluster_usage_results.json`

#### B. Key Terms to Define in the Write-Up
* **Selection Bias / Distributional Drift:** The statistical distortion created if a covert sender over-selects certain categories or semantic concepts from a corpus.
* **Voronoi Constellation Partitioning (VCP):** Dividing a high-dimensional continuous embedding space into $K=256$ discrete topological cells centered at spherical $k$-means centroids.
* **Energy Distance:** The weighted difference between cross-sample and within-sample mean distances, zero exactly when the two distributions coincide; reported in unbiased U-statistic form with permutation calibration.
* **Squared Maximum Mean Discrepancy ($\mathrm{MMD}^2$):** The distance between RBF kernel mean embeddings with a median-heuristic bandwidth; unbiased estimator, calibrated by the same permutation test as the energy distance.
* **Two-Sample Kolmogorov-Smirnov Family ($D_{\mathrm{KS}}$):** Non-parametric per-dimension screen over 512 embedding coordinates under Benjamini-Hochberg control, with a family-wise permutation test of the maximum ECDF gap.
* **Wasserstein-1 Distance ($W_1$ / Earth Mover's Distance):** The minimum work required to morph the probability density of transmitted carriers into that of the natural corpus:
  $$W_1(P_{\text{carrier}}, P_{\text{corpus}}) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x,y)\sim \gamma}[\|x - y\|]$$
  reported as a one-dimensional PC1 projection with its variance share, not as a multivariate transport cost.
* **Cluster Shannon Entropy ($H(C)$):** Measure of centroid usage uniformity, recorded on carrier slots and on unique media. Theoretical maximum for 256 centroids is $\log_2(256) = 8.000\text{ bits}$.
* **t-SNE Manifold Alignment:** 2D non-linear dimensionality reduction showing interleaving between disjoint reference points and covert carrier points.

#### C. Exact Numerical Data to Quote
* **Headline multivariate tests:** Image energy $0.00011$ ($p = 0.009$), $\mathrm{MMD}^2 = 6.10\times10^{-5}$ ($p = 0.012$); Text energy $0.00010$ ($p = 0.004$), $\mathrm{MMD}^2 = 8.36\times10^{-5}$ ($p = 0.004$); Audio energy $-0.00003$ ($p = 0.574$), $\mathrm{MMD}^2 = -3.28\times10^{-5}$ ($p = 0.642$). Unique-media energy: $0.00007$ ($p = 0.035$), $0.00016$ ($p = 0.001$), $-0.00004$ ($p = 0.683$).
* **Wasserstein-1 on PC1:** Image $0.00810$, Text $0.00612$, Audio $0.00278$, with PC1 variance shares $5.74\%$, $6.20\%$, and $27.03\%$.
* **Kolmogorov-Smirnov family:** Zero BH discoveries at $q = 0.05$ in all three modalities, 0 of 512 each; family-wise permutation $p = 0.315$, $0.148$, $0.092$; the mean $D_{\mathrm{KS}}$ lands on the finite-sample null centers of $0.0152$ (image), $0.0152$ (text), and $0.0153$ (audio) — derived from the asymptotic mean constant $\sqrt{\pi/2}\ln 2 \approx 0.8687$ over $\sqrt{(n+m)/(nm)}$ with the per-modality carrier counts ($9{,}250/9{,}500/9{,}100$ against $5{,}000$ reference vectors), not a stored null.
* **Voronoi Entropy:** $H(C) = 7.8090\text{ bits}$ on 27,850 slots ($97.6\%$) and $7.7814\text{ bits}$ on 23,890 unique media; the modality mixture identity closes the deficit exactly, $6.2242 + 1.5847 = 7.8090$.
* **Component 2b mechanism test:** 10,800 carriers over 200 messages; chi-square vs uniformity 3,253.4 with permutation $p = 0.0005$, vs mechanism 268.6 on 253 degrees of freedom with permutation $p = 0.208$; largest standardized deviation $+2.62$ against a permutation ceiling of $4.08$; entropy 7.7935 bits on slots and 7.7873 on unique media.

#### D. "Why It Acts Like This" (Statistical Explanation)
* *Why naive coverless systems fail:* Early coverless steganography mapped words to images (e.g., word "apple" $\to$ image of an apple). An adversary monitoring traffic detects an unnatural spike in "apple" photos.
* *Why DCASS succeeds:* DSSC applies a session-keyed HMAC-DRBG permutation across the payload before Voronoi centroid assignment, so message-dependent structure does not survive into carrier choice. The measurable remainder is the pipeline's own public schedule: two semantic families per modality under deterministic round robin, which produces the small predicted deviations and the structural entropy deficit. A dedicated 200-message mechanism test shows the transmitted histogram is statistically consistent with that schedule while decisively rejecting uniformity.

---

### Component 3: Network Traffic Timing & DPI Warden Evasion

#### A. Artifacts to Reference
* **LaTeX Tables:** `storage/benchmarks/tables/table3_traffic_evasion.tex`, `storage/benchmarks/tables/table3b_warden_score_quantiles.tex`
* **Plots:** `storage/benchmarks/plots/fig_interarrival_cdf.png`, `fig_interception_rates.png`, `fig_circadian_diurnal.png`, `fig_3b_warden_scores.png`
* **Source Metrics:** `storage/benchmarks/component_3_results.json`, `storage/benchmarks/component_3b_warden_score_summary.json`; the conservative PPO operating point is sourced from `storage/benchmarks/component_3_results_baseline.json`

#### B. Key Terms to Define in the Write-Up
* **Deep Packet Inspection (DPI) Flow Warden:** Network intermediary analyzing packet timestamps to detect non-human automated traffic.
* **Sliding-Window Burst Detection:** Warden rule flagging sequences where $\ge 6\text{ packets}$ arrive within a sliding window of $5.0\text{s}$.
* **Inter-Arrival Time ($\Delta t$):** Time elapsed between consecutive packet transmissions.
* **Timing Entropy ($H_T$):** Shannon entropy of quantized inter-arrival delays in bits.
* **Closed-Loop Channel Masking:** Dynamically tracking per-channel cooldown state ($t - t_{\mathrm{last}}[c] \ge T_c$) to avoid platform rate-limit HTTP 429 throttling.
* **Circadian Diurnal Adaptation:** Conditioning a generative temporal model on time-of-day ($t \in [0, 24]$) to match biological human active/sleep cycles; the shipped checkpoint shows no measurable adaptation (see `BENCHMARK_DEEP_EXPLANATION.md` section 3.4).
* **Observed Delay Floor:** The generator's softplus head plus the canonical mapping $(\text{raw}-0.5)\times 1500+1.5$ place generated delays just above $1.5\text{s}$; no scheduler-level refractory rule exists in the recorded code.

#### C. Exact Numerical Data to Quote
* **Static Cadence:** Interception rate **$100.0\%$**, not-flagged rate **$0.0\%$**, Delay entropy **$0.000\text{ bits}$**, Mean delay $3.00\text{s}$.
* **Stochastic Poisson:** Interception rate **$54.0\%$**, not-flagged rate **$46.0\%$**, Delay entropy $4.234\text{ bits}$, Mean delay $3.04\text{s}$.
* **DCASS Temporal WGAN:** Interception rate **$100.0\%$**, not-flagged rate **$0.0\%$**, Delay entropy $3.381\text{ bits}$, Mean delay $3.93\text{s}$, Triggering rule: soft $50/50$.
* **DCASS Hybrid Engine:** Interception rate **$100.0\%$**, not-flagged rate **$0.0\%$**, Delay entropy $3.441\text{ bits}$, Mean delay $3.89\text{s}$, Triggering rule: soft $50/50$. The soft branch is a trained traffic detector's bot probability, and the Hybrid traffic scores $0.994$ to $0.995$ against a flag threshold of $0.0181$ that was calibrated on held-out human traffic at a measured $4.60\%$ false-positive rate. The earlier $0.0\%$ for this row was the soft branch's arithmetic and is withdrawn.
* **DCASS PPO RL Agent (retrained):** Interception rate $0.0\%$, not-flagged rate $100.0\%$, Mean delay $5.20\text{s}$. The soft branch scores it $0.0040$ to $0.0047$ against a threshold of $0.0181$, and it breaches the per-channel cooldowns 15 times across 50 trials. The measurement uses a detector held out from RL training, so the row is not the agent gamming its own opponent. The agent it replaces was flagged at $100\%$, because it was trained against a warden whose bot probability was identically zero. The scope limit travels with the row: the human side is the repository's own synthetic corpus, and the detector is a narrow timing model.
* **Diurnal Conditioning Check (identical mapping):**
  * Hour 14:00: Mean delay **$4.03\text{s}$** ($\sigma = 2.62\text{s}$), interception **$100.0\%$.
  * Hour 03:00: Mean delay **$3.70\text{s}$** ($\sigma = 2.81\text{s}$), interception **$100.0\%$.
  * No measurable adaptation in the shipped checkpoint; the earlier $3.19\text{s}$ / $13.07\text{s}$ split was an artifact of per-regime analysis constants and is withdrawn.

#### D. "Why It Acts Like This" (Behavioral & Statistical Deep-Dive)
* *Why Static fails ($100\%$ interception):* A fixed $3.0\text{s}$ metronome records coefficient of variation $0.0000$ and log-bin entropy $0.0\text{ bits}$; the periodic rule fires in all 50 trials.
* *Why Poisson fails ($54\%$ interception):* Exponential delays defeat the periodic rules, but 27 of 50 launches place six or more packets inside some sliding $5.0\text{s}$ window, and the burst rule scores those $0.71$ to $0.89$.
* *Why the DCASS CDF has no mass below $1.5\text{s}$:* The generator's softplus head plus the canonical mapping $(\text{raw}-0.5)\times 1500 + 1.5$ place every generated delay just above $1.5\text{s}$ (measured minimum near $1.6\text{s}$, raw output band about $0.2\%$ of the unit range). No scheduler-level refractory rule exists in the recorded code; the surrogate's $28.1\%$ mass below $1.5\text{s}$ is unmatched.
* *Why the human curve plateaus from $30\text{s}$ to $3{,}000\text{s}$:* The surrogate human reference mixes $80\%$ short exponential delays and $20\%$ log-normal tail mass, so its cumulative distribution thins gradually along the tail. DCASS completes a 50-carrier message within $3.27\text{ minutes}$, so it cannot reproduce the long tail.

---

### Component 4: Codec Capacity, Permutation & Channel Fault Tolerance

#### A. Artifacts to Reference
* **LaTeX Tables:** `storage/benchmarks/tables/table4a_capacity.tex`, `table4b_codec_robustness.tex`, `table4c_ecc_failure_modes.tex`
* **Plots:** `storage/benchmarks/plots/fig_4a_capacity_comparison.png`, `fig_4b_permutation_scaling.png`, `fig_4b_ecc_recovery_curve.png`, `fig_4b_key_separation.png`, `fig_4c_ecc_failure_modes.png`
* **Source Metrics:** `storage/benchmarks/component_4a_capacity_results.json`, `component_4b_robustness_results.json`, `component_4c_ecc_failure_results.json`

#### B. Key Terms to Define in the Write-Up
* **Bits Per Carrier (BPC):** exact_vcp is fixed at $\log_2 256 = 8.00$ bits per carrier; DSSC reads a symbol per carrier from a session-private candidate pool with per-chunk width $\lfloor \log_2 |\mathcal{C}_i| \rfloor$ clamped to 2 through 16, recording a mean of 13.60.
* **Reed-Solomon Error Correction (RS-ECC):** Block code operating over Galois Field $\mathrm{GF}(2^8)$ appending $k$ parity bytes to payload blocks, correctable up to $t = \lfloor k/2 \rfloor$ byte errors at unknown positions or $k$ known-position erasures.
* **Cryptographic Avalanche Effect:** Property where flipping 1 bit in a key changes approximately $50\%$ of the output permutation.
* **Pearson Correlation ($\rho$):** Linear dependence between two permutation index arrays, computed with `np.corrcoef`; $\rho \approx 0.00$ implies no linear relationship between permutations under different keys.
* **False Acceptance Rate (FAR):** Probability that an unauthorized or mutated key decrypts valid, authenticated plaintext.

#### C. Exact Numerical Data to Quote
* **Bit Density:** **$13.60\text{ bpc}$** mean across the five multi-modal payload sizes (13.50 to 13.66), **$15.43$ to $15.97\text{ bpc}$** text-only, against **$8.00\text{ bpc}$** for exact_vcp.
* **Permutation Scaling:** $0.0575\text{ ms}$ at $N=1,000$ candidates, scaling to $0.4056\text{ ms}$ at $N=35,000$ candidates, about $616\times$ faster than the legacy pure-Python loop.
* **Substitution Recovery:** $100.0\%$ perfect reconstruction at $0\%$ substitution; $70.0\%$ recovery at $5\%$ substitution with $2.7\text{ bytes}$ fixed; the 80-trial extension refines the 5 percent point to $61.2\%$ $[50.3, 71.2]$ and shows recovery exactly at the $t=4$ damage budget in all 640 substitution trials.
* **Dropout Recovery:** $0.0\%$ at every recorded rate from $2\%$; one deleted carrier damages about $32.7$ codeword bytes because positional enumeration shifts every later carrier.
* **Algebraic Layer:** byte errors decode at $4$ injected bytes and fail at $5$; known erasures decode at $8$ and fail at $9$, the exact thresholds of $RS(63,55)$.
* **Erasure-Bound Replay:** known positions would keep recovery at $100\%$ through $6\%$ substitution, $98.8\%$ at $8\%$, falling to $51.2\%$ at $10\%$.
* **Cryptographic Key Separation:**
  * 100 independent random wrong keys: **$0/100$ accepted ($\text{FAR} = 0.000000$)**.
  * Exhaustive 256 single-bit flipped keys: **$0/256$ accepted ($\text{FAR} = 0.000000$)**.
  * Mean key sensitivity $\rho = -0.00198 \approx 0.00$; the per-attempt acceptance bound is the CRC-16 frame scope of about $2^{-16}$ on the recorded path.

#### D. "Why It Acts Like This" (Cryptographic & Algebraic Deep-Dive)
* *Why capacity exceeds 8 bits per carrier:* exact_vcp is capped at $\log_2 256 = 8.00$ because each carrier indexes one public centroid. DSSC reads each symbol from a session-private candidate pool, so the per-chunk width is $\lfloor \log_2 |\mathcal{C}_i| \rfloor$ clamped to 16 bits and the recorded mean is 13.60.
* *Why FAR is strictly 0.000000:* Inverting the centroid sequence with an incorrect key derives different families and a different permutation, so the symbol stream is scrambled. The Reed-Solomon layer reports uncorrectable damage without raising an exception, and the CRC-16 frame rejects whatever survives; the per-attempt bound is about $2^{-16}$, raised to $2^{-256}$ by the optional HMAC frame.

---

### Component 5: Direct-Engine Codec Latency, Speed & Concurrency Stress

#### A. Artifacts to Reference
* **LaTeX Table:** `storage/benchmarks/tables/table5_system_performance.tex`
* **Plots:** `storage/benchmarks/plots/fig5_latency_breakdown.png`, `fig5_concurrency_scaling.png`
* **Source Metrics:** `storage/benchmarks/component_5_results.json`

#### B. Key Terms to Define in the Write-Up
* **Direct-Engine Encoding Latency ($T_{\mathrm{enc}}$):** Wall-clock time for `SemanticEngine.encode` to convert a plaintext string into carrier IDs.
* **Direct-Engine Decoding Latency ($T_{\mathrm{dec}}$):** Wall-clock time for `SemanticEngine.decode` to map carrier IDs back to centroids and reconstruct plaintext.
* **Latency Percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$):** In-task encode-plus-decode service time under worker-thread load; queue wait, HTTP, and transmit delays are excluded.
* **Observed Swap Rate:** Percentage of sessions where a decoded payload differs from the session's own secret and carries the shared marker. This is not an API session-isolation test.
* **Sessions Per Second:** Completed direct-engine sessions, where each session is one encode and one decode.

#### C. Exact Numerical Data to Quote
* **Payload Scaling:**
  * $32\text{B}$ payload ($27$ carriers): Encode **$107.1\text{ ms}$**, Decode **$222.6\text{ ms}$**, Throughput $0.29\text{ KB/s}$, Carrier speed $81.9\text{ carriers/s}$.
  * $64\text{B}$ payload ($46$ carriers): Encode **$111.0\text{ ms}$**, Decode **$321.7\text{ ms}$**, Throughput $0.56\text{ KB/s}$, Carrier speed $106.3\text{ carriers/s}$.
  * $128\text{B}$ payload ($83$ carriers): Encode **$123.2\text{ ms}$**, Decode **$484.7\text{ ms}$**, Throughput $1.01\text{ KB/s}$, Carrier speed $136.5\text{ carriers/s}$.
  * $256\text{B}$ payload ($163$ carriers): Encode **$158.5\text{ ms}$**, Decode **$856.6\text{ ms}$**, Throughput $1.58\text{ KB/s}$, Carrier speed $160.6\text{ carriers/s}$.
  * $512\text{B}$ payload ($317$ carriers): Encode **$206.3\text{ ms}$**, Decode **$1590.0\text{ ms}$**, Throughput $2.42\text{ KB/s}$, Carrier speed $176.5\text{ carriers/s}$.
* **Concurrency Stress Testing (100 sessions per level, direct engine):**
  * 10 Threads: Throughput **$2.00\text{ sessions/s}$**, $p_{50} = 4,976\text{ ms}$, $p_{90} = 5,502\text{ ms}$, $p_{95} = 5,706\text{ ms}$, $p_{99} = 5,979\text{ ms}$, Success **$100.0\%$**, Observed Swaps **$0.000\%\%$.
  * 20 Threads: Throughput **$1.95\text{ sessions/s}$**, $p_{50} = 10,089\text{ ms}$, $p_{90} = 11,220\text{ ms}$, $p_{95} = 11,455\text{ ms}$, $p_{99} = 11,730\text{ ms}$, Success **$100.0\%$**, Observed Swaps **$0.000\%\%$.
  * 50 Threads: Throughput **$1.89\text{ sessions/s}$**, $p_{50} = 24,367\text{ ms}$, $p_{90} = 26,989\text{ ms}$, $p_{95} = 27,598\text{ ms}$, $p_{99} = 27,949\text{ ms}$, Success **$100.0\%$**, Observed Swaps **$0.000\%\%$.

#### D. "Why It Acts Like This" (Systems Deep-Dive)
* *Why encoding stays under 210 ms:* Sub-stage profiling puts HMAC-DRBG permutation at $0.067\text{ ms}$ and RS-ECC framing at $0.022\text{ ms}$. The FAISS vector search over $245,279$ index items is the dominant residual stage, at $107.1\text{ ms}$ for 32 B and $193.3\text{ ms}$ for 512 B. The shipped indices are exact CPU `IndexFlatIP` indexes. The stage rows are computed by subtraction, not direct FAISS timing.
* *Why observed swaps are 0.000% in the direct-engine test:* Each direct-engine session uses a distinct 32-byte key and a distinct keyed permutation of the candidate space. The check flags a different payload that carries the shared marker. The FastAPI transmit path is not exercised by this benchmark.

---

## 4. Master LaTeX Paper Integration Guide

When assembling your paper LaTeX source file (`main.tex`), include the tables and figures in this order:

```latex
% ----------------------------------------------------------------------
% Section V: Experimental Evaluation
% ----------------------------------------------------------------------
\section{Experimental Evaluation and Security Analysis}
\label{sec:experiments}

% --- V-A: Media Steganalysis ---
\subsection{Media-Layer Steganalysis Blindness}
\input{storage/benchmarks/tables/table1_steganalysis.tex}
% Embed Figure 1:
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{storage/benchmarks/plots/fig1_roc_curves.pdf}
  \caption{Steganalysis detection accuracy and ROC AUC evaluation across SRNet, ZhuNet, and YeNet.}
  \label{fig:roc_curves}
\end{figure}

% --- V-B: Semantic Distribution Drift ---
\subsection{Semantic Distribution Invariance}
\input{storage/benchmarks/tables/table2_distributional_drift.tex}
% Embed Figures for Component 2:
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig2_distributional_stats.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_ks_heatmap.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_manifold_tsne.png}
  \caption{Cross-modal distributional drift evaluation: headline statistics with per-draw dots, KS distance heatmap, and 2D t-SNE manifold alignment.}
  \label{fig:distributional_drift}
\end{figure*}

% --- V-C: Traffic Timing & DPI Evasion ---
\subsection{Network Traffic Cadence and DPI Warden Evasion}
\input{storage/benchmarks/tables/table3_traffic_evasion.tex}
% Embed Figures for Component 3:
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_interarrival_cdf.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_interception_rates.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_circadian_diurnal.png}
  \caption{Traffic timing analysis: Inter-arrival CDFs vs human baseline, DPI interception rates, and the diurnal conditioning check under one identical mapping.}
  \label{fig:traffic_evasion}
\end{figure*}

% --- V-D: Codec Capacity & Channel Faults ---
\subsection{Codec Capacity, Fault Tolerance, and Cryptographic Isolation}
\input{storage/benchmarks/tables/table4a_capacity.tex}
\input{storage/benchmarks/tables/table4b_codec_robustness.tex}
\input{storage/benchmarks/tables/table4c_ecc_failure_modes.tex}
% Embed Figures for Component 4:
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4a_capacity_comparison.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_permutation_scaling.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_key_separation.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4c_ecc_failure_modes.png}
  \caption{Codec performance and failure anatomy: Bit-density scaling, permutation latency, substitution recovery, and zero-FAR key isolation.}
  \label{fig:codec_performance}
\end{figure*}

% --- V-E: Computational Performance ---
\subsection{Computational Performance and System Scalability}
\input{storage/benchmarks/tables/table5_system_performance.tex}
% Embed Figures for Component 5:
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.49\linewidth]{storage/benchmarks/plots/fig5_latency_breakdown.png}
  \includegraphics[width=0.49\linewidth]{storage/benchmarks/plots/fig5_concurrency_scaling.png}
  \caption{Computational overhead: Granular latency breakdown and throughput scaling under concurrent client workloads.}
  \label{fig:system_performance}
\end{figure}
```

---

## 5. Reviewer FAQ & Defense Matrix

When drafting your paper, preempt these four common reviewer questions:

| Reviewer Objection | Scientific Defense & Benchmark Proof to Cite |
| :--- | :--- |
| **"Isn't coverless steganography vulnerable to semantic selection bias?"** | Cite **Component 2**: headline energy distance and MMD² for image and text at permutation $p = 0.009$ and $p = 0.004$, nothing for audio; zero of 1,536 KS coordinates reject under Benjamini-Hochberg control; combined entropy $H(C) = 7.8090\text{ bits}$ on slots and $7.7814$ on unique media, with the deficit closed by the modality mixture identity. Add **Component 2b**: the transmitted histogram is consistent with the pipeline's public mechanism, chi-square 268.6 on 253 degrees of freedom with permutation $p = 0.21$. |
| **"Doesn't automated network transmission trigger DPI flow alarms?"** | Cite **Component 3**: the recorded composite warden flags the DCASS Hybrid Engine at **$100\%$**, as it does the WGAN arm, while static cadence is flagged at $100\%$ and Poisson at $54\%$. The retrained PPO arm is not flagged. The soft branch is a trained traffic detector whose threshold is calibrated on held-out human traffic at a measured $4.60\%$ false-positive rate, so these are measurements rather than artifacts. Disclose that the human side of the comparison is the repository's own synthetic traffic corpus, not recorded human browsing, and that no diurnal adaptation is measurable ($4.03\text{s}$ at hour 14 vs $3.70\text{s}$ at hour 3 under one identical mapping). |
| **"What if the channel experiences packet loss or an attacker attempts key forgery?"** | Cite **Component 4**: Galois Field $\mathrm{GF}(2^8)$ RS-ECC corrects up to $t=4$ damaged bytes per $RS(63,55)$ block and recovers $70.0\%$ of messages at $5\%$ carrier substitution, refined to $61.2\%$ $[50.3, 71.2]$ by the 80-trial extension, with an algebraic cliff beyond that budget; carrier dropout is disclosed as unrecoverable in the current format. 256-bit key sensitivity yields $\rho \approx 0.00$ and **$\text{FAR} = 0.000000$** (all 356 wrong-key attempts rejected, per-attempt bound the CRC-16 frame scope). |
| **"Is vector search over hundreds of thousands of media items too slow for real-time use?"** | Cite **Component 5**: Direct-engine encoding is **$107.1\text{ ms}$** at 32B and **$206.3\text{ ms}$** at 512B; carrier speed is **$81.9\text{ carriers/s}$** at 32B and **$176.5\text{ carriers/s}$** at 512B; the 50-worker direct-engine run records **$0.000\%$ observed marker-payload swaps**. The FastAPI transmit path is not covered by this benchmark. |

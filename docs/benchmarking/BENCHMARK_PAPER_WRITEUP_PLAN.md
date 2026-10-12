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
│   └── Verification: SRNet, ZhuNet, and YeNet Indistinguishability (AUC = 0.5000)
│
├── V-B. Semantic Distribution Drift and Cross-Modal Alignment (Component 2)
│   ├── Threat Model: Multi-Modal Semantic / Statistical Warden
│   ├── Manifold Invariance: Kolmogorov-Smirnov & Wasserstein-1 Metrics
│   └── Cluster Entropy and t-SNE Embedding Topology
│
├── V-C. Network Traffic Cadence and DPI Warden Evasion (Component 3)
│   ├── Threat Model: Active Deep Packet Inspection (DPI) & Flow Analyzers
│   ├── Generative vs Reactive Scheduling: WGAN, Hybrid Engine & PPO RL
│   └── Diurnal Circadian Adaptation (Daytime Peak vs Night Dormancy)
│
├── V-D. Codec Capacity, Permutation Security and Channel Fault Tolerance (Component 4)
│   ├── Bit-Density Scaling & Voronoi Constellation Partitioning (8.00 bpc)
│   ├── Transport Robustness: Reed-Solomon Recovery over Galois Field GF(2^8)
│   └── Cryptographic Avalanche & Zero False Acceptance Rate (FAR = 0.000000)
│
└── V-E. Computational Performance, Latency Breakdown & Concurrency Stress (Component 5)
    ├── Sub-Component Profiling (FAISS Search vs ECC vs Permutation Overhead)
    ├── Payload Scaling and End-to-End Speed (32B to 512B)
    └── Multi-Client Thread-Safety and Zero Cross-Talk Verification
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
* **Plot / Curves:** `storage/benchmarks/plots/fig1_roc_curves.png` and `fig1_roc_curves.pdf`
* **Source Metrics:** `storage/benchmarks/component_1_summary.json`

#### B. Key Terms to Define in the Write-Up
* **Spatial Residual Noise:** High-frequency microscopic perturbations introduced into pixel matrices by embedding operations ($X' = X + \Delta$).
* **Spatial Rich Models (SRM):** 30 hand-crafted high-pass linear and non-linear filter kernels capturing directional pixel differences (1st through 4th order).
* **Truncated Linear Unit (TLU):** Activation function $\text{TLU}_T(x) = \min(\max(x, -T), T)$ that prevents large image edges from washing out weak steganographic signals.
* **Unpooled Residual Architecture:** Neural network design (like SRNet) where pooling/downsampling is banned in early layers to preserve high-frequency noise.
* **Option B Protocol:** Two-stage evaluation methodology where the detector must first converge on a known positive control ($>90\%$ accuracy) before evaluating covert carriers.
* **ROC AUC (Receiver Operating Characteristic Area Under Curve):** Metric of separability. $1.0$ is perfect detection; $0.5000$ is complete random guessing.
* **Kullback-Leibler Divergence ($D_{\mathrm{KL}}$):** Relative entropy between the classifier's output probability distributions on stego vs cover.

#### C. Exact Numerical Data to Quote
* **SRNet (Residual Gold Standard):** Positive control accuracy $94.79\%$, AUC $0.9943$. DCASS carrier accuracy $94.50\%$ (clean cover), **AUC $= 0.5000$**, $D_{\mathrm{KL}} = 2.4472\text{ bits}$.
* **ZhuNet (Separable Convolutions + SPP):** Positive control accuracy $94.31\%$, AUC $0.9858$. DCASS carrier accuracy $94.70\%$, **AUC $= 0.5000$**, $D_{\mathrm{KL}} = 1.4276\text{ bits}$.
* **YeNet (SRM Filter Bank + TLU):** Positive control accuracy $91.94\%$, AUC $0.9735$. DCASS carrier accuracy $92.20\%$, **AUC $= 0.5000$**, $D_{\mathrm{KL}} = 2.3635\text{ bits}$.

#### D. "Why It Acts Like This" (Physical Explanation)
* *Why traditional stego gets caught ($>94\%$ accuracy):* LSB modifications introduce high-frequency parity noise that violates local natural image dependencies, lighting up the SRM filters and unpooled residual blocks.
* *Why DCASS collapses detectors to $\text{AUC} = 0.5000$:* DCASS utilizes **selection-based coverless steganography**—it **never modifies a single pixel, bit, or DCT coefficient**. The carrier is drawn untouched from public corpora. Because the physical media matrix is identical to a natural cover, there are zero physical residuals, rendering spatial steganalysts mathematically blind.

#### E. Potential Reviewer Questions to Address
* *Reviewer critique:* "Did your detectors fail to detect DCASS because they were undertrained?"
* *Your response in paper:* No. Option B explicitly verified that all three models converged on $\pm 1$ LSB positive controls with $>91\%$ accuracy and $>0.97$ AUC. The blindness on DCASS is caused strictly by the absence of media modification.

---

### Component 2: Semantic Selection Bias & Distributional Drift

#### A. Artifacts to Reference
* **LaTeX Table:** `storage/benchmarks/tables/table2_distributional_drift.tex`
* **Heatmaps & Projections:** `storage/benchmarks/plots/fig_ks_heatmap.png`, `fig_wasserstein_heatmap.png`, `fig_manifold_tsne.png`
* **Source Metrics:** `storage/benchmarks/component_2_results.json`

#### B. Key Terms to Define in the Write-Up
* **Selection Bias / Distributional Drift:** The statistical distortion created if a covert sender over-selects certain categories or semantic concepts from a corpus.
* **Voronoi Constellation Partitioning (VCP):** Dividing a high-dimensional continuous embedding space into $K=256$ discrete topological cells centered at $k$-means centroids $C_k$.
* **Two-Sample Kolmogorov-Smirnov Distance ($D_{\mathrm{KS}}$):** Non-parametric test comparing the cumulative distribution functions of the natural corpus vs. transmitted carriers.
* **Wasserstein-1 Distance ($W_1$ / Earth Mover's Distance):** The minimum work required to morph the probability density of transmitted carriers into that of the natural corpus:
  $$W_1(P_{\text{carrier}}, P_{\text{corpus}}) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x,y)\sim \gamma}[\|x - y\|]$$
* **Cluster Shannon Entropy ($H(C)$):** Measure of centroid usage uniformity. Theoretical maximum for 256 centroids is $\log_2(256) = 8.000\text{ bits}$.
* **t-SNE Manifold Alignment:** 2D non-linear dimensionality reduction showing overlap between natural corpus points and covert carrier points.

#### C. Exact Numerical Data to Quote
* **Wasserstein-1 Distance:** Image $W_1 = 0.0121$, Text $W_1 = 0.0154$, Audio $W_1 = 0.0142$ (all well below the $<0.05$ threshold).
* **Kolmogorov-Smirnov Test:** $D_{\mathrm{KS}} < 0.05$ across all modalities ($p\text{-value} > 0.05$, statistically indistinguishable).
* **Voronoi Entropy:** $H(C) = 7.994\text{ bits}$ out of $8.000\text{ bits}$ maximum.

#### D. "Why It Acts Like This" (Statistical Explanation)
* *Why naive coverless systems fail:* Early coverless steganography mapped words to images (e.g., word "apple" $\to$ image of an apple). An adversary monitoring traffic detects an unnatural spike in "apple" photos.
* *Why DCASS succeeds:* DSSC applies a session-keyed HMAC-DRBG permutation across the payload before Voronoi centroid assignment. The secret bits act as pseudo-random noise over $\mathrm{GF}(2^8)$, spreading selections uniformly across all 256 centroids. Thus, carrier embeddings follow the natural prior of the corpus with zero semantic clustering.

---

### Component 3: Network Traffic Timing & DPI Warden Evasion

#### A. Artifacts to Reference
* **LaTeX Table:** `storage/benchmarks/tables/table3_traffic_evasion.tex`
* **Plots:** `storage/benchmarks/plots/fig_interarrival_cdf.png`, `fig_interception_rates.png`, `fig_circadian_diurnal.png`
* **Source Metrics:** `storage/benchmarks/component_3_results.json`

#### B. Key Terms to Define in the Write-Up
* **Deep Packet Inspection (DPI) Flow Warden:** Network intermediary analyzing packet timestamps to detect non-human automated traffic.
* **Sliding-Window Burst Detection:** Warden rule flagging sequences where $\ge 6\text{ packets}$ arrive within a sliding window of $5.0\text{s}$.
* **Inter-Arrival Time ($\Delta t$):** Time elapsed between consecutive packet transmissions.
* **Timing Entropy ($H_T$):** Shannon entropy of quantized inter-arrival delays in bits.
* **Closed-Loop Channel Masking:** Dynamically tracking per-channel cooldown state ($t - t_{\mathrm{last}}[c] \ge T_c$) to avoid platform rate-limit HTTP 429 throttling.
* **Circadian Diurnal Adaptation:** Conditioning a generative temporal model on time-of-day ($t \in [0, 24]$) to match biological human active/sleep cycles.
* **Refractory Floor:** Minimum allowable delay ($\Delta t \ge 1.5\text{s}$) preventing burst alarms.

#### C. Exact Numerical Data to Quote
* **Static Cadence:** Interception rate **$100.0\%$**, Evasion rate **$0.0\%$**, Delay entropy **$0.000\text{ bits}$**, Mean delay $3.00\text{s}$.
* **Stochastic Poisson:** Interception rate **$60.0\%$**, Evasion rate **$40.0\%$**, Delay entropy $4.261\text{ bits}$, Mean delay $2.95\text{s}$.
* **DCASS Temporal WGAN:** Interception rate **$0.0\%$**, Evasion rate **$100.0\%$**, Delay entropy $3.381\text{ bits}$, Mean delay $3.93\text{s}$.
* **DCASS Hybrid Engine:** Interception rate **$0.0\%$**, Evasion rate **$100.0\%$**, Delay entropy $3.435\text{ bits}$, Mean delay $3.92\text{s}$, **Zero channel violations**.
* **DCASS PPO RL Agent:** Interception rate $30.0\%$, Evasion rate $70.0\%$, Mean delay $15.22\text{s}$ ($2.9\times$ speedup over baseline $44\text{s}$).
* **Diurnal Breakdown:**
  * Daytime Peak (14:00): Mean delay **$3.19\text{s}$** ($\sigma = 1.79\text{s}$), Evasion **$100.0\%$**.
  * Night Dormancy (03:00): Mean delay **$13.07\text{s}$** ($\sigma = 6.08\text{s}$), Evasion **$100.0\%$**.

#### D. "Why It Acts Like This" (Behavioral & Statistical Deep-Dive)
* *Why Static fails ($100\%$ interception):* A fixed $3.0\text{s}$ metronome has zero variance and zero entropy ($0.0\text{ bits}$), immediately triggering periodicity/autocorrelation detectors.
* *Why Poisson fails ($60\%$ interception):* Due to the memoryless property $P(\Delta t < t) = 1 - e^{-\lambda t}$, over $25\%$ of Poisson intervals fall below $0.91\text{s}$, causing packet clumping that trips the sliding-window burst filter.
* *Why DCASS CDF is shifted to the right of human baseline below $1.5\text{s}$:* In natural human traffic, $25\%$ of clicks are rapid micro-bursts ($<1.5\text{s}$). If a covert sender sends multi-carrier image payloads at that speed, it triggers API rate limits (HTTP 429) or burst alarms. DCASS enforces a **refractory floor** ($>1.5\text{s}$), stripping away suicidal micro-bursts to guarantee $100\%$ evasion.
* *Why the human curve plateaus on the right ($>500\text{s}$):* Human internet usage is bimodal: $90\%$ active browsing ($1-10\text{s}$) and $10\%$ session hiatuses ($\sim 3,600\text{s}$ / 1 hour). DCASS covert communication completes an entire 50-carrier message within $3.27\text{ minutes}$ without stalling for an hour mid-sentence.

---

### Component 4: Codec Capacity, Permutation & Channel Fault Tolerance

#### A. Artifacts to Reference
* **LaTeX Tables:** `storage/benchmarks/tables/table4a_capacity.tex`, `table4b_codec_robustness.tex`
* **Plots:** `storage/benchmarks/plots/fig_4a_capacity_comparison.png`, `fig_4b_permutation_scaling.png`, `fig_4b_ecc_recovery_curve.png`, `fig_4b_key_separation.png`
* **Source Metrics:** `storage/benchmarks/component_4a_capacity_results.json`, `component_4b_robustness_results.json`

#### B. Key Terms to Define in the Write-Up
* **Bits Per Carrier (BPC):** The channel bit density per transmitted media file ($\text{BPC} = \log_2(K)$).
* **Reed-Solomon Error Correction (RS-ECC):** Block code operating over Galois Field $\mathrm{GF}(2^8)$ appending $R$ parity bytes to payload blocks, correctable up to $t = \lfloor R/2 \rfloor$ arbitrary symbol errors or packet dropouts.
* **Cryptographic Avalanche Effect:** Property where flipping 1 bit in a key changes approximately $50\%$ of the output permutation.
* **Spearman Rank Correlation ($\rho$):** Non-parametric measure of statistical dependence between two permutations. $\rho \approx 0.00$ implies perfect statistical independence.
* **False Acceptance Rate (FAR):** Probability that an unauthorized or mutated key decrypts valid, authenticated plaintext.

#### C. Exact Numerical Data to Quote
* **Bit Density:** **$8.00\text{ bpc}$** deterministic channel rate across all 256 Voronoi centroids.
* **Permutation Scaling:** $0.056\text{ ms}$ at $N=1,000$ candidates, scaling to $0.568\text{ ms}$ at $N=35,000$ candidates.
* **Channel Loss Recovery:** $100.0\%$ perfect reconstruction at $0\%$ loss; $70.0\%$ recovery at $5\%$ loss with $2.7\text{ bytes}$ fixed; graceful degradation beyond RS Shannon bound.
* **Cryptographic Key Separation:**
  * 100 independent random wrong keys: **$0/100$ accepted ($\text{FAR} = 0.000000$)**.
  * Exhaustive 256 single-bit flipped keys: **$0/256$ accepted ($\text{FAR} = 0.000000$)**.
  * Mean key sensitivity $\rho = -0.00198 \approx 0.00$.

#### D. "Why It Acts Like This" (Cryptographic & Algebraic Deep-Dive)
* *Why capacity is exactly 8.00 bpc:* With $K=256$ centroids, every carrier index maps to a unique 8-bit byte ($\log_2 256 = 8$).
* *Why FAR is strictly 0.000000:* Inverting the Voronoi centroid sequence with an incorrect key applies a random permutation $\pi_{\text{wrong}}^{-1}$ to the RS codewords. The Berlekamp-Massey decoder detects that the syndrome polynomial is non-zero with uncorrectable errors, raising an uncorrectable syndrome exception and immediately rejecting forgery.

---

### Component 5: Computational Latency, Speed & Concurrency Stress

#### A. Artifacts to Reference
* **LaTeX Table:** `storage/benchmarks/tables/table5_system_performance.tex`
* **Plots:** `storage/benchmarks/plots/fig5_latency_breakdown.png`, `fig5_concurrency_scaling.png`
* **Source Metrics:** `storage/benchmarks/component_5_results.json`

#### B. Key Terms to Define in the Write-Up
* **End-to-End Encoding Latency ($T_{\mathrm{enc}}$):** Wall-clock time to convert plaintext string into carrier IDs.
* **End-to-End Decoding Latency ($T_{\mathrm{dec}}$):** Wall-clock time to map carrier IDs back to centroids and reconstruct plaintext.
* **Latency Percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$):** Response time distribution under multi-client thread load.
* **Session Cross-Talk Rate:** Percentage of sessions where one client's plaintext leaks into another concurrent client's session ($\equiv 0.000\%$).
* **Requests Per Second (RPS):** System throughput capacity on a single node.

#### C. Exact Numerical Data to Quote
* **Payload Scaling:**
  * $32\text{B}$ payload ($27$ carriers): Encode **$94.6\text{ ms}$**, Decode **$216.0\text{ ms}$**, Throughput $0.33\text{ KB/s}$, Carrier speed $86.9\text{ carriers/s}$.
  * $64\text{B}$ payload ($46$ carriers): Encode **$99.1\text{ ms}$**, Decode **$298.1\text{ ms}$**, Throughput $0.63\text{ KB/s}$, Carrier speed $115.8\text{ carriers/s}$.
  * $128\text{B}$ payload ($83$ carriers): Encode **$117.7\text{ ms}$**, Decode **$462.5\text{ ms}$**, Throughput $1.06\text{ KB/s}$, Carrier speed $143.0\text{ carriers/s}$.
  * $256\text{B}$ payload ($163$ carriers): Encode **$136.0\text{ ms}$**, Decode **$787.6\text{ ms}$**, Throughput $1.84\text{ KB/s}$, Carrier speed $176.5\text{ carriers/s}$.
  * $512\text{B}$ payload ($317$ carriers): Encode **$191.6\text{ ms}$**, Decode **$1442.1\text{ ms}$**, Throughput $2.61\text{ KB/s}$, Carrier speed $194.0\text{ carriers/s}$.
* **Concurrency Stress Testing (100 sessions per level):**
  * 10 Threads: Throughput **$2.73\text{ RPS}$**, $p_{50} = 3,632\text{ ms}$, $p_{95} = 4,129\text{ ms}$, Success **$100.0\%$**, Cross-Talk **$0.000\%$**.
  * 20 Threads: Throughput **$2.62\text{ RPS}$**, $p_{50} = 7,229\text{ ms}$, $p_{95} = 8,969\text{ ms}$, Success **$100.0\%$**, Cross-Talk **$0.000\%$**.
  * 50 Threads: Throughput **$2.57\text{ RPS}$**, $p_{50} = 17,668\text{ ms}$, $p_{95} = 20,573\text{ ms}$, Success **$100.0\%$**, Cross-Talk **$0.000\%$**.

#### D. "Why It Acts Like This" (Systems Deep-Dive)
* *Why encoding is sub-100ms:* Sub-stage profiling reveals that HMAC-DRBG permutation ($0.066\text{ ms}$) and RS-ECC framing ($0.022\text{ ms}$) have negligible overhead. The FAISS vector search across $245,000$ index items is executed on GPU/IVFFlat, completing within $94\text{ ms}$.
* *Why cross-talk is strictly 0.000%:* Session state is strictly partitioned per UUID and per 32-byte cryptographic key; no global variables are mutated in the critical encoding/decoding path.

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
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_ks_heatmap.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_wasserstein_heatmap.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_manifold_tsne.png}
  \caption{Cross-modal distributional drift evaluation: KS distance, Wasserstein-1 divergence, and 2D t-SNE manifold alignment.}
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
  \caption{Traffic timing analysis: Inter-arrival CDFs vs human baseline, DPI interception rates, and circadian diurnal adaptation.}
  \label{fig:traffic_evasion}
\end{figure*}

% --- V-D: Codec Capacity & Channel Faults ---
\subsection{Codec Capacity, Fault Tolerance, and Cryptographic Isolation}
\input{storage/benchmarks/tables/table4a_capacity.tex}
\input{storage/benchmarks/tables/table4b_codec_robustness.tex}
% Embed Figures for Component 4:
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.24\linewidth]{storage/benchmarks/plots/fig_4a_capacity_comparison.png}
  \includegraphics[width=0.24\linewidth]{storage/benchmarks/plots/fig_4b_permutation_scaling.png}
  \includegraphics[width=0.24\linewidth]{storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png}
  \includegraphics[width=0.24\linewidth]{storage/benchmarks/plots/fig_4b_key_separation.png}
  \caption{Codec performance: Bit-density scaling, permutation latency, RS-ECC channel recovery, and zero FAR key isolation.}
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
| **"Isn't coverless steganography vulnerable to semantic selection bias?"** | Cite **Component 2**: $W_1 < 0.02$ and $D_{\mathrm{KS}} < 0.05$. Show that HMAC-DRBG permutation uniformly scatters carriers across all 256 Voronoi centroids ($H(C) = 7.994\text{ bits}$), matching natural corpus statistics. |
| **"Doesn't automated network transmission trigger DPI flow alarms?"** | Cite **Component 3**: DCASS Hybrid Engine achieves **$100.0\%$ evasion** by enforcing a refractory floor ($>1.5\text{s}$), complying with platform cooldowns ($T_c$), and adapting to circadian diurnal cycles ($3.19\text{s}$ daytime vs $13.07\text{s}$ night). |
| **"What if the channel experiences packet loss or an attacker attempts key forgery?"** | Cite **Component 4**: Galois Field $\mathrm{GF}(2^8)$ RS-ECC recovers dropouts up to the parity bound, while 256-bit key sensitivity yields $\rho \approx 0.00$ and **$\text{FAR} = 0.000000$** (100% rejection across all mutated keys). |
| **"Is vector search over hundreds of thousands of media items too slow for real-time use?"** | Cite **Component 5**: End-to-end encoding requires only **$94.6\text{ ms}$** at 32B, throughput reaches **$194\text{ carriers/sec}$**, and concurrent stress tests demonstrate **$0.000\%$ cross-talk** across 50 parallel threads. |

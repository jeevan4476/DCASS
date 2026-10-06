# Component 1: Media-Layer Content Steganalysis
## Pixel-Level & Acoustic Invariance Evaluation

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Empirical Proof of Cachin $\epsilon$-Steganographic Security ($\epsilon = 0$) against Deep Residual and Statistical Detectors  
> **Implementation Script:** `scripts/analysis/benchmark_1_media_steganalysis.py`

---

## 1. Threat Model & Theoretical Mechanics

In spatial and acoustic steganography, the adversary is a **Passive Warden** who intercepts media objects $I$ transmitted across public networks and attempts to solve the binary hypothesis testing problem:

$$\mathcal{H}_0: I \sim P_{\text{cover}} \quad \text{vs.} \quad \mathcal{H}_1: I \sim P_{\text{stego}}$$

### 1.1 Why Traditional Steganography Fails
Traditional embedding algorithms (LSB Matching, S-UNIWARD, WOW, HILL) modify individual pixel values by $\Delta \in \{-1, 0, +1\}$. While invisible to human vision, natural images exhibit high spatial smoothness: neighboring pixels have high correlation ($\rho \approx 0.95\text{--}0.99$).

When payload noise $\mathbf{n}$ is injected:
$$I_{\text{stego}} = I_{\text{cover}} + \mathbf{n}$$
The high-frequency noise variance $\sigma_n^2$ disrupts local inter-pixel dependencies. Detectors isolate this noise by applying high-pass spatial filter kernels $K$ that subtract neighboring pixels:
$$R = I * K$$
In natural regions, smooth image content cancels out ($R \approx 0$). In stego images, the independent embedding noise does not cancel out, causing a detectable surge in residual variance and co-occurrence energy.

### 1.2 Why DCASS Achieves Strict Invariance ($D_{KL} \equiv 0.0000$)
DCASS employs **Dynamic Semantic State-Space Coding (DSSC)**. Under DSSC:
1. The message is decomposed into semantic chunks.
2. The sender selects an unmodified carrier media item $I_{\text{carrier}}$ directly from the public corpus $\mathcal{C}_{\text{public}}$ based on a session-keyed permutation of Voronoi cluster states.
3. The transmitted file is **bit-for-bit identical** to the public corpus file:
   $$\Delta \equiv 0 \implies I_{\text{carrier}} \equiv I_{\text{cover}}$$
4. Consequently, for any feature extraction function $\Phi(\cdot)$ and any detector classifier $f_\theta(\cdot)$:
   $$f_\theta(\Phi(I_{\text{carrier}})) \equiv f_\theta(\Phi(I_{\text{cover}}))$$
   $$P(f_\theta = 1 \mid \mathcal{H}_1) \equiv P(f_\theta = 1 \mid \mathcal{H}_0)$$
   $$\text{ROC AUC} \equiv 0.5000 \quad \text{and} \quad P_E \equiv 0.5000 \quad (\text{Pure Coin Flip})$$

---

## 2. In-Depth Architecture of the 5 Image Steganalysis Detectors

```
                       ┌──────────────────────────────────────────────┐
                       │           Input Image I (1xHxW)              │
                       └──────────────────────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
      ┌───────────────────────────────┐               ┌───────────────────────────────┐
      │   Classical SRM Pipeline      │               │   Deep CNN Steganalysts       │
      │   (Fridrich & Kodovsky 2012)  │               │   (Xu, Ye, Zhu, SRNet)        │
      └───────────────────────────────┘               └───────────────────────────────┘
                      │                                               │
                      ▼                                               ▼
      ┌───────────────────────────────┐               ┌───────────────────────────────┐
      │ 30 High-Pass Filter Residuals │               │ High-Pass Residual Front-End  │
      │ 4-Pixel Spatial Co-occurrence │               │ Specialized Non-Linearity     │
      │ 34,671-d Feature Vector       │               │ (ABS / TLU / Unpooled Skips)  │
      └───────────────────────────────┘               └───────────────────────────────┘
                      │                                               │
                      ▼                                               ▼
      ┌───────────────────────────────┐               ┌───────────────────────────────┐
      │ Fisher Linear Discriminant    │               │ Softmax Classification Head   │
      │ Ensemble Classifier (FLDA)    │               │ P(Stego) vs P(Cover)          │
      └───────────────────────────────┘               └───────────────────────────────┘
```

---

### 2.1 Detector 1: Spatial Rich Model (SRM) + FLDA
* **Citation:** J. Fridrich and J. Kodovský, *"Rich Models for Steganalysis of Digital Images"*, IEEE Transactions on Information Forensics and Security (TIFS), 2012.
* **Filter Bank:** 30 high-pass submodel filters capturing horizontal, vertical, diagonal, edge, and corner dependencies:
  - 1st-order differences: $D_h = [ -1, 1 ]$, $D_v = [ -1, 1 ]^T$
  - 2nd-order differences: $[ -1, 2, -1 ]$
  - 3rd-order differences: $[ -1, 3, -3, 1 ]$
  - 3x3 Laplacian:
    $$K_{\text{Lap}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
  - 5x5 EDGE and SQUARE kernels (Min/Max non-linear filters).
* **Quantization & Truncation:**
  Residuals are quantized by step size $q \in \{1, 1.5, 2\}$ and truncated to range $[-T, T]$ with $T=2$:
  $$R_{i,j}^{(q)} = \text{trunc}_T \left( \text{round} \left( \frac{R_{i,j}}{q} \right) \right)$$
* **Co-occurrence Tensor:**
  Computes 4D joint empirical histograms of 4 horizontally/vertically adjacent quantized residual symbols:
  $$\mathbf{C}(d_1, d_2, d_3, d_4) = \sum_{i,j} \mathbb{I}(R_{i,j}=d_1, R_{i,j+1}=d_2, R_{i,j+2}=d_3, R_{i,j+3}=d_4)$$
* **Feature Dimension:** Symmetrized co-occurrences form a **34,671-dimensional** feature vector.
* **Classifier:** Random subspace Ensemble Classifier consisting of Fisher Linear Discriminants (FLDA) with ridge regularization.

---

### 2.2 Detector 2: Xu-Net
* **Citation:** G. Xu, H. Z. Wu, and Y. Q. Shi, *"Structural Design of Convolutional Neural Networks for Steganalysis"*, IEEE Signal Processing Letters, 2016.
* **Key Theoretical Contributions:**
  1. **Fixed KV High-Pass Filter:** Layer 1 is frozen with the $5\times 5$ KV kernel to strip low-frequency image content:
     $$K_{\text{KV}} = \frac{1}{12} \begin{bmatrix} -1 & 2 & -2 & 2 & -1 \\ 2 & -6 & 8 & -6 & 2 \\ -2 & 8 & -12 & 8 & -2 \\ 2 & -6 & 8 & -6 & 2 \\ -1 & 2 & -2 & 2 & -1 \end{bmatrix}$$
  2. **Absolute Value (ABS) Activation:**
     Steganographic embedding noise $\pm 1$ has symmetric positive and negative signs. Standard ReLUs zero out all negative activations ($x < 0 \to 0$), throwing away 50% of the stego signal. Xu-Net places an $\text{ABS}(x) = |x|$ activation immediately after the KV filter, mapping negative embedding perturbations into positive energy.
  3. **Average Pooling (Never Max Pooling):**
     Max pooling selects only extreme local values, which deletes weak, diffuse steganographic noise. Xu-Net strictly enforces $5\times 5$ average pooling with stride 2.
* **Network Layers:**
  - `Input (1xHxW)` $\to$ `Conv2d(KV, 5x5, stride 1)` $\to$ `ABS()`
  - `Conv2d(16, 5x5)` $\to$ `BatchNorm` $\to$ `Tanh` $\to$ `AvgPool2d(5x5, stride 2)`
  - `Conv2d(32, 5x5)` $\to$ `BatchNorm` $\to$ `Tanh` $\to$ `AvgPool2d(5x5, stride 2)`
  - `Conv2d(64, 5x5)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AvgPool2d(5x5, stride 2)`
  - `Conv2d(128, 5x5)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AvgPool2d(Adaptive 1x1)`
  - `Linear(128, 2)` $\to$ `Softmax`

---

### 2.3 Detector 3: Ye-Net
* **Citation:** J. Ye, J. Ni, and Y. Yi, *"Deep Learning Hierarchical Representations for Image Steganalysis"*, IEEE Transactions on Information Forensics and Security (TIFS), 2017.
* **Key Theoretical Contributions:**
  1. **30-SRM Filter Initialization:** Layer 1 is initialized with all 30 SRM spatial filters and fine-tuned during backpropagation.
  2. **Truncated Linear Units (TLU):**
     Large residual values come from sharp image edges (high-contrast visual boundaries that confuse the detector). Small residual values come from steganographic noise. TLU clamps activations to $[-T, T]$ ($T = 3.0$):
     $$\text{TLU}_T(x) = \max(-T, \min(T, x))$$
     This bounds the dynamic range, forcing the network to ignore large natural edges and focus solely on micro-variations.
* **Network Layers:**
  - `Conv2d(30 SRM Filters, 5x5)` $\to$ `TLU(T=3.0)`
  - `Conv2d(30, 32, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AvgPool2d(3x3, stride 2)`
  - `Conv2d(32, 32, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AvgPool2d(3x3, stride 2)`
  - `Conv2d(32, 64, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AvgPool2d(3x3, stride 2)`
  - `Linear(64, 2)` $\to$ `Softmax`

---

### 2.4 Detector 4: Zhu-Net
* **Citation:** K. Zhu, C. Chen, and Y. Q. Shi, *"Universal Steganalysis of Digital Images Based on Deep Separable Convolutions"*, IEEE Transactions on Information Forensics and Security (TIFS), 2020.
* **Key Theoretical Contributions:**
  1. **Depthwise Separable Convolutions:** Splits standard spatial-channel convolution into:
     - Depthwise Convolution: $3\times 3$ spatial filtering per channel independently.
     - Pointwise Convolution: $1\times 1$ linear combination across channels.
     This dramatically cuts parameters while capturing inter-channel and directional noise correlations.
  2. **Spatial Pyramid Pooling (SPP):**
     Captures multi-scale residual noise patterns across pooling bin dimensions $\{1\times 1, 2\times 2, 4\times 4\}$, allowing the network to process variable-resolution image patches.

---

### 2.5 Detector 5: SRNet (Spatial Residual Network — The Gold Standard)
* **Citation:** M. Boroumand, M. Chen, and J. Fridrich, *"Deep Residual Network for Steganalysis of Digital Images"*, IEEE Transactions on Information Forensics and Security (TIFS), 2018.
* **The Paradigm Shift:**
  Previous CNNs (ResNet, VGG) failed in steganalysis because standard downsampling early in the network averages out subtle embedding noise. SRNet introduced a **12-layer residual architecture** with four distinct block types:
* **Block Architectures:**
  1. **Type 1 (Layers 1–2 — Front End):**
     $3\times 3$ Conv (64 channels) $\to$ BatchNorm $\to$ ReLU. Full spatial resolution ($1\times$) is strictly preserved.
  2. **Type 2 (Layers 3–7 — Unpooled Residual Blocks):**
     5 residual blocks with identity skip connections and **zero pooling**:
     $$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$
     $$\mathcal{F}(\mathbf{x}) = \text{Conv}(3\times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv}(3\times 3) \to \text{BN}$$
     The network learns complex spatial noise correlations over 5 deep layers without downsampling.
  3. **Type 3 (Layers 8–11 — Downsampling Residual Blocks):**
     4 residual blocks where downsampling is performed gradually via $3\times 3$ Average Pooling (stride 2) on both the residual path and skip path:
     $$\mathbf{y} = \text{AvgPool}(\text{Conv}_{1\times 1}(\mathbf{x})) + \text{AvgPool}(\mathcal{F}(\mathbf{x}))$$
  4. **Type 4 (Layer 12 — Decision Head):**
     Global Average Pooling $\to$ Fully Connected(512) $\to$ Fully Connected(2) $\to$ Softmax.
* **Efficacy:** Capable of detecting S-UNIWARD, HILL, and WOW at ultra-low payloads down to $0.1$ bits per pixel.

---

## 3. Acoustic Steganalysis (Audio Carrier Verification)

In addition to image carriers, DCASS transmits audio carriers (ESC-50 environmental sounds).
* **Adversarial Evaluator:** Acoustic Spectral Residual Detector.
* **Feature Extraction:**
  1. Short-Time Fourier Transform (STFT) with Hann window (2048 FFT size, 512 hop length).
  2. Compute 128-band Mel-spectrogram: $\mathbf{S}_{\text{mel}}(f, t)$.
  3. Extract high-frequency acoustic spectral flux and phase modulation variance:
     $$\Delta \Phi(f, t) = \left| \mathbf{S}(f, t) - \mathbf{S}(f, t-1) \right|$$
* **Baseline Comparison:**
  - Positive Control: Echo steganography (delay $\tau = 1\text{ms}$) and phase coding on ESC-50 clips.
  - DCASS Audio Carriers: Unmodified WAV files selected by DSSC.
  - Metric: Spectral distortion distance and relative entropy $D_{KL} = 0.0000$.

---

## 4. The Experimental Protocol (Why Option B is Required)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CRITICAL EXPERIMENTAL INTEGRITY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. NEVER use BOSSbase uncompressed weights on Flickr JPEG images:           │
│    Doing so causes Cover-Source Mismatch (CSM) — the detector fails due to  │
│    JPEG compression artifacts, not steganography. Reviewers will reject it. │
│                                                                             │
│ 2. OPTION B PROTOCOL (Standard in Top-Tier Security Conferences):           │
│    • Train the detectors directly on the Flickr30k corpus.                  │
│    • Negative Class: 350 Natural Flickr Images.                             │
│    • Positive Class: 350 Flickr Images + Spatial ±1 LSB Stego (0.4 bpp).    │
│    • Verify: Detector achieves > 90% accuracy on held-out stego test set!   │
│    • Test on DCASS: Feed 150 untouched DCASS carriers into the SAME model! │
│    • Result: Accuracy collapses to 50.00% (AUC = 0.5000, D_KL = 0.0000).    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Quantitative Evaluation Metrics

1. **Total Classification Error Probability ($P_E$):**
   $$P_E = \min_{\tau} \frac{1}{2} \left( P_{FA}(\tau) + P_{MD}(\tau) \right)$$
   - $P_{FA}$: False Alarm Rate ($\Pr(\text{Predict Stego} \mid \text{Cover})$).
   - $P_{MD}$: Missed Detection Rate ($\Pr(\text{Predict Cover} \mid \text{Stego})$).
   - For a blind detector, $P_{FA} = 0.5$ and $P_{MD} = 0.5 \implies P_E = 0.5000$.

2. **Empirical ROC Area Under Curve (AUC):**
   $$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
   Computed via `sklearn.metrics.roc_auc_score` over model predicted probabilities.
   - Traditional Stego: $\text{AUC} \ge 0.9400$.
   - DCASS Carriers: $\text{AUC} = 0.5000 \pm \epsilon$ (pure random guessing).

3. **Relative Entropy / Kullback-Leibler Divergence ($D_{KL}$):**
   $$D_{KL}(P_{\text{cover}} \parallel P_{\text{carrier}}) = \sum_{i=1}^B P_{\text{cover}}(b_i) \log_2 \left( \frac{P_{\text{cover}}(b_i)}{P_{\text{carrier}}(b_i)} \right)$$
   Computed via `scipy.stats.entropy` over empirical probability density histograms (30 bins).
   - Proves Cachin $\epsilon$-security: $\epsilon = 0.0000$ bits.

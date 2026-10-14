# Component 2: Semantic Distribution & Selection-Bias Benchmark
## Manifold Divergence, Statistical Indistinguishability & Cluster Entropy

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Defeat the Statistical "Semantic Warden" by proving that the carrier selection process introduces zero statistically detectable distribution bias.  
> **Implementation Script:** `scripts/analysis/benchmark_2_semantic_distribution.py`

---

## 1. The Coverless Threat Model (The Semantic Warden)

In digital steganography, a naive adversary only inspects pixel or sample residuals. However, a modern security adversary (the **Semantic Warden**) recognizes that coverless steganography does not alter pixels.

### The Attack Surface:
The Semantic Warden intercepts a sequence of media files $\mathcal{S} = \{m_1, m_2, \dots, m_K\}$ transmitted by a user and asks:
> *"Does the statistical distribution of these selected media items deviate from the natural distribution of media items posted by ordinary users on this platform?"*

If a coverless system naively selects carriers:
* **Clustering Hotspot Vulnerability:** If the algorithm maps frequently used characters or bits into a narrow subset of carriers, those specific media items will appear unnaturally often, creating detectable frequency spikes.
* **Semantic Anomaly Vulnerability:** If carriers are sampled from weird semantic corners, their feature representations will form an anomalous cluster distinct from the general population.
* **Modality Skew Vulnerability:** If the system claims to be multi-modal but transmits 99% text and 1% images, the channel footprint is unbalanced and suspicious.

---

## 2. Theoretical Formulation & Manifold Geometry

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CLIP/CLAP MULTI-MODAL MANIFOLD                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│        Natural Corpus Distribution P_cov            DCASS Carrier Set P_stego│
│             (1,000 Random Samples)                   (1,000 DSSC Carriers)  │
│                                                                             │
│                  •     •                                •     •             │
│               •     •     •                          •     •     •          │
│             •    [Centroid]  •                     •    [Centroid]  •       │
│               •     •     •                          •     •     •          │
│                  •     •                                •     •             │
│                                                                             │
│      Statistical Hypothesis Test: H_0: P_stego == P_cov (p-value > 0.05)    │
│      Wasserstein-1 Distance:      W_1(P_stego, P_cov) -> 0.000             │
│      Voronoi Cluster Entropy:     H(C) -> 8.00 bits (Uniform Codebook Use)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Let $\mathcal{M} \subset \mathbb{R}^D$ be the $D$-dimensional continuous embedding manifold defined by pre-trained foundation models:
* Image and Text Embeddings: CLIP ViT-B/32 ($D = 512$).
* Audio Embeddings: CLAP HTSAT-unfused ($D = 512$).

Let $\mathbf{z} = \Phi(m) \in \mathbb{R}^D$ be the normalized latent embedding vector of media item $m$ ($\|\mathbf{z}\|_2 = 1$).
Let $\mathcal{Z}_{\text{corpus}} \sim P_{\text{corpus}}$ be the distribution of embeddings of the public reference corpus.
Let $\mathcal{Z}_{\text{carrier}} \sim P_{\text{carrier}}$ be the distribution of embeddings of DCASS transmitted carriers.

---

## 3. Evaluators & Statistical Tests

### 3.1 Two-Sample Kolmogorov-Smirnov (KS) Test
The Two-Sample KS test is a non-parametric test evaluating whether two continuous empirical distributions are drawn from the same underlying probability distribution:

$$D_{KS} = \sup_{x} \left| F_{\text{carrier}}(x) - F_{\text{corpus}}(x) \right|$$

* **Null Hypothesis ($H_0$):** $P_{\text{carrier}} = P_{\text{corpus}}$ (The carrier selection is drawn identically from the background corpus).
* **Significance Level:** $\alpha = 0.05$.
* **Evaluation Criterion:** If the asymptotic $p$-value satisfies:
  $$p > 0.05$$
  The adversary **fails to reject the null hypothesis**. The carrier set is statistically indistinguishable from natural sampling.
* **Dimensional Projection:** The KS test is evaluated across each of the 512 latent embedding dimensions as well as the 1D PCA/t-SNE projection of the manifold.

---

### 3.2 Wasserstein-1 Distance (Earth Mover's Distance)
The Wasserstein-1 distance measures the minimum cost of transporting probability mass from the carrier distribution to the natural corpus distribution:

$$W_1(P_{\text{corpus}}, P_{\text{carrier}}) = \inf_{\gamma \in \Pi(P_{\text{corpus}}, P_{\text{carrier}})} \mathbb{E}_{(x, y) \sim \gamma} \left[ \| x - y \| \right]$$

* By the Kantorovich-Rubinstein duality:
  $$W_1(u, v) = \int_{-\infty}^{+\infty} | U(x) - V(x) | \, dx$$
  Where $U(x)$ and $V(x)$ are the cumulative distribution functions of the 1D projected representations.
* **Interpretation:**
  - Traditional steganography with semantic distortion: $W_1 > 0.25$.
  - DCASS DSSC Selection: $W_1 < 0.02$ (vanishingly small transport cost, demonstrating manifold alignment).

---

### 3.3 Voronoi Codebook Cluster Coverage Entropy
DCASS partitions the multi-modal semantic space into $K = 256$ Voronoi topological clusters using balanced spherical $k$-means on normalized CLIP/CLAP vectors.

Let $c_i \in \{0, 1, \dots, 255\}$ be the cluster index of a transmitted carrier. Over a transmission of $M$ carriers, let $n_k$ be the count of carriers assigned to cluster $k$, with empirical probability $p_k = n_k / M$.

The **Cluster Coverage Entropy** is:
$$H(C) = -\sum_{k=0}^{K-1} p_k \log_2(p_k)$$

* **Theoretical Maximum Entropy:**
  $$H_{\max} = \log_2(256) = 8.0000 \text{ bits}$$
  (Occurs when carriers are perfectly uniformly distributed across all 256 clusters).
* **The "Hotspot" Anomaly:**
  If an adversary observes $H(C) \ll 8.0$ (e.g. $H < 4.0$ bits), the system is repeatedly hitting only a few clusters (hotspots), exposing the channel.
* **DCASS Target:**
  Because DSSC applies session-keyed pseudo-random permutations $\pi_s$ over the admissible candidate spaces:
  $$H_{\text{DCASS}}(C) \ge 7.60 \text{ bits} \quad (\approx 95\%\text{ of theoretical maximum})$$
  This proves that carriers span the entire topological volume without clustering in detectable corners.

---

## 4. Multi-Modal Balance Verification

Under the Tier-1 enhancement in [`src/engine/dssc_encoder.py`](file:///home/img1/projects/DCASS/src/engine/dssc_encoder.py), DSSC enforces deterministic round-robin modality cycling across chunks:
$$\text{modality}_i = \text{MODALITY\_SEQUENCE}[i \pmod 3] \quad (\text{Text} \to \text{Image} \to \text{Audio})$$

The benchmark transmits 50 diverse messages (generating >1,000 carriers) and measures the empirical modality proportions:

$$\mathbf{p}_{\text{modal}} = \left[ \frac{N_{\text{text}}}{N_{\text{total}}}, \frac{N_{\text{image}}}{N_{\text{total}}}, \frac{N_{\text{audio}}}{N_{\text{total}}} \right]$$

* **Corpus Skew Before Fix:** $93\%$ text, $5\%$ audio, $2\%$ image (heavily dominated by Wikipedia).
* **DSSC Balanced Target:**
  $$\text{Text}: 40\text{--}50\%, \quad \text{Audio}: 30\text{--}35\%, \quad \text{Image}: 15\text{--}25\%$$
  This proves that DCASS is genuinely multi-modal across text, vision, and acoustics.

---

## 5. Output Artifacts Generated

The script outputs:
1. `paper_tables/table2_distributional_drift.tex`:
   - Wasserstein-1 distance per modality.
   - Mean Kolmogorov-Smirnov statistic and $p$-value.
   - Empirical cluster entropy $H(C)$ vs $H_{\max}$.
2. `paper_plots/fig_manifold_tsne.png`:
   - 2D t-SNE plot visualizing natural corpus points (blue) vs DCASS carriers (red), showing complete geometric overlap.

# Component 2: Semantic Distribution & Selection-Bias Benchmark
## Manifold Divergence, Selection Bias & Cluster Entropy

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Goal:** Measure whether carrier selection introduces distributional bias beyond the pipeline's own public mechanism, and state the result with its scope. The recorded outcome is qualified: image and text carriers sit about one ten-thousandth away from a uniform corpus draw at permutation p 0.009 and 0.004, audio resolves nothing at p 0.57, 0/1,536 KS coordinates reject after Benjamini-Hochberg control, and the uniform cluster null is rejected while the mechanism null is not.
> **Implementation Script:** `scripts/analysis/benchmark_2_semantic_distribution.py`  
> **Companion Script:** `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py`

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
│    Disjoint Reference Corpus P_corpus          DCASS Carrier Set P_carrier  │
│         (5,000 embeddings)                        (5,000 balanced slots)    │
│                                                                             │
│                  •     •                                •     •             │
│               •     •     •                          •     •     •          │
│             •    [Centroid]  •                     •    [Centroid]  •       │
│               •     •     •                          •     •     •          │
│                  •     •                                •     •             │
│                                                                             │
│   Headline tests: energy distance and RBF MMD^2, calibrated by 1,000        │
│   joint label permutations on one shared distance matrix                    │
│   Secondary: W_1 on PC1 of a pooled PCA fit, with its variance share        │
│   Cluster entropy H(C) read against the structural mixture identity         │
└─────────────────────────────────────────────────────────────────────────────┘
```

Let $\mathcal{M} \subset \mathbb{R}^D$ be the $D$-dimensional continuous embedding manifold defined by pre-trained foundation models:
* Image and Text Embeddings: CLIP ViT-B/32 ($D = 512$).
* Audio Embeddings: CLAP HTSAT-unfused ($D = 512$).

Let $\mathbf{z} = \Phi(m) \in \mathbb{R}^D$ be the normalized latent embedding vector of media item $m$ ($\|\mathbf{z}\|_2 = 1$).
Let $\mathcal{Z}_{\text{corpus}} \sim P_{\text{corpus}}$ be the distribution of embeddings of the public reference corpus.
Let $\mathcal{Z}_{\text{carrier}} \sim P_{\text{carrier}}$ be the distribution of embeddings of DCASS transmitted carriers.

The recorded campaign runs ten independent draws of fifty fresh DSSC messages each, five hundred messages and 27,850 carrier slots in total, with fresh session keys per message. Every test builds its reference sample from corpus items that no draw ever transmitted: 5,000 embeddings per modality sampled uniformly from the FAISS offsets after excluding every carrier media id seen in any draw. Reference and carrier sets are disjoint by construction. The headline comparisons subsample the carrier slots to 5,000 as well, keeping the pooled distance matrix balanced at 10,000 points.

---

## 3. Evaluators & Statistical Tests

### 3.1 Headline Multivariate Tests: Energy Distance and MMD²
The headline test is the energy distance of Szekely and Rizzo, the weighted difference between cross-sample and within-sample mean distances:

$$\mathcal{E} = 2\,\mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|$$

It is zero exactly when the two distributions coincide, and the reported estimator uses the unbiased U-statistic form over off-diagonal within-group pairs, so headline and per-draw values share one scale.

The second headline statistic is the squared maximum mean discrepancy under an RBF kernel with median-heuristic bandwidth:

$$\mathrm{MMD}^2 = \mathbb{E}\,k(X, X') + \mathbb{E}\,k(Y, Y') - 2\,\mathbb{E}\,k(X, Y)$$

Both statistics are calibrated by one permutation test: 1,000 relabelings of the pooled 10,000-point set on a shared distance matrix. *Null Hypothesis ($H_0$):* $P_{\text{carrier}} = P_{\text{corpus}}$ against the disjoint reference.

### 3.2 Two-Sample Kolmogorov-Smirnov Family
The Two-Sample KS test is a non-parametric test evaluating whether two continuous empirical distributions are drawn from the same underlying probability distribution:

$$D_{KS} = \sup_{x} \left| F_{\text{carrier}}(x) - F_{\text{corpus}}(x) \right|$$

* **Family:** All 512 embedding dimensions of each modality are screened, 1,536 tests in total, with scipy's two-sample implementation.
* **Multiplicity control:** Benjamini-Hochberg at $q = 0.05$ within each modality family, plus a family-wise permutation test of the maximum ECDF gap over 500 joint relabelings.
* **Separate projection entry:** The pooled-PC1 coordinate is tested once as a prespecified single test, quoted without multiplicity adjustment.
* **Evaluation Criterion:** A BH discovery or a family-wise rejection means the adversary rejects the null; otherwise the coordinate family is statistically indistinguishable from the disjoint reference.

### 3.3 Wasserstein-1 Distance (Earth Mover's Distance)
The Wasserstein-1 distance measures the minimum cost of transporting probability mass from the carrier distribution to the natural corpus distribution:

$$W_1(P_{\text{corpus}}, P_{\text{carrier}}) = \inf_{\gamma \in \Pi(P_{\text{corpus}}, P_{\text{carrier}})} \mathbb{E}_{(x, y) \sim \gamma} \left[ \| x - y \| \right]$$

* By the Kantorovich-Rubinstein duality:
  $$W_1(u, v) = \int_{-\infty}^{+\infty} | U(x) - V(x) | \, dx$$
  Where $U(x)$ and $V(x)$ are the cumulative distribution functions of the 1D projected representations.
* **Protocol:** $W_1$ is measured on the first principal coordinate of a per-modality PCA fit on the pooled sample and reported with the share of pooled variance that coordinate carries. It is explicitly a one-dimensional projection, not a multivariate transport cost, and values are not comparable across modalities with different PC1 shares.

### 3.4 Voronoi Codebook Cluster Coverage Entropy
DCASS partitions the multi-modal semantic space into $K = 256$ Voronoi topological clusters using spherical $k$-means on normalized CLIP/CLAP vectors. The fit is not density balanced: corpus cluster densities span 59 to 8,814 items, with a median of 341.

Let $c_i \in \{0, 1, \dots, 255\}$ be the cluster index of a transmitted carrier. Over a transmission of $M$ carriers, let $n_k$ be the count of carriers assigned to cluster $k$, with empirical probability $p_k = n_k / M$.

The **Cluster Coverage Entropy** is:
$$H(C) = -\sum_{k=0}^{K-1} p_k \log_2(p_k)$$

* **Theoretical Maximum Entropy:**
  $$H_{\max} = \log_2(256) = 8.0000 \text{ bits}$$
  (Occurs when carriers are perfectly uniformly distributed across all 256 clusters).
* **The "Hotspot" Anomaly:**
  If an adversary observes $H(C) \ll 8.0$ (e.g. $H < 4.0$ bits), the system is repeatedly hitting only a few clusters (hotspots), exposing the channel.
* **Structural Deficit:** The recorded entropy, 7.8090 bits over 27,850 slots, is 97.6% of the maximum, and the deficit is not unexploited capacity. The stream is a mixture of three modality distributions under deterministic round robin, and the mixture identity decomposes exactly:
  $$H_{\text{mix}} = \sum_m w_m H(P_m) + H(\mathbf{w})$$
  With measured weights, the weighted modality average is 6.2242 bits and the mixing term is 1.5847 bits, summing to the observed 7.8090 with a residual below two millionths of a bit. The 0.191 bit gap to the ceiling is the structural price of modality routing.
* **Repeat Structure:** The 27,850 slots draw on 23,890 unique media, a reuse factor of 1.1658. Entropy read on unique media alone is 7.7814 bits, so repeats do not hide a different cluster profile.
* **Mechanism Consistency (Component 2b):** A dedicated 200-message campaign compares the 256-bin histogram against two nulls. Against uniformity it rejects decisively, chi-square 3,253.4 with permutation $p = 0.0005$, because the corpus itself is not uniform. Against the pipeline's own structural expectation it is unremarkable, chi-square 268.6 on 253 degrees of freedom with permutation $p = 0.208$. The correct reference for the observed entropy is therefore the mechanism null, which spans 7.7837 to 7.8098 bits over 2,000 simulated campaigns, and not the raw 8.0000-bit ceiling.

---

## 4. Multi-Modal Balance Verification

Under the Tier-1 enhancement in [`src/engine/dssc_encoder.py`](file:///home/img1/projects/DCASS/src/engine/dssc_encoder.py), DSSC assigns each chunk its modality by deterministic round-robin over the availability-filtered modality list (`available_modalities[chunk_idx % len(available_modalities)]`, text-first order), then picks its semantic family HMAC-keyed per chunk (`hmac(session_key, f"family:{chunk_idx}")`, family index = digest mod the modality's native families) — family choice never derives from message text.

The campaign transmits 500 messages across ten draws, 27,850 carrier slots, and measures the empirical modality proportions:

$$\mathbf{p}_{\text{modal}} = \left[ \frac{N_{\text{text}}}{N_{\text{total}}}, \frac{N_{\text{image}}}{N_{\text{total}}}, \frac{N_{\text{audio}}}{N_{\text{total}}} \right]$$

* **Corpus Proportions:** Text $81.54\%$, Image $12.96\%$, Audio $5.50\%$. The raw corpus is dominated by text.
* **DSSC Measured Split:**
  $$\text{Image}: 33.21\%, \quad \text{Text}: 34.11\%, \quad \text{Audio}: 32.68\%$$
  The residual deviation from a perfect third stays under 0.8 percentage points per modality. Round robin is exact at the chunk level, and messages do not split the same number of chunks across modalities, so a few tenths of a point of deterministic packaging skew is expected rather than sampling noise.
  This proves that DCASS is genuinely multi-modal across text, vision, and acoustics.

---

## 5. Output Artifacts Generated

The component writes:
1. `storage/benchmarks/tables/table2_distributional_drift.tex`:
   - Energy distance and MMD² with permutation p-values per modality.
   - Wasserstein-1 on PC1 per modality.
   - KS Benjamini-Hochberg discoveries and family-wise p-values.
   - Unique-media cluster entropy and carrier share.
2. `storage/benchmarks/plots/fig2_distributional_stats.png`:
   - Energy, MMD², and W₁ panels with per-draw dots and calibration p-values.
3. `storage/benchmarks/plots/fig_manifold_tsne.png`:
   - 2D t-SNE plot visualizing disjoint reference points in blue against DCASS carriers in red, showing broad interleaving.
4. `storage/benchmarks/plots/fig_ks_heatmap.png`:
   - $D_{KS}$ across the 512 dimensions per modality with BH and family-wise annotations.
5. `storage/benchmarks/component_2_results.json`:
   - The full schema-v2 record, including per-draw values, ledger counters, reference provenance, and repeat statistics.

The companion Component 2b script writes `storage/benchmarks/component_2b_cluster_usage_results.json`, `storage/benchmarks/plots/fig_2b_cluster_usage.png`, and `storage/benchmarks/tables/table2b_cluster_uniformity.tex`.

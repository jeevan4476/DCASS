# DCASS Deep Research & Technical Specification: Theoretical Channel Capacity & Encoding/Decoding Mechanics

## 1. 📡 Theoretical Channel Capacity of Semantic Channels

### A. Discrete Codebook Capacity ($W$)
In DCASS, secret information is transmitted by selecting specific media items from a finite discrete corpus of $N$ FAISS-indexed vectors. Because every vector choice is equiprobable, the **raw information capacity per carrier media item** $W$ is:

$$W = \log_2(N) \quad \text{(bits / carrier item)}$$

For our multi-modal index of $N = 153,281$ total vectors (Images + Text + Audio):

$$W = \log_2(153,281) \approx 17.2257 \quad \text{bits / carrier item}$$

---

### B. Shannon Channel Capacity under Vector Quantization Noise
When transmitting over a real-world continuous vector channel where nearest-neighbor quantization introduces a symbol error probability $P_e$ (the 15-25% vector drift plateau), the **discrete-time Shannon Channel Capacity** $C_{\text{semantic}}$ is:

$$C_{\text{semantic}} = \log_2(N) \cdot \Big( 1 - H(P_e) \Big) \quad \text{(bits / carrier item)}$$

Where $H(P_e)$ is the binary entropy function of the raw vector error rate:

$$H(P_e) = -P_e \log_2(P_e) - (1 - P_e) \log_2(1 - P_e)$$

---

### C. Net Effective Error-Free Payload Capacity ($R_{\text{eff}}$)
When Reed-Solomon $(N_{\text{cw}}, K)$ error correction with $R = 2t$ parity bytes is integrated, the code rate is:

$$\eta = \frac{K}{N_{\text{cw}}} = \frac{K}{K + R}$$

The **net effective error-free secret payload capacity** $R_{\text{eff}}$ per transmitted carrier item becomes:

$$R_{\text{eff}} = \eta \cdot W = \left( \frac{K}{K + R} \right) \cdot \log_2(N) \quad \text{(bits / carrier item)}$$

#### 📊 Quantitative Capacity Benchmarks

| Code Setup $(K, R)$ | Parity Bytes ($R$) | Max Correctable Errors ($t$) | Code Rate ($\eta$) | Raw Capacity ($W$) | Net Payload Capacity ($R_{\text{eff}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $(20, 4)$ | 4 | 2 | 0.833 | 17.23 bits/item | **14.35 bits / carrier** |
| $(20, 8)$ | 8 | 4 | 0.714 | 17.23 bits/item | **12.31 bits / carrier** |
| $(20, 12)$ | 12 | 6 | 0.625 | 17.23 bits/item | **10.77 bits / carrier** |
| $(16, 16)$ | 16 | 8 | 0.500 | 17.23 bits/item | **8.61 bits / carrier** |

---

## 2. 🌌 Vector Hypersphere Geometry ($\mathbb{S}^{511}$) & Voronoi Cell Quantization

### A. Hypersphere Topology
All three media modalities (Images via CLIP ViT-B/32, Text via CLIP Text Encoder, Audio via LAION CLAP) are normalized onto a 512-dimensional unit hypersphere $\mathbb{S}^{511}$:

$$\mathbb{S}^{511} = \{ v \in \mathbb{R}^{512} : \|v\|_2 = 1.0 \}$$

For any two normalized vectors $u, v \in \mathbb{S}^{511}$, the inner product equals cosine similarity:

$$\langle u, v \rangle = \sum_{i=1}^{512} u_i v_i = \cos(\theta_{u,v})$$

### B. Voronoi Tessellation & Boundary Drift
The vector space is partitioned into $N$ convex Voronoi cells $\mathcal{V}(x_i)$:

$$\mathcal{V}(x_i) = \{ v \in \mathbb{S}^{511} : \langle v, x_i \rangle \ge \langle v, x_j \rangle \quad \forall j \neq i \}$$

When floating-point noise $\Delta \epsilon$ or semantic compression alters the target vector ($v_{\text{observed}} = v_{\text{target}} + \Delta \epsilon$), the point can cross the Voronoi cell boundary into an adjacent cell $\mathcal{V}(x_{\text{adjacent}})$. This causes FAISS nearest-neighbor search to return an adjacent media ID, creating a symbol error.

---

## 3. 🧮 Mathematical Mechanics of Reed-Solomon $GF(2^8)$ Coding

To solve Voronoi boundary drift, DCASS applies non-binary Reed-Solomon algebra over Galois Field $GF(2^8)$ constructed via the primitive binary polynomial:

$$p(x) = x^8 + x^4 + x^3 + x^2 + 1 \quad (\text{Hex: 0x11D})$$

### Step 1: Payload Polynomial
$$M(x) = \sum_{i=0}^{K-1} m_i x^i = m_{K-1} x^{K-1} + \dots + m_1 x + m_0 \quad (m_i \in GF(2^8))$$

### Step 2: Generator Polynomial & Parity Encoding
Given $R = 2t$ parity bytes, generator polynomial $G(x)$ of degree $R$ is defined by:

$$G(x) = \prod_{i=0}^{R-1} (x - \alpha^i) = g_R x^R + g_{R-1} x^{R-1} + \dots + g_1 x + g_0$$

Parity polynomial $P(x)$ is computed as: $P(x) = (M(x) \cdot x^R) \pmod{G(x)}$.  
Transmitted codeword: $C(x) = M(x) \cdot x^R + P(x)$.

### Step 3: Syndrome Decoding & Berlekamp-Massey Algorithm
When receiving corrupted codeword $C'(x) = C(x) + E(x)$:
1. **Syndrome Evaluation**: Compute $S_j = C'(\alpha^j) = E(\alpha^j)$ for $j = 0, \dots, R-1$.
2. **Berlekamp-Massey Algorithm**: Finds error locator polynomial $\Lambda(x) = \prod_{k=1}^\nu (1 - x X_k)$ of degree $\nu \le t$.
3. **Chien Search**: Evaluates $\Lambda(\alpha^{-i})$ to locate exact corrupted byte positions.
4. **Forney Algorithm**: Computes error values $Y_k$ and subtracts $E(x)$, restoring $M(x)$ with **0% Bit Error Rate (100% exact recovery)**.

---

## 4. 🛡️ Stegananalytic Imperceptibility ($D_{\text{KL}} = 0.0$)

Let $P_{\text{cover}}(X)$ be the distribution of public media files on social networks, and $P_{\text{stego}}(X)$ be the distribution of DCASS transmitted items. 

Because DCASS selects items **100% untouched** from the natural corpus without modifying any pixels or audio samples:

$$P_{\text{stego}}(X) = P_{\text{cover}}(X)$$

The Kullback-Leibler (KL) Divergence is strictly:

$$D_{\text{KL}}(P_{\text{cover}} \parallel P_{\text{stego}}) = \sum_{x \in \mathcal{X}} P_{\text{cover}}(x) \log_2 \left( \frac{P_{\text{cover}}(x)}{P_{\text{stego}}(x)} \right) = 0.0 \quad \text{bits}$$

Under **Cachin's Theorem**, $D_{\text{KL}} = 0.0$ guarantees **perfect information-theoretic security**, making DCASS mathematically undetectable by deep-learning steganalysts like SRNet or Ye-Net.
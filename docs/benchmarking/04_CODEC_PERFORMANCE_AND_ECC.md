# Component 4: Codec Capacity, Throughput & Robustness
## DSSC Capacity, Sub-Millisecond Permutations & Adaptive RS-ECC Stress Testing

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Establish operational channel capacity, interactive encode latency, and algebraic error correction resilience under network loss  
> **Implementation Script:** `scripts/analysis/benchmark_4_codec_performance.py`

---

## 1. Codec Architecture: DSSC vs. exact_vcp

DCASS supports two primary coding modes:
1. **exact_vcp (Baseline Exact Voronoi Codebook Projection):**
   - Maps each 8-bit byte of payload into one of 256 Voronoi centroids.
   - Fixed capacity: exactly $8.0\text{ bits per carrier}$ ($1\text{ byte/carrier}$).
   - Requires 1 carrier per byte of Reed-Solomon codeword.
2. **DSSC (Dynamic Semantic State-Space Coding — Proposed Primary Codec):**
   - Decomposes the payload into semantic chunks.
   - For each chunk, determines admissible semantic families and extracts a candidate set $\mathcal{C}_i$ from the public corpus ($|\mathcal{C}_i| = N$).
   - Computes capacity:
     $$\eta_i = \max\left(2, \min\left(16, \lfloor \log_2 N \rfloor\right)\right) \text{ bits per carrier}$$
   - When $N \approx 4,000\text{--}35,000$, $\eta_i \approx 11\text{--}15\text{ bits/carrier}$.
   - **Carrier Reduction Ratio:**
     $$\Delta_{\text{reduction}} = 1 - \frac{N_{\text{DSSC}}}{N_{\text{exact\_vcp}}} \approx 35\%\text{--}46\%$$

---

## 2. Cryptographic PRNG Permutation Acceleration

In DSSC, the candidate set $\mathcal{C}_i$ is permuted using a private session key $K_{\text{session}}$:

$$\pi_i = \text{Permute}(|\mathcal{C}_i|, K_{\text{session}}, \text{context\_salt})$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PERMUTATION ACCELERATION EVOLUTION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ OLD: O(N) Pure-Python HMAC Loop:                                            │
│      for j in range(35000):                                                 │
│          h = HMAC(K, f"salt:{j}")                                           │
│      -> 35,000 Python calls/chunk = 250ms/chunk (7.5s for 30 carriers)      │
│                                                                             │
│ NEW: HMAC-DRBG via NumPy PCG64 (Tier-1 Enhancement):                        │
│      seed = int(HMAC(K, salt)[:16])                                         │
│      rng = np.random.default_rng(seed)                                      │
│      perm = rng.permutation(N)                                              │
│      -> Vectorized C execution = 0.3ms/chunk (2,500x speedup!)              │
│      -> Total message encode latency < 50ms (Interactive speed!)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

The benchmark tests:
* Determinism: $\pi_{\text{enc}} \equiv \pi_{\text{dec}}$ for identical key and salt.
* Key Sensitivity: Even a single-bit flip in $K_{\text{session}}$ produces a completely uncorrelated permutation ($\rho \approx 0.0$).
* Benchmark speed: 100 permutations of 35,000 items in $< 30\text{ms}$.

---

## 3. Adaptive Reed-Solomon Error Correction (RS-ECC)

### 3.1 Mathematical Foundation
Reed-Solomon is a non-binary cyclic error-correcting code defined over the Galois Field $GF(2^8)$ with primitive polynomial:
$$p(x) = x^8 + x^4 + x^3 + x^2 + 1 \quad (0\text{x}11D)$$

For a framed payload of length $L$ bytes and parity bytes $k$:
* Total Codeword Length: $N = L + k$ bytes.
* Maximum Correctable Byte Errors:
  $$t = \left\lfloor \frac{k}{2} \right\rfloor$$
* Maximum Correctable Erasures (known dropouts):
  $$\nu = k$$

### 3.2 Dynamic Adaptive Sizing Formula
Under Tier-1, DCASS dynamically scales parity bytes $k$ to maintain approximately $12\%$ overhead:
$$k = 2 \times \max\left(2, \min\left(127, \lceil 0.06 \times L \rceil\right)\right)$$

| Payload Size ($L$) | Parity Bytes ($k$) | Error Capacity ($t$) | Parity Overhead ($\%$) |
| :---: | :---: | :---: | :---: |
| 14 bytes (Short) | 8 bytes | 4 byte errors | $22.2\%$ |
| 50 bytes (Medium) | 8 bytes | 4 byte errors | $13.8\%$ |
| 150 bytes (Long) | 18 bytes | 9 byte errors | $10.7\%$ |
| 500 bytes (Document) | 60 bytes | 30 byte errors | $10.7\%$ |

### 3.3 Decoder Trial Resolution
Because Reed-Solomon parity is outside the frame, the decoder tests candidate parities $k \in \{4, 6, 8, \dots, 32\}$ using the HMAC-SHA256 frame tag as an exact cryptographic oracle. Any incorrect parity choice fails RS syndrome decoding or produces an invalid HMAC tag with probability $1 - 2^{-256}$.

---

## 4. Error Correction Stress Testing (Channel Erasures)

In real asynchronous covert channels (cloud drops, forums, IPFS), carriers can be lost, corrupted, or re-ordered in transit.

The benchmark simulates noisy channel conditions:
1. **Carrier Dropout / Erasure Test:**
   - Drops $5\%$, $10\%$, $15\%$, $20\%$ of transmitted carriers at random.
   - Evaluates whether Reed-Solomon error correction successfully recovers the 100% exact plaintext.
   - Plots Recovery Rate ($\%$) vs Dropout Rate ($\%$).
2. **Symbol Bit-Flip Test:**
   - Injects random byte corruption into carrier symbol values.
   - Verifies Berlekamp-Massey syndrome decoding up to capacity $t$.
3. **Cryptographic Key Separation Test:**
   - Encodes a message with key $K_A$.
   - Attempts decoding with 1,000 independent random keys $K_B \ne K_A$ and 1-bit mutated keys.
   - Verifies that:
     $$\text{Plaintext Recovery} = 0.000\%$$
     $$\text{False Acceptance Rate (FAR)} = 0.000\% \quad (< 2^{-256})$$

---

## 5. Output Artifacts Generated

The script outputs:
1. `paper_tables/table4_codec_performance.tex`:
   - Payload length vs carrier count (exact_vcp vs DSSC).
   - Bits per carrier ($\eta$).
   - Carrier reduction percentage ($\Delta_{\text{reduction}}$).
   - End-to-end encode and decode latency (ms).
2. `paper_plots/fig_ecc_recovery_curve.png`:
   - Plaintext recovery rate as a function of carrier loss percentage.

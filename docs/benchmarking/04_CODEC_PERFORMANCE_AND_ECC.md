# Component 4: Codec Capacity, Throughput & Robustness
## DSSC Capacity Advantage, Sub-Millisecond Permutations & RS-ECC Failure Anatomy

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Goal:** Quantify the DSSC capacity advantage over the exact_vcp baseline, measure keyed permutation throughput at production pool sizes, and decompose Reed-Solomon resilience into its exact end-to-end failure mechanisms.
> **Capacity note:** reported bits/carrier are **coded** (including framing and RS parity). User-payload capacity is 7.00 bpc at 14 B and 12.90 bpc at 500 B.
> **Implementation Scripts:** `scripts/analysis/benchmark_4a_capacity.py`, `scripts/analysis/benchmark_4b_robustness.py`  
> **Companion Script:** `scripts/analysis/benchmark_4c_ecc_failure_analysis.py`

**Rebuild note:** The first post-audit Component 4c run found that the legacy raw-payload fallback could accept wrong plaintext when ECC was enabled. The decoder now requires framed payloads under ECC, and legacy raw payloads remain decodable only when ECC is off. Section 4.5 records the finding, the fix, and the archived pre-hardening artifact.

---

## 1. Codec Architecture: DSSC vs. exact_vcp

DCASS ships two coding modes over the same corpus, the same framing layer, and the same Reed-Solomon parity.

1. **exact_vcp (baseline exact Voronoi codebook projection):**
   - Frames the payload and maps each byte of the resulting RS codeword onto one of 256 Voronoi cluster symbols, selecting one carrier per codeword byte.
   - Fixed capacity of exactly $8.0$ bits per carrier, and no keying unless a context manager is supplied.
2. **DSSC (dynamic semantic state-space coding, the primary codec):**
   - Frames the payload, applies RS parity, and reads the codeword as a continuous bitstream, emitting one carrier per symbol read.
   - At carrier position $i$, deterministic round robin picks the modality in the order Text, Image, Audio, and an HMAC-SHA256 digest of the session key and the position picks one of that modality's two native semantic families.
   - The candidate pool $\mathcal{C}_i$ is every corpus item whose Voronoi cluster belongs to the chosen family, with capacity:
     $$\eta_i = \max\left(2, \min\left(16, \lfloor \log_2 |\mathcal{C}_i| \rfloor\right)\right) \text{ bits per carrier}$$
   - A keyed session permutation of the pool maps each $\eta_i$-bit symbol onto a carrier media ID, and the decoder inverts the same construction.

The recorded "bits per carrier" is a codec-level average over the whole message, $8 \times \text{codeword bytes} / \text{carrier count}$, where the codeword is the framed payload plus parity. At the 50-byte payload used by Components 4b and 4c the codeword is 63 bytes, so each DSSC carrier averages $504 / 37 = 13.62$ bits.

### Recorded capacity, Component 4a, five trials per row, $k = 8$ pinned

| Payload (B) | exact_vcp carriers | DSSC multi-modal carriers | Reduction | DSSC bits/carrier |
| :---: | :---: | :---: | :---: | :---: |
| 14 | 27 | 16 | 40.7% | 13.50 |
| 50 | 63 | 37 | 41.3% | 13.62 |
| 150 | 163 | 96 | 41.1% | 13.58 |
| 300 | 321 | 188 | 41.4% | 13.66 |
| 500 | 529 | 310 | 41.4% | 13.65 |

* Multi-modal DSSC reduces the carrier count by 40.7 to 41.4 percent at every payload size, and the mean capacity is 13.602 bits per carrier against the fixed 8.0 of exact_vcp, a ratio of 1.70.
* The exact_vcp baseline in this record is the all-modality reference mapper. It spends one carrier per codeword byte, so its 8.00 bits per carrier and its 27 to 529 carrier counts do not depend on which modalities are requested. A strict text-only exact_vcp baseline is not available for arbitrary payloads: when a requested text cluster lacks a carrier for a byte symbol, the reference mapper falls back to image or audio. The benchmark therefore does not label that row as a text-only head-to-head.
* Text-only DSSC uses only text carriers and reaches 15.43 to 15.97 bits per carrier. Its carrier count is 48.1 to 49.9 percent below the one-carrier-per-codeword-byte exact_vcp count. That is a comparison to the exact_vcp rate bound, not to a text-only exact_vcp run.
* At the 300 and 500 byte payloads the framed codeword exceeds one GF(256) block, so the parity layer splits it into 255-byte blocks of 247 data bytes plus 8 parity. The recorded exact_vcp counts, 321 and 529, are the sums of those blocks, and DSSC packs the identical byte volume.
* The latency record is reference-implementation wall-clock time, not a codec-only speed comparison. exact_vcp encode costs 1.78 to 51.70 seconds per message because the shipped reference mapper scores semantic candidates for every codeword byte; the Voronoi assignment itself is precomputed. DSSC encode stays between 89 and 297 milliseconds at every size. Decode runs the other way: exact_vcp decodes in 2.2 to 40.7 milliseconds with a direct symbol lookup, while DSSC decode re-derives the candidate pools and resolves every carrier, costing 181 to 3,066 milliseconds. The capacity comparison does not depend on these latency columns, and the record makes no interactive-latency claim for DSSC decode.

---

## 2. Cryptographic Permutation Engine

The mapping from payload symbols to carriers is keyed per position:

$$\pi_i = \text{Permute}(|\mathcal{C}_i|, K_{\text{session}}, \text{context\_salt})$$

The implementation derives a seed with a single HMAC-SHA256 call and expands it with NumPy's vectorized PCG64 generator:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PERMUTATION ENGINE EVOLUTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ LEGACY: O(N) pure-Python HMAC loop over the candidate list:                 │
│      for j in range(N): h = HMAC(K, f"salt:{j}")                            │
│      -> roughly 250 ms per 35,000-candidate chunk                           │
│                                                                             │
│ CURRENT: HMAC-DRBG seed expanded by a vectorized NumPy PCG64 generator:     │
│      seed = HMAC-SHA256(K, "dssc:{chunk}:{family}")[:8]  (big-endian)       │
│      rng = np.random.default_rng(seed)                                      │
│      perm = rng.permutation(N)                                              │
│      -> 0.0575 ms at N = 1,000 rising to 0.4056 ms at N = 35,000            │
│      -> roughly 616x faster than the legacy loop at the top size            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recorded scaling, Component 4b, 100 permutations per size

| Candidate pool $N$ | Mean (ms) | p50 (ms) | Throughput (perm/s) | Determinism | Key sensitivity $\rho$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1,000 | 0.0575 | 0.0576 | 17,378 | pass | +0.043505 |
| 5,000 | 0.1673 | 0.1749 | 5,978 | pass | -0.016776 |
| 10,000 | 0.1575 | 0.1292 | 6,350 | pass | +0.004498 |
| 20,000 | 0.3013 | 0.3411 | 3,319 | pass | -0.002327 |
| 35,000 | 0.4056 | 0.3611 | 2,466 | pass | -0.001982 |

* **Determinism:** encode and decode derive byte-identical permutations from the same key and salt at every size.
* **Key sensitivity:** flipping bit 0 of byte 15 of the 32-byte key yields an uncorrelated permutation at every size. The Pearson correlation between the original and mutated index arrays stays inside $\pm 0.044$ with no trend, and at the two largest sizes it is inside $\pm 0.0024$.
* **Throughput:** the 35,000-candidate case has a mean of 0.4056 ms and a median of 0.3611 ms, about 616 times faster than the roughly 250 ms pure-Python loop it replaced. Mean latency stays below 0.18 ms through 10,000 candidates and is 0.41 ms at the largest tested size. The 20,000 row has a p50 slightly above its mean and the 35,000 row a p50 slightly below, so dispersion is small; the p50 column is the better central estimate under scheduler noise.

---

## 3. The Reed-Solomon Layer

### 3.1 Mathematical Foundation
Reed-Solomon is a non-binary cyclic code over the Galois Field $GF(2^8)$ with primitive polynomial:
$$p(x) = x^8 + x^4 + x^3 + x^2 + 1 \quad (0\text{x}11D)$$

For a codeword of $L$ data bytes and $k$ parity bytes:
* Maximum correctable byte errors at unknown positions: $t = \lfloor k / 2 \rfloor$.
* Maximum correctable erasures at known positions: $\nu = k$.
* The decoder never raises on uncorrectable input. `decode_bytes` returns the raw slice with `ok = False`, and the engine treats that as a failed message rather than an exception.

### 3.2 Adaptive Parity Ladder
The engine default scales parity with the framed length:
$$k = 2 \times \max\left(2, \min\left(127, \lceil 0.06 \times L \rceil\right)\right)$$

| Payload (B) | Framed length $L$ | Parity $k$ | Error capacity $t$ | Parity overhead |
| :---: | :---: | :---: | :---: | :---: |
| 14 | 19 | 4 | 2 | 21.1% |
| 50 | 55 | 8 | 4 | 14.5% |
| 150 | 155 | 20 | 10 | 12.9% |
| 300 | 305 | 38 | 19 | 12.5% |
| 500 | 505 | 62 | 31 | 12.3% |

The recorded campaigns pin $k = 8$ at every payload size, so the adaptive ladder is dormant in the record. Components 4b and 4c, which use the 50-byte payload, run on the single 63-byte codeword $RS(63,55)$.

### 3.3 Decoder Resolution and the Integrity Gate
The wire format carries neither the parity count $k$ nor per-carrier positions. The receiver supplies $k$, decodes with the matching RS configuration, and validates the result against the 5-byte CRC-16 frame, version 0x01. A wrong $k$ fails syndrome decoding or the CRC with probability about $1 - 2^{-16}$ per attempt, so the frame is an integrity gate, not an authenticator. The opt-in HMAC frame, version 0x02 with a 35-byte header, upgrades the gate to $2^{-256}$ when a shared secret is present. The recorded campaigns use the CRC path, and DSSC framing deliberately omits the HMAC tag so that the session key protects the carrier permutation rather than the frame bytes.

With ECC enabled, the decoder requires a framed payload and rejects an unframed result. Legacy raw payloads remain decodable only when ECC is off. Component 4c found that the earlier fallback could accept a deletion-shifted codeword as a raw message and report success with wrong plaintext. Section 4.5 records the finding and the hardening.

### 3.4 Recorded Algebraic Limits, Component 4c
The same $RS(63,55)$ configuration was exercised directly with known damage, 40 repeats per point:

* **Byte errors at unknown positions:** correct at 0 through 4 injected bytes, 0 percent at 5 and above, matching $t = 4$ exactly.
* **Known-position erasures:** correct at 0 through 8 injected bytes, 0 percent at 9 and above, matching $\nu = k = 8$ exactly.

---

## 4. End-to-End Failure Anatomy, Component 4c

Component 4b records a recovery curve that falls from 100 percent to 70 percent at 5 percent carrier substitution and collapses at 10 percent. Component 4c decomposes that shape mechanically: 80 trials per rate, 50-character payloads, 37 carriers, $k = 8$, $t = 4$, 1,120 trials in total. The number of damaged carriers per trial is $\text{round}(37 \times \text{rate})$ with positions chosen uniformly at random. A mirror instrumentation rebuilds the clean and received bitstreams byte for byte from the same public building blocks the decoder uses, guarded by a verification pass against the constructed codeword, and counts the damaged codeword bytes per trial. Intervals are Wilson 95 percent.

### 4.1 Substitution, Carriers Corrupted in Place

| Rate | Recovered | 95% CI | Mean damaged bytes | Per corrupted carrier |
| :---: | :---: | :---: | :---: | :---: |
| 0% | 80/80 | [95.4, 100.0] | 0.0 | 0.00 |
| 2% | 80/80 | [95.4, 100.0] | 2.2 | 2.17 |
| 3% | 80/80 | [95.4, 100.0] | 2.2 | 2.20 |
| 4% | 80/80 | [95.4, 100.0] | 2.1 | 2.10 |
| 5% | 49/80 | [50.3, 71.2] | 4.3 | 2.17 |
| 6% | 52/80 | [54.1, 74.5] | 4.4 | 2.19 |
| 8% | 0/80 | [0.0, 4.6] | 6.6 | 2.19 |
| 10% | 0/80 | [0.0, 4.6] | 8.6 | 2.16 |

* The substitution condition replaces a carrier with a different random corpus ID. Some replacements map to the same VCP symbol and leave the bitstream unchanged; the artifact records the share of replacements that actually changed the symbol. The recorded damage is measured from the resulting bitstream, so the curve describes random media replacement, not a worst-case forced symbol flip.
* One substituted carrier damages 2.1 to 2.2 codeword bytes on average, because an $\eta$-bit symbol straddles two or three byte boundaries.
* The recovery predicate is exact: in all 640 substitution trials, exact recovery occurred in precisely those trials whose damaged-byte count was at most $t = 4$. The separate acceptance count, which includes any CRC-passing decode even with the wrong plaintext, recorded no acceptance above the limit and no wrong plaintext in this condition. The CRC-16 gate rejected every Reed-Solomon miscorrection in the substitution arm, so the accepted count equals the exact recovery count in every row. The dropout arm exposed a separate legacy fallback; Section 4.5 records it.
* The cliff follows from the arithmetic. At 5 percent the harness corrupts exactly two of 37 carriers, and two damaged spans average 4.3 bytes against a 4-byte budget. Whether a trial recovers depends on whether the two spans fit, which is why the 5 and 6 percent points land near two thirds rather than at a sharp edge.

### 4.2 Dropout, Carriers Deleted

| Rate | Exact | Accepted | 95% CI | Mean damaged bytes |
| :---: | :---: | :---: | :---: | :---: |
| 0% | 80/80 | 80/80 | [95.4, 100.0] | 0.0 |
| 1% | 80/80 | 80/80 | [95.4, 100.0] | 0.0 |
| 2% | 0/80 | 0/80 | [0.0, 4.6] | 32.7 |
| 3% | 0/80 | 0/80 | [0.0, 4.6] | 32.9 |
| 4% | 0/80 | 0/80 | [0.0, 4.6] | 33.7 |
| 5% | 0/80 | 0/80 | [0.0, 4.6] | 44.5 |

* At 1 percent the rounded corrupted count is zero, so the trial is clean. From 2 percent on, every trial deletes one or two carriers and every trial fails.
* DSSC extracts symbols by positional enumeration of the received list, so a deletion early in the list shifts every following position and damages about 32.7 codeword bytes on average, eight times the error budget.
* A few dropout trials at 2, 3, and 4 percent have at most four damaged bytes yet still fail: one, four, and two of 80 trials respectively. These are deletions that shorten the received list rather than shift enough following carriers to exceed the byte budget. The decoder rejects the short codeword even though the byte-difference count is small, so damaged bytes are not a sufficient statistic for dropout.
* Post-hardening, the accepted count equals the exact count in every row. The accepted column is retained because it was the metric that exposed the pre-hardening fallback; Section 4.5 records that finding.
* Dropout is the destructive channel and substitution is the benign one. An adversary that deletes traffic is far more effective than one that replaces items, and the record says so plainly.

### 4.3 Erasure-Bound Replay
The measured damage counts can be replayed against the erasure limit: the share of substitution trials whose damaged bytes stay within $k = 8$ is 100 percent through 6 percent, 98.8 percent at 8 percent, and 51.2 percent at 10 percent. If carrier positions were known to the decoder, the same damage would be correctable about twice as far, roughly four corrupted carriers or 10 percent of carriers. The wire format carries no per-carrier sequence markers, so this headroom is latent rather than available, and it is the recorded basis for any future position-tracking protocol work.

### 4.4 Cross-Check Against Component 4b
The coarser 4b curve, 20 trials per rate on the same mechanism, records 100 percent at 0 percent, 70.0 percent at 5 percent, and 0 percent from 10 through 25 percent. The 80-trial 4c estimate at 5 percent, 61.2 percent with interval [50.3, 71.2], contains it, so the two campaigns agree within sampling error.

### 4.5 Integrity Finding and Decoder Hardening
The first rebuilt 4c run, before the decoder fix, recorded 41 CRC-passing wrong-plaintext decodes across the 320 dropout trials at 2 through 5 percent deletion: 7 at 2 percent, 7 at 3 percent, 10 at 4 percent, and 17 at 5 percent. The accepted shares were 8.8, 8.8, 12.5, and 21.2 percent while exact recovery was zero at every one of those rates. The decoded text did not match the payload, and the decoder reported success.

The cause was the legacy raw-payload fallback. A deletion moved the first byte away from the version marker, Reed-Solomon still returned a codeword, and `unframe_payload` treated the result as an unframed legacy message. Both `dssc_decoder.py` and `decoder.py` discarded the `was_framed` flag, so the fallback became a success path.

The decoder now requires a framed payload whenever ECC is enabled. An unframed result raises `FrameError`, and the decode fails closed. Legacy raw payloads remain decodable when ECC is off, which preserves the compatibility path without exposing the ECC-protected path. The archived pre-hardening artifact and table are stored under `storage/benchmarks/archive/` and record the 41 false accepts. The current 4c artifact and table record zero false accepts, and exact recovery is unchanged.

This finding changes the interpretation of the dropout result. Dropout is still unrecoverable in the current format, but it now fails closed rather than returning wrong plaintext. The accepted column in the current table equals the exact column, and the archived run is the evidence for the before-hardening behavior.

---

## 5. Cryptographic Key Separation

The key separation campaign encodes one message under a 32-byte authorized key, then attempts recovery with 100 independent random wrong keys and all 256 single-bit mutations of the authorized key.

| Test | Attempts | False acceptances | FAR 95% upper | Exact matches |
| :---: | :---: | :---: | :---: | :---: |
| Random wrong keys | 100 | 0 | 3.70% | 0 |
| Single-bit flipped keys | 256 | 0 | 1.48% | 0 |

* A false acceptance is any unauthorized decode for which the CRC gate returns success, whether or not the returned plaintext equals the original. The campaign recorded zero such decodes across 356 attempts and zero exact-message matches.
* The zero count is not a measurement of the $2^{-16}$ CRC bound. The Wilson 95 percent upper bounds are 3.70 percent at 100 random keys and 1.48 percent at 256 bit flips, so the campaign is consistent with the CRC-16 bound but cannot resolve it. The per-attempt bound remains theoretical: about $2^{-16}$ for the CRC path and about $2^{-256}$ for the opt-in HMAC frame, which these runs do not use.
* Rejection composes two mechanisms: a wrong key derives different families and different permutations, scrambling the symbol stream, and the CRC-16 frame rejects whatever survives. The keyed exact_vcp epoch search now also requires a frame whenever ECC or a frame secret is present, so a wrong epoch cannot fall back to a raw payload.
* The permutation-level counterpart is the avalanche record in Section 2: a one-bit key change decorrelates the carrier ordering at every pool size.

---

## 6. Output Artifacts Generated

The component writes:
1. `storage/benchmarks/component_4a_capacity_results.json`, `storage/benchmarks/tables/table4a_capacity.tex`, and `storage/benchmarks/plots/fig_4a_capacity_comparison.png`: carriers, bits per carrier, reduction, and latency for three modes across five payload sizes.
2. `storage/benchmarks/component_4b_robustness_results.json`, `storage/benchmarks/tables/table4b_codec_robustness.tex`, and `storage/benchmarks/plots/fig_4b_permutation_scaling.png`, `storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png`, `storage/benchmarks/plots/fig_4b_key_separation.png`: permutation scaling, the substitution recovery curve, and key separation.
3. `storage/benchmarks/component_4c_ecc_failure_results.json`, `storage/benchmarks/tables/table4c_ecc_failure_modes.tex`, and `storage/benchmarks/plots/fig_4c_ecc_failure_modes.png`: substitution and dropout anatomy with per-trial damage samples, the erasure-bound replay, and the algebraic ladder.
4. `storage/benchmarks/archive/`: the pre-hardening Component 4c artifact, table, and plot, plus a README that records the 41 false accepts and the decoder change that removed them.

Reproduction:
* `.venv/bin/python scripts/analysis/benchmark_4a_capacity.py`, or add `--artifacts-only` to rebuild the table and plot from the recorded JSON without re-running measurements.
* `.venv/bin/python scripts/analysis/benchmark_4b_robustness.py`, or add `--regenerate-from-json` to rebuild the table and plots from the recorded JSON without re-running measurements.
* `.venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py`, or add `--replot` to rebuild from JSON, or `--quick` for a smoke run.

All three scripts seed every random generator with `SEED = 42`. The recorded artifacts carry a schema version and a live provenance block with the git commit, timestamp, Python and torch versions, device, platform, command, seed, and index status. The key-separation acceptance metric counts any CRC-passing decode under an unauthorized key, not only an exact plaintext match.

# HMAC-SHA256 Keyed Derivation in DCASS
## Mathematical Specification & Zero-Overhead Synchronization

---

### 1. What Is It?

In DCASS (DSSC architecture), **HMAC-SHA256** (Hash-based Message Authentication Code with SHA-256) is used as a **Cryptographic Pseudorandom Function (PRF)**.

Instead of sending configuration data, domain choices, or permutation tables across the network, Alice and Bob share a single **256-bit Session Key ($K_{\text{session}}$)**. Both parties use HMAC-SHA256 to independently derive the exact same decisions locally with **0 bytes transmitted**.

```
                HMAC-SHA256 ZERO-OVERHEAD SYNCHRONIZATION

        Alice (Sender)                             Bob (Receiver)
   ┌───────────────────────┐                 ┌───────────────────────┐
   │ Shared 32-Byte Key K  │                 │ Shared 32-Byte Key K  │
   └──────────┬────────────┘                 └──────────┬────────────┘
              │                                         │
              ▼                                         ▼
   HMAC(K, "family:0") mod 6                 HMAC(K, "family:0") mod 6
   = Family 0 (nature_outdoor)               = Family 0 (nature_outdoor)
              │                                         │
              ▼                                         ▼
   HMAC(K, "dssc:0:cand:i")                  HMAC(K, "dssc:0:cand:i")
   = Shuffled Permutation π                  = Shuffled Permutation π
              │                                         │
              ▼                                         ▼
    Alice selects Carrier                      Bob inverts Carrier
       Media ID at π[S]                           at π[S] → Extracts S
```

---

### 2. The Formal Mathematical Definition

Given:
* **Key $K$:** 256-bit secret key ($32\text{ bytes}$).
* **Message $M$:** Structured context salt (ASCII string).
* **Inner pad ($\text{ipad}$):** Byte `0x36` repeated 64 times.
* **Outer pad ($\text{opad}$):** Byte `0x5C` repeated 64 times.

$$\text{HMAC-SHA256}(K, M) = \text{SHA256}\Big((K \oplus \text{opad}) \parallel \text{SHA256}\big((K \oplus \text{ipad}) \parallel M\big)\Big)$$

The output is a 256-bit ($32\text{-byte}$) pseudo-random cryptographic digest.

---

### 3. Two Core Functions Powered by HMAC in DCASS

#### Function A: Deterministic Semantic Domain Routing
For carrier chunk index $i \in \{0, \dots, 14\}$:
1. Compute digest:
   $$\mathbf{D} = \text{HMAC-SHA256}\big(K_{\text{session}}, \text{"family:"} \parallel i\big)$$
2. Extract first 4 bytes as a big-endian 32-bit unsigned integer:
   $$R = \text{int.from\_bytes}(\mathbf{D}[0:4], \text{"big"})$$
3. Modulo by the 6 semantic macro-families:
   $$\text{family\_idx} = R \pmod 6$$

**Source Code Reference:** [`src/engine/dssc_encoder.py:L157-L163`](file:///home/jeevan/projects/DCASS/src/engine/dssc_encoder.py#L157-L163)

---

#### Function B: State-Space Fisher-Yates Permutation
To securely shuffle the $N = 32,768$ candidates within a semantic family:
1. For each candidate index $j \in \{0, \dots, N-1\}$:
   $$\text{Weight}_j = \text{int.from\_bytes}\Big(\text{HMAC-SHA256}\big(K, \text{"dssc:"} \parallel i \parallel \text{family} \parallel \text{":"} \parallel j\big)[0:8], \text{"big"}\Big)$$
2. Sort candidate indices by their 64-bit cryptographic weights:
   $$\pi = \text{argsort}(\mathbf{Weight})$$
3. Mapping:
   $$\text{Transmitted Carrier} = \text{Candidates}[\pi[S_i]]$$
   where $S_i$ is the secret 15-bit integer payload symbol.

**Source Code Reference:** [`src/engine/dssc_state_space.py:L93-L109`](file:///home/jeevan/projects/DCASS/src/engine/dssc_state_space.py#L93-L109)

---

### 4. Why This Eliminates Transmission Overhead & Topic Leakage

1. **Zero Over-The-Air Payload:**
   Because HMAC-SHA256 is deterministic, Alice and Bob never transmit domain tags, permutation indexes, or headers over the wire. The wire contains **only unmodified public media IDs**.
2. **Cryptographic Avalanche Effect:**
   Changing even 1 single bit in the 256-bit session key flips ~50% of the bits in the digest, producing a completely unrelated permutation.
3. **Impossibility of Brute Force:**
   The permutation space contains $N! = 32,768! \approx 10^{134,800}$ possible orderings. Without the 256-bit key, an eavesdropper cannot map any media item to its payload symbol.

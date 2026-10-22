# DCASS Research & Technical Progress Report: Multi-Modal Steganography & Vector Infrastructure Milestone

## 1. Executive Summary & Capstone Goals

### 1.1 Executive Summary
Dynamic Context-Aware Semantic Steganography (DCASS) is an advanced multi-modal steganography framework designed to encode covert secret messages into dynamic, natural, multi-modal carrier media streams (images, text, audio) without altering the physical underlying bytes or feature signals of the carrier media itself. By leveraging unified vector spaces (CLIP, Sentence-Transformers, CLAP) indexed with Facebook AI Similarity Search (FAISS), DCASS maps secret message payloads into nearest-neighbor semantic carrier selections that are intended to resemble normal user activity. Selection and timing are evaluated separately; the recorded C3 traffic result is negative/limited, not demonstrated indistinguishability.

### 1.2 Capstone Goals & Objectives
1. **Multi-Modal Semantic Indexing**: Construct scalable FAISS vector databases across image, text, and audio corpora totaling over 150,000 vectors and 170,000 raw media assets.
2. **GPU-Accelerated Feature Extraction & Retrieval**: Eliminate CPU embedding extraction bottlenecks by optimizing batching on dedicated GPU hardware (NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.1).
3. **Bounded-Error Decoding**: Reduce the 70–85% accuracy plateau of continuous vector quantization drift with Reed-Solomon Error Correcting Codes (RS-ECC) over $GF(2^8)$. The recorded result is 0% BER on clean channels and within the $t=4$ byte budget; 5% substitution drops exact recovery to 61.2–70%, and carrier dropout is unrecoverable in the current wire format.
4. **Context-Aware Dynamic Keying & Stealth Scheduling**: Synthesize dynamic public context feeds and GAN-based behavioral scheduling to attempt to reduce traffic-analysis signal. A trained traffic detector flags the WGAN and Hybrid schedules at 100% against a threshold calibrated on held-out human traffic, and the retrained PPO arm is not flagged; the human side of that comparison is a synthetic corpus, so evasion is not demonstrated.

---

## 2. Complete Multi-Modal Corpus Breakdown

### 2.1 Corpus Statistics & Quantitative Metrics
The DCASS framework integrates three heterogeneous media modalities to support diverse steganographic carrier channels. The indexing architecture transforms raw media items into normalized 512-dimensional vector embeddings stored in unified FAISS indices.

| Modality | Datasets / Sources | Raw Media Items | FAISS Vectors | Embedding Model | Vector Dimension |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Image Channel** | Flickr30k + Flickr8k | 31,783 `.jpg` images | 31,783 vectors | OpenAI CLIP (`ViT-B/32`) | **512** |
| **Text Channel** | Wikipedia + Captions | 200,000 raw sentences | 200,000 vectors | OpenAI CLIP Text Encoder (`ViT-B/32`) | **512** |
| **Audio Channel** | `libretta` TTS | 13,496 `.wav` clips | 13,496 vectors | LAION CLAP (`clap-htsat-unfused`) | **512** |
| **Total System Volume** | **Multi-Modal Integration** | **245,279 media items** | **245,279 FAISS vectors** | **Unified CLIP/CLAP Stack** | **512 (Unified)** |

### 2.2 Ingestion & FAISS Vector Pipeline Diagram
```mermaid
flowchart TD
    subgraph Raw_Corpora["Raw Media Storage (245,279 Indexed Items)"]
        IMG["Flickr30k + Flickr8k\n(31,783 .jpg Images)"]
        TXT["Wikipedia + Captions\n(200,000 Sentences)"]
        AUD["Libretta TTS\n(13,496 .wav Clips)"]
    end

    subgraph GPU_Extractors["GPU Feature Extraction Pipeline (CUDA 12.1)"]
        CLIP_IMG["CLIP ViT-B/32 Image Encoder\n(~350 images/sec)"]
        CLIP_TXT["CLIP ViT-B/32 Text Encoder\n(~1,500 sentences/sec)"]
        CLAP_AUD["LAION CLAP Audio Encoder\n(~80 clips/sec)"]
    end

    subgraph Vector_DB["FAISS Unified Multi-Modal Index (245,279 Vectors)"]
        IDX_IMG["Image Index\n(31,783 Vectors, 512-d)"]
        IDX_TXT["Text Index\n(200,000 Vectors, 512-d)"]
        IDX_AUD["Audio Index\n(13,496 Vectors, 512-d)"]
    end

    IMG --> CLIP_IMG --> IDX_IMG
    TXT --> CLIP_TXT --> IDX_TXT
    AUD --> CLAP_AUD --> IDX_AUD
```

---

## 3. GPU Acceleration Benchmarks

#### Quantitative Throughput Performance

| Pipeline Stage | CPU Baseline Throughput |  GPU Throughput | Acceleration Factor |
| :--- | :---: | :---: | :---: |
| **Image Ingestion (CLIP ViT-B/32)** | ~12 images/sec | **~350 images/sec** | **29.1x Speedup** |
| **Text Ingestion (CLIP Text Encoder)** | ~65 sentences/sec | **~1,500 sentences/sec** | **23.1x Speedup** |
| **FAISS K-NN Nearest Search** | 1.8 ms/query | **0.06 ms/query** | **30.0x Speedup** |
| **Total Ingestion Time** | ~36.5 hours | **~18.4 minutes** | **~119x Overall Efficiency** |

---

## 4. Accuracy Bottleneck & Reed-Solomon ECC Solution

### 4.1 The 70-85% Decoding Accuracy Plateau
In semantic steganography, secret message bytes are mapped into discrete indices in vector space. During decoding, receiver nodes attempt to reconstruct the original message by matching observed carrier media back to vector index embeddings.

However, empirical experiments revealed a persistent **70-85% raw retrieval accuracy plateau** (15-25% Bit Error Rate / Character Error Rate).

#### Causes of Vector Noise & Drift:
1. **Continuous Floating-Point Precision Noise**: Minor floating-point variance ($\Delta \epsilon \approx 10^{-6}$) across heterogeneous PyTorch / CUDA precision execution.
2. **Quantization & Nearest-Neighbor Ambiguity**: In high-dimensional semantic vector spaces, multiple semantically related carrier items cluster tightly together. Small perturbations cause nearest-neighbor search to return an adjacent vector index rather than the exact target index.
3. **Lossy Compression / Transmission Perturbations**: Re-encoding JPEG images or text tokenization introduces semantic vector drift.

### 4.2 Mathematical Mechanics of Reed-Solomon ECC

To eliminate decoding errors without modifying carrier media or introducing detectable statistical artifacts, DCASS integrates **Reed-Solomon Error Correcting Codes (RS-ECC)** operating over Galois Field $GF(2^8)$.

#### Theoretical Formulation
Let a secret message chunk be represented as a message polynomial $M(x)$ of degree $k-1$ with coefficients in $GF(2^8)$:

$$M(x) = m_{k-1} x^{k-1} + m_{k-2} x^{k-2} + \dots + m_1 x + m_0$$

The Reed-Solomon encoder multiplies $M(x)$ by $x^{2t}$ and computes the remainder modulo a generator polynomial $G(x)$ of degree $2t = n - k$:

$$P(x) = M(x) \cdot x^{2t} \pmod{G(x)}$$

The transmitted codeword polynomial $C(x)$ of length $n$ bytes is:

$$C(x) = M(x) \cdot x^{2t} + P(x)$$

Where:
- $n$: Total block size in bytes (e.g., $n = 255$).
- $k$: Original payload length in bytes.
- $2t = n - k$: Added parity bytes.
- $t$: Maximum number of byte errors correctable in any arbitrary location:

$$t = \left\lfloor \frac{n - k}{2} \right\rfloor$$

#### Clean-Channel 0% BER Within the Correction Budget
By appending $2t$ parity bytes prior to semantic embedding lookup, the RS Berlekamp-Massey decoder reconstructs $M(x)$ exactly while the damaged-byte count stays at or below $t=4$. The recorded clean-channel campaign reaches **0% BER**. This is not a guarantee for a 15–20% byte-error channel or for carrier dropout: 5% carrier substitution drops exact recovery to 61.2–70%, and deletion is unrecoverable in the current wire format. Parity bytes are encoded as standard semantic carriers.

```mermaid
sequenceDiagram
    autonumber
    participant Sender as Sender System
    participant RS as Reed-Solomon Encoder GF(2^8)
    participant FAISS as FAISS Multi-Modal Vector DB
    participant Channel as Communication Channel (Public Media Stream)
    participant Receiver as Receiver System

    Sender->>RS: Raw Secret Message Payload M(x)
    RS->>RS: Compute Parity Bytes P(x) = M(x)·x^(2t) mod G(x)
    RS-->>Sender: Codeword Bytes C(x) [Data + Parity]
    Sender->>FAISS: Vector Nearest Neighbor Lookup for C(x)
    FAISS-->>Sender: Selected Carrier Media Items (Images/Text/Audio)
    Sender->>Channel: Transmit Steganographic Media Sequence
    Note over Channel: Continuous Vector Noise / Drift (15-20% Byte Errors)
    Channel->>Receiver: Received Carrier Media Sequence
    Receiver->>FAISS: Extract Vector Indices -> Erroneous Codeword C'(x)
    Receiver->>RS: Decode C'(x) via Berlekamp-Massey Algorithm
    RS-->>Receiver: Bit-exact payload within t=4 [clean-channel BER 0%]
```

---
# DCASS: Dynamic Context-Aware Semantic Steganography

Zero-modification covert communication framework that transmits messages by curating naturally occurring public media and scheduling transmissions with human-like behavioral models.

- **Zero-Modification Steganography**: Nullify neural content steganalysis ($D_{\text{KL}} = 0.0$) by eliminating physical media manipulation.
- **100% Recovery via RS-ECC**: Overcome Voronoi cell boundary quantization drift using Reed-Solomon $GF(2^8)$ error-correcting codes.
- **Dynamic Context Keying**: Prevent static media-to-symbol correlation through synchronized temporal epochs and decoy topics.
- **Behavioral Stealth Scheduling**: Defeat Deep Packet Inspection traffic analysis using PPO Reinforcement Learning and WGAN-GP temporal models.
- **Multi-Modal and Multi-Channel**: Support image, text, and audio carriers distributed across independent communication channels.

## Overview

DCASS (Dynamic Context-Aware Semantic Steganography) is an academic research and engineering prototype for zero-modification information hiding. Instead of embedding secret bits into the pixels, frequency coefficients, or audio waveforms of a carrier file, DCASS transmits covert messages by selecting and sequencing naturally occurring, unmodified media items (text, images, and audio) from public corpora and distributing them according to human behavioral schedules.

## The Problem DCASS Solves

### Physical Steganalysis Vulnerability
Traditional steganographic techniques (such as Least Significant Bit insertion, discrete cosine transform coefficient shifts, or audio sample manipulation) alter the physical bitstream of cover media. Even when perturbations are invisible to the human eye or ear, convolutional neural networks and vision transformers (such as SRNet, Zhu-Net, and Ye-Net) detect the residual noise signatures introduced by these modifications.

### Zero-Modification Formulation
DCASS eliminates physical media manipulation entirely. Carrier items are pulled directly from public datasets (Flickr8k, Flickr30k, Wikipedia) without altering a single byte:

$$P_{\text{stego}}(X) = P_{\text{cover}}(X)$$

The Kullback-Leibler divergence between cover and stego media is zero:

$$D_{\text{KL}}(P_{\text{cover}} \parallel P_{\text{stego}}) = 0.0$$

Under Cachin's information-theoretic steganography model, a system with zero relative entropy against cover distributions is secure against content-based steganalysis detectors.

### Traffic Analysis Defense
Covert systems often fail against network perimeter monitors ("Wardens" or Deep Packet Inspection appliances) that record transmission metadata:
- Inter-transmission intervals ($\Delta t_i = t_i - t_{i-1}$)
- Transmission burstiness and timing variance
- Channel switching frequencies across egress platforms
- Diurnal human sleep and wake cycles

If transmissions use fixed delays or uniform random pauses, fast Fourier transform (FFT) analysis and Kolmogorov-Smirnov tests detect an automated bot. DCASS mitigates this using multi-tier behavioral scheduling.

## Core Mechanisms and Architecture

The DCASS pipeline consists of five stages:

```
Secret Message
    │
    ▼
[Semantic Chunker & RS-ECC GF(2^8)]
    │
    ▼
[Vector Hypersphere Lookup via FAISS (CLIP / CLAP 512-d)]
    │
    ▼
[Dynamic Context Key Derivation]
    │
    ▼
[3-Tier Stealth Scheduler: RL PPO -> WGAN-GP -> NoiseController]
    │
    ▼
[Multi-Channel Dispatcher (Console / Local Folder / Shared Channels)]
    │
    ▼
[Receiver: Ingestion -> Inverted Voronoi Codebook -> RS-ECC Berlekamp-Massey -> Plaintext]
```

### 1. Vector Hypersphere Representation
All modalities are embedded into a unified 512-dimensional Euclidean space and normalized onto the unit hypersphere $\mathbb{S}^{511}$:

$$\mathbb{S}^{511} = \{ v \in \mathbb{R}^{512} : \|v\|_2 = 1.0 \}$$

- **Images**: Embedded via CLIP ViT-B/32 ([`ImageEmbedder`](file:///home/img1/projects/DCASS/src/corpus/embedders/image_embedder.py#L22-L95)).
- **Text**: Embedded via CLIP text transformer ([`BaseEmbedder`](file:///home/img1/projects/DCASS/src/corpus/embedders/base_embedder.py#L20-L80)).
- **Audio**: Embedded via LAION CLAP ([`AudioEmbedder`](file:///home/img1/projects/DCASS/src/corpus/embedders/audio_embedder.py#L20-L100)).
- **Vector Search**: Indexed using FAISS inner-product search, equivalent to cosine similarity on unit vectors ([`UnifiedSemanticIndex`](file:///home/img1/projects/DCASS/src/corpus/index/unified_index.py#L25-L160)).

### 2. Voronoi Quantization and Reed-Solomon Error Correction
When continuous vector spaces are partitioned into discrete Voronoi cells around $N$ indexed corpus items, floating-point precision shifts and dense semantic boundaries introduce a 15% to 25% symbol error rate during nearest-neighbor retrieval.

To achieve exact reconstruction without altering media, DCASS uses Reed-Solomon error-correcting codes over Galois Field $GF(2^8)$ with primitive polynomial $p(x) = x^8 + x^4 + x^3 + x^2 + 1$ (0x11D):
- The payload is split into chunks and framed with length, CRC-16, and parity symbols ([`RSErrorCorrection`](file:///home/img1/projects/DCASS/src/engine/ecc.py#L18-L165) and [`frame_payload`](file:///home/img1/projects/DCASS/src/engine/payload_framing.py#L15-L80)).
- Adding $R = 2t$ parity symbols allows the receiver to correct up to $t$ corrupted symbols per codeword using the Berlekamp-Massey algorithm, Chien search, and Forney algorithm.
- This resolves the Voronoi drift plateau, yielding 0% Bit Error Rate (100% exact message recovery).

### 3. Dynamic Context Keys
To prevent an adversary from constructing a static dictionary of media-to-symbol associations, DCASS uses dynamic context keys:
- Keys are derived from synchronized epoch timestamps, decoy cover topics, or shared salt ([`KeyManager`](file:///home/img1/projects/DCASS/src/engine/context/key_manager.py#L20-L120)).
- Context transformations shift target vector coordinates before corpus retrieval, ensuring the same secret message selects different media sequences across different sessions.

### 4. Three-Tier Stealth Scheduling
To defeat traffic-analysis wardens, transmission intervals and channel selection are handled by a three-tier hierarchy ([`StealthScheduler`](file:///home/img1/projects/DCASS/src/stealth/stealth_scheduler.py#L26-L160)):
1. **PPO Reinforcement Learning Agent (Closed-Loop Adaptive)**: Observes a 16-dimensional state vector (outbound queue ratio, time-of-day diurnal sin/cos encoding, channel congestion, rolling warden suspicion score) and selects continuous inter-packet delays and channel assignments ([`PPOAgent`](file:///home/img1/projects/DCASS/src/stealth/rl/agent.py#L30-L210) and [`StealthTransmissionEnv`](file:///home/img1/projects/DCASS/src/stealth/rl/environment.py#L25-L180)).
2. **WGAN-GP Temporal Generator (Open-Loop Generative)**: A Wasserstein GAN with Gradient Penalty trained on empirical human social media timestamps. Generates bursty delays matching natural 1/f pink noise distributions ([`TemporalPatternGenerator`](file:///home/img1/projects/DCASS/src/stealth/gan/generator.py#L18-L95)).
3. **NoiseController (Deterministic Fallback)**: Uses Poisson processes and Pareto distributions calibrated against behavioral profiles (casual, power-user, night-owl) when machine learning checkpoints are absent ([`NoiseController`](file:///home/img1/projects/DCASS/src/distribution/noise.py#L20-L110)).

### 5. Multi-Channel Distribution and Reassembly
Packets containing sequence metadata and media IDs are routed across pluggable channels ([`ChannelDispatcher`](file:///home/img1/projects/DCASS/src/distribution/dispatcher.py#L20-L120)):
- Console simulation ([`ConsoleChannel`](file:///home/img1/projects/DCASS/src/distribution/console_channel.py#L12-L45))
- Local file-system directories simulating shared folders ([`LocalFolderChannel`](file:///home/img1/projects/DCASS/src/distribution/local_folder_channel.py#L15-L60))
- Simulated social/cloud endpoints
- The receiver ingests the packets, extracts the media embeddings from the shared corpus, maps them through the Voronoi codebook, executes Reed-Solomon decoding, and restores the original message ([`SemanticDecoder`](file:///home/img1/projects/DCASS/src/engine/decoder.py#L25-L180)).

## Key Technical Specifications

| Parameter | Specification |
|---|---|
| Embedding Dimensions | 512 ($d = 512$, unit normalized on $\mathbb{S}^{511}$) |
| Vision / Text Models | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Audio Model | LAION CLAP (`laion/clap-htsat-unfused`) |
| Vector Index | FAISS `IndexFlatIP` |
| Error Correction | Reed-Solomon $GF(2^8)$, configurable parity ($R \in [4, 16]$) |
| Stealth Engine | Custom PyTorch WGAN-GP and custom Actor-Critic PPO (zero SB3 dependency) |
| Backend API | FastAPI (`src/api/server.py`) with JWT authentication |
| Frontend | Next.js 14, React, Tailwind CSS (`frontend/`) |
| Corpus Support | Flickr8k, Flickr30k, Wikipedia image/text, Google Drive corpus sync |

## Repository Organization

- [`src/corpus/`](file:///home/img1/projects/DCASS/src/corpus/): Dataset ingestion, CLIP/CLAP feature extractors, Voronoi codebook generation, FAISS indexing.
- [`src/engine/`](file:///home/img1/projects/DCASS/src/engine/): Semantic chunker, Reed-Solomon error correction, context key manager, encoder, and decoder.
- [`src/stealth/`](file:///home/img1/projects/DCASS/src/stealth/): WGAN-GP delay generator, PPO reinforcement learning agent, and stealth pipeline.
- [`src/distribution/`](file:///home/img1/projects/DCASS/src/distribution/): Traffic scheduler, behavioral profiles, noise controller, and channel dispatcher.
- [`src/analysis/`](file:///home/img1/projects/DCASS/src/analysis/): Adversarial Warden detector, bit-error rate calculations, and benchmark suites.
- [`src/api/`](file:///home/img1/projects/DCASS/src/api/): FastAPI REST interface with token authentication and live transmission tracking.
- [`frontend/`](file:///home/img1/projects/DCASS/frontend/): Next.js web application for encoding, decoding, and monitoring traffic schedules.
- [`scripts/`](file:///home/img1/projects/DCASS/scripts/): Dataset builders, training routines for GAN and RL models, demos, and validation tools.
- [`docs/`](file:///home/img1/projects/DCASS/docs/): Mathematical proofs, architecture diagrams, user guides, and research specifications.

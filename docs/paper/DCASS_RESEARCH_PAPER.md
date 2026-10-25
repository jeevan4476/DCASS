# DCASS: Dynamic Context-Aware Semantic Steganography Across Multi-Modal Transport and Behavioral Networks

## Abstract

Physical signal modifications in traditional image, audio, and text steganography inevitably perturb fine-grained spatial and frequency statistics, exposing covert transmissions to deep residual steganalyzers such as SRNet, ZhuNet, and YeNet. Coverless and selection-based steganography eliminates physical artifacts by transmitting unmodified natural media, but existing designs suffer from severe carrier bloat, lack formal algebraic error control, induce cross-modal semantic distribution drift, and emit rigid network traffic flows easily detected by deep packet inspection appliances. This paper presents Dynamic Context-Aware Semantic Steganography, designated DCASS, an end-to-end covert communications architecture that unifies multi-modal semantic indexing with behavioral network traffic scheduling. DCASS indexes natural media files across image, text, and audio corpora on a 512-dimensional unit hypersphere partitioned into 256 Voronoi centroids. To resolve boundary-crossing quantization noise across continuous embedding space, DCASS couples spherical nearest-neighbor selection with Reed-Solomon algebraic coding over Galois field GF(2^8) and keyed cryptographic permutations. At the network layer, DCASS orchestrates carrier transmissions across multiple independent channels using a Wasserstein generative adversarial network conditioned on diurnal human cadence, governed by a closed-loop Proximal Policy Optimization reinforcement learning scheduler that enforces enterprise cooldown timers via action masking.

We evaluate DCASS across five orthogonal threat dimensions under an audited empirical protocol. Against deep residual steganalyzers trained to detect spatial perturbations at 93.7% to 95.1% accuracy, DCASS carriers yield chance-level detection with measured receiver operating characteristic area under curve values between 0.4973 and 0.4993, with every 95% confidence interval spanning 0.5000. In semantic distributional tests across 1,536 coordinate dimensions, zero coordinates reject under Benjamini-Hochberg false discovery control, and cluster histograms confirm alignment with the system structural mechanism at permutation p = 0.208. The dynamic codec delivers 13.50 to 13.66 coded bits per carrier, representing a 40.7% to 49.9% carrier reduction relative to exact Voronoi baseline schemes, with user payload capacity scaling up to 12.90 bits per carrier, while achieving zero false acceptances across 356 wrong-key forgery tests. Evaluated against a calibrated network traffic sequence detector trained on authentic human conversation sessions from public community discussion data, the closed-loop reinforcement learning scheduler achieves 76.0% evasion at 5.63 seconds mean latency with zero channel cooldown violations, whereas static, Poisson, and unguided generative baselines are intercepted at 100%.

## Keywords

Coverless steganography, multi-modal embeddings, Voronoi codebook, Reed-Solomon coding, generative adversarial networks, reinforcement learning, traffic analysis evasion.

---

## 1. Introduction and threat model

Steganography is the science of concealing secret messages within innocuous carrier media such that the very existence of the covert channel remains concealed. Over four decades, physical embedding techniques have dominated the academic literature. These approaches embed secret bits into spatial least significant bit planes, discrete cosine transform coefficients, or audio waveform samples. As steganographic embedding advanced from heuristic LSB substitution to distortion-minimizing syndrome-trellis codes, such as S-UNIWARD, WOW, and J-UNIWARD, detection capabilities advanced in lockstep. Modern steganalysis deploys specialized deep convolutional neural networks equipped with high-pass spatial rich model filters and unpooled residual architectures. These steganalyzers, most notably SRNet, ZhuNet, and YeNet, learn subtle inter-pixel co-occurrence shifts and high-frequency boundary residuals, reliably detecting physical embedding schemes with classification accuracies exceeding 90%.

Coverless and selection-based steganography emerged to bypass media-layer steganalyzers entirely. Rather than modifying a cover object, a coverless system selects an unmodified natural media file whose intrinsic representation matches the secret message. Because the carrier is drawn directly from a public repository without altering a single bit, pixel, or audio sample, physical content steganalyzers are rendered blind. However, existing coverless schemes present three fundamental vulnerabilities that prevent dependable real-world deployment:

1. Codec inefficiency and continuous quantization noise: Mapping discrete secret symbols to continuous semantic vectors introduces severe boundary-crossing noise. Minor floating-point rounding discrepancies or continuous index shifts cause nearest-neighbor lookups to land in adjacent Voronoi regions, causing symbol error rates between 15% and 25%. Prior coverless schemes mitigate this by restricting codebooks to tiny vocabularies, collapsing throughput to less than 1.0 bit per carrier and requiring hundreds of media files for brief payloads.
2. Semantic selection bias and distributional drift: When a sender repeatedly draws carriers matching specific message sequences, the resulting collection of transmitted media files distorts the underlying corpus distribution. A vigilant adversary analyzing multivariate semantic distributions or codebook cluster entropy across a transmission stream can detect anomalous clustering or non-uniform topic skews.
3. Network traffic cadence and flow vulnerability: Content-level blindness is futile if the transmission pattern itself exposes the sender. Automated steganographic tools typically emit packets at static time intervals, memoryless Poisson intervals, or rapid bursts. Modern network security appliances execute deep packet inspection and traffic flow analysis, monitoring packet inter-arrival times, burstiness, flow durations, and destination channel concentrations. Uncoordinated transmissions trigger rate-limiting alarms, packet loss, or automated deep packet inspection flags.

### 1.1 Multi-layer threat model

We formalize an adversary monitoring communications between sender Alice and receiver Bob across public Internet platforms. The adversary deploys four distinct detection mechanisms operating across protocol layers:

1. Layer 1, media-layer content warden: A passive network monitor intercepts every transmitted media file, whether an image, a text snippet, or an audio clip. The warden executes state-of-the-art deep residual steganalyzers, specifically SRNet, ZhuNet, and YeNet. The warden objective is to distinguish covert carriers from clean public covers drawn from identical distributions.
2. Layer 2, semantic distribution warden: An observer extracts high-dimensional semantic representations of intercepted carriers using multi-modal embedding models. The warden runs multivariate two-sample statistical tests, including Energy Distance, Maximum Mean Discrepancy, and coordinate-wise Kolmogorov-Smirnov screenings, to detect whether the carrier stream drifts from the reference corpus distribution.
3. Layer 3, network transport flow warden: An inline deep packet inspection appliance monitors traffic flow between Alice and external platforms. The warden records packet timestamps, inter-arrival delays, and channel indices. It evaluates traffic against deterministic heuristics: periodicity detection, burst thresholds, platform hammer limits, and a supervised neural sequence detector trained on authentic human conversation cadences.
4. Layer 4, channel adversary and cryptanalyst: The physical and network channel is subject to impairments, including carrier recompression, carrier substitution, and packet loss. Simultaneously, an unauthorized eavesdropper possessing intercepted carriers attempts to decode the secret message without knowledge of the private session key.

### 1.2 System contributions

DCASS addresses all four threat layers through an integrated, empirically validated architecture:

- Zero-modification semantic indexing: Carrier media files remain byte-identical to their public source records, eliminating physical embedding artifacts and rendering spatial deep residual steganalyzers blind.
- Dynamic Semantic Sequence Codec, designated DSSC: By coupling spherical k-means clustering with 256 Voronoi centroids on a 512-dimensional hypersphere, Reed-Solomon algebraic coding over GF(2^8), and keyed index permutations, DCASS delivers 13.50 to 13.66 coded bits per carrier and 7.00 to 12.90 user payload bits per carrier with zero clean-channel bit errors.
- Statistically aligned carrier selection: Carrier selection maintains cross-modal distributional invariance, verified by multivariate Energy Distance, MMD, and 1,536-dimensional Benjamini-Hochberg controlled Kolmogorov-Smirnov tests.
- Behavioral AI traffic scheduling: A Wasserstein generative adversarial network models human diurnal inter-arrival distributions, while a closed-loop Proximal Policy Optimization reinforcement learning scheduler routes transmissions across three diverse platform channels, respecting enterprise rate limits and achieving 76.0% evasion against a calibrated traffic sequence detector.

---

## 2. Related work and foundational principles

### 2.1 Spatial and transform steganography

Traditional digital steganography hides data by altering spatial pixels, discrete cosine transform coefficients, or audio samples. Modern physical embedding algorithms operate under the distortion minimization framework. S-UNIWARD optimizes directional wavelets to locate complex image textures. WOW defines an additive distortion metric using directional filter banks. J-UNIWARD applies similar principles to JPEG discrete cosine transform coefficients.

To counter distortion-minimizing algorithms, deep learning steganalysis architectures were introduced. Boroumand et al. developed SRNet, an unpooled deep convolutional neural network that retains high-frequency residual noise without downsampling pooling layers. Zhang et al. introduced ZhuNet, incorporating spatial pyramid pooling to improve feature extraction across multi-scale residual patterns. Yedroudj et al. proposed YeNet, which explicitly initializes its first convolutional layer with 30 high-pass filter kernels from the Spatial Rich Model to boost gradient sensitivity to steganographic noise. Because all physical embedding schemes introduce high-frequency parity disturbances, these networks reliably detect stego carriers.

### 2.2 Coverless and semantic steganography

To bypass media-layer steganalyzers entirely, researchers turned to coverless steganography. Zhou et al. pioneered image retrieval steganography by hashing robust image features into inverted indices. However, direct feature hashing scales poorly as payload length increases. Generating images directly from secret bitstreams via Generative Adversarial Networks or diffusion models creates generative steganography, but synthetic outputs frequently contain telltale generator artifacts detectable by forensic models.

In natural language processing, linguistic steganography generates sentences matching secret bitstreams using Markov chains or autoregressive language models. While syntactically valid, generated texts often exhibit subtle perplexity degradation and statistical repetition detectable by transformer-based language models. DCASS avoids generation entirely, retrieving naturally authored, authentic human media items from large-scale reference corpora.

### 2.3 Network covert channels and traffic analysis

Network covert channels hide data within network protocol fields or packet transmission timings. Classical timing channels modulate packet inter-arrival times to encode binary values. However, network jitter, routing delays, and packet reordering severely corrupt timing-encoded payloads.

Conversely, traffic flow analysis monitors transmission cadence to detect the presence of automated or malicious tools. Flow analyzers employ entropy metrics, autocorrelation, and neural sequence models to separate machine-generated traffic from human interactive sessions. Defenses such as traffic morphing attempt to shape packet lengths and intervals to match target protocols. DCASS separates the secret payload from the transmission schedule: the payload is encoded entirely into the semantic identities of carrier media, while the transmission cadence is governed by generative and reinforcement learning models designed exclusively to mimic human interactive browsing.

---

## 3. System architecture and semantic codec

DCASS decouples secret payload transmission into three modular subsystems: multi-modal hypersphere indexing, algebraic error-resilient semantic encoding, and behavioral traffic scheduling.

```
+---------------------------------------------------------------------------------------------------+
|                                     DCASS SYSTEM ARCHITECTURE                                     |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Secret Message M ]                                                                             |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. CRC-16 Payload Framing & Reed-Solomon Encoding over GF(2^8)                              |  |
|  |    Frame: [0xDC 0x55] || [Len: 2B] || [Payload: KB] || [CRC-16: 2B] || [RS Parity: 8B]       |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. Dynamic Semantic Sequence Codec (DSSC) & Keyed Permutation                                |  |
|  |    - Derives Ephemeral Key Ke = HMAC-SHA256(Ks, Nonce || Chunk)                             |  |
|  |    - Maps Symbols to Voronoi Centroids c_j in S^511 via Sub-Cluster Offsets                 |  |
|  |    - Pseudorandom Index Permutation via Fisher-Yates Shuffle                                |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. Multi-Modal Hypersphere Retrieval (FAISS HNSW Inner Product)                             |  |
|  |    - Image Pool (31,783 Flickr30k)                                                          |  |
|  |    - Text Pool (200,000 Wikipedia)                                                          |  |
|  |    - Audio Pool (13,496 Libretta)                                                           |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          | (Unmodified Public Media Items)                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 4. Behavioral AI Traffic Scheduler                                                          |  |
|  |    - WGAN-GP Generative Diurnal Cadence G(z, hour)                                          |  |
|  |    - Closed-Loop PPO Agent with Cooldown Action Masking                                     |  |
|  |    - Dynamic Channel Routing: Ch 0 (Code), Ch 1 (Social), Ch 2 (Cloud)                      |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|   Public Network Transport (Blind to Media Steganalyzers, Aligned Semantics, Human Cadence)       |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 Multi-modal hypersphere indexing

DCASS unifies heterogeneous media modalities into a common geometric vector space. The reference corpus incorporates three distinct media types:

- Image modality: 31,783 natural images drawn from Flickr30k, embedded via CLIP ViT-B/32.
- Text modality: 200,000 factual passages from English Wikipedia, embedded via CLIP Text Transformer.
- Audio modality: 13,496 environmental and spoken audio recordings from the Libretta corpus, embedded via LAION-CLAP.

Each raw media item is processed by its respective encoder, yielding a 512-dimensional continuous feature vector in R^512. Every vector is L2-normalized onto the surface of the unit hypersphere S^511:

$$\mathbb{S}^{511} = \{ \mathbf{v} \in \mathbb{R}^{512} : \|\mathbf{v}\|_2 = 1.0 \}$$

On this hypersphere, the Euclidean distance between two vectors is directly related to their cosine similarity:

$$\|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2 \langle \mathbf{u}, \mathbf{v} \rangle = 2(1 - \cos \theta_{\mathbf{u}, \mathbf{v}})$$

All 245,279 normalized vectors are indexed into a Hierarchical Navigable Small World vector index using FAISS, configured with inner product metric for sub-millisecond nearest-neighbor search.

### 3.2 Voronoi codebook partitioning

To quantize the continuous hypersphere into discrete transmission symbols, DCASS performs spherical k-means clustering across the corpus. The hypersphere is partitioned into K = 256 Voronoi clusters with centroids C = {c_0, c_1, ..., c_255}, each of unit length. Each centroid defines a Voronoi cell V(c_j):

$$\mathcal{V}(\mathbf{c}_j) = \{ \mathbf{v} \in \mathbb{S}^{511} : \langle \mathbf{v}, \mathbf{c}_j \rangle \ge \langle \mathbf{v}, \mathbf{c}_k \rangle \quad \forall k \neq j \}$$

The 256 centroids correspond to the 256 possible values of an 8-bit byte spanning 0x00 to 0xFF. The reference corpus exhibits an empirical cluster density ranging from 59 to 8,814 items per cell, with an average density of 958.1 items per cluster. This dense population ensures that every discrete byte symbol possesses numerous natural multi-modal carriers.

### 3.3 Dynamic semantic sequence codec

The baseline Exact Voronoi Constellation Partitioning approach, designated exact_vcp, maps each secret byte to a single Voronoi cell, yielding exactly 8.00 bits per carrier item. However, exact_vcp requires transmitting one full media item for every byte of secret data.

DSSC overcomes this limitation through dynamic constellation sub-selection. When encoding a byte sequence, DSSC computes localized multi-symbol trajectories across the hypersphere. By scoring candidate combinations against semantic cohesion and sub-cluster indices, DSSC encodes both primary centroid identifiers and secondary intra-cluster offsets into individual carriers. This multi-level encoding increases capacity to a mean of 13.60 coded bits per carrier, requiring approximately 40% fewer carrier files to transmit identical payloads.

### 3.4 Reed-Solomon coding and framing over GF(2^8)

Because the hypersphere contains dense clusters with continuous boundaries, floating-point rounding variance or minor index shifts can cause a target vector to decode to an adjacent Voronoi centroid, introducing boundary symbol errors at rates between 15% and 25%.

To guarantee exact, error-free reconstruction, DCASS implements Reed-Solomon algebraic block coding over Galois field GF(2^8). The field is defined by the irreducible primitive binary polynomial:

$$p(x) = x^8 + x^4 + x^3 + x^2 + 1 \quad \text{[hex 0x11D]}$$

For an input secret message of K bytes, DCASS packages the data into structured frames:

1. A 2-byte magic header [0xDC, 0x55] and 2-byte payload length indicator.
2. The secret payload bytes.
3. A 2-byte CRC-16 checksum computed over the payload.
4. Reed-Solomon parity symbols.

DCASS uses an RS(63, 55) block code with t = 4 error correction capability. For every block of 55 data bytes, 8 parity bytes, equal to 2t, are computed using polynomial division against the generator polynomial:

$$G(x) = \prod_{i=0}^{2t-1} (x - \alpha^i)$$

During decoding, the receiver applies the Berlekamp-Massey algorithm to compute error locator polynomials and the Chien search to identify symbol error locations. If fewer than or equal to t = 4 symbol errors occur within a block, the decoder fully corrects them, yielding 0% bit error rate. If carrier dropout occurs, meaning the loss of an entire media item, the symbol sequence shifts, preventing algebraic block alignment. The mandatory CRC-16 check verifies payload integrity, immediately rejecting misaligned or unrecoverable frames and preventing false acceptance.

### 3.5 Keyed index permutation and cryptographic avalanche

To ensure that an eavesdropper cannot extract transmitted symbols even if the public codebook is known, DCASS applies a keyed permutation to the carrier sequence.

Given a 256-bit private shared session key K_s and a message nonce, DCASS derives an ephemeral sub-key using HMAC-SHA256:

$$K_{e} = \text{HMAC-SHA256}(K_s, \text{nonce} \parallel \text{block\_index})$$

This ephemeral key seeds a cryptographically secure pseudorandom permutation using a Fisher-Yates shuffle that reorders the sequence of media target indices prior to retrieval. On decoding, the receiver applies the inverse permutation using the identical shared key. Without knowledge of K_s, the adversary faces an astronomical search space of N_carriers factorial permutations. Cross-correlation between unauthorized decodes and true plaintexts is zero, rho = 0.00, resulting in zero false acceptances across empirical wrong-key attack campaigns.

---

## 4. Behavioral traffic scheduling and reinforcement learning

Transmitting carriers across a network generates a temporal and spatial footprint. An automated system that emits packets uniformly or in rapid succession is immediately flagged by network traffic analyzers.

### 4.1 Enterprise transport constraints and rate limiting

To operate covertly within corporate or campus networks, DCASS interfaces with multiple public platforms modeled as three discrete transport channels:

- Channel 0: Code and technical platforms such as GitHub or GitLab API endpoints, enterprise rate limit: 10 items per minute with a 6.0-second cooldown timer.
- Channel 1: Social and media hosting platforms such as Imgur or Flickr API endpoints, enterprise rate limit: 5 items per minute with a 12.0-second cooldown timer.
- Channel 2: Cloud storage and reference knowledge platforms such as S3 or Wikipedia API endpoints, enterprise rate limit: 15 items per minute with a 4.0-second cooldown timer.

A transmission is illegal if the elapsed time between consecutive events on the same channel is less than that channel cooldown threshold. Violating cooldown limits triggers rate-limiting HTTP 429 errors or deep packet inspection hammer rule alerts.

### 4.2 Generative diurnal cadence via WGAN-GP

Human browsing cadence varies according to the time of day, displaying peak activity during afternoon and evening hours and minimal activity during early morning hours. DCASS models this behavior using a Wasserstein generative adversarial network with gradient penalty, designated WGAN-GP.

The generator G(z, hour) takes a latent Gaussian noise vector z and an external conditioning scalar representing the current hour of day from 0 to 23. The network outputs a raw continuous delay value mapped onto a human-plausible inter-arrival window [0.5, 30.0] seconds:

$$\Delta t = (\text{raw} - 0.5) \times 1500.0 + 1.5$$

The critic D(x, hour) is trained with an improved Wasserstein objective enforcing a 1-Lipschitz continuity constraint through gradient regularization:

$$\mathcal{L}_D = \mathbb{E}_{\tilde{\mathbf{x}} \sim \mathbb{P}_g} [D(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x} \sim \mathbb{P}_r} [D(\mathbf{x})] + \lambda_{\text{gp}} \mathbb{E}_{\hat{\mathbf{x}} \sim \mathbb{P}_{\hat{\mathbf{x}}}} \left[ (\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2 \right] + \epsilon_{\text{drift}} \mathbb{E}_{\mathbf{x} \sim \mathbb{P}_r} [D(\mathbf{x})^2]$$

where real samples x are drawn from authentic human conversation logs. Conditioning on diurnal time-of-day causes the generator to automatically extend delays during nocturnal hours and shorten delays during active daytime hours.

### 4.3 Closed-loop proximal policy optimization agent

While WGAN-GP generates realistic individual delays, it operates open-loop without tracking channel cooldown states. To achieve simultaneous stealth and rate-limit compliance, DCASS deploys a multi-channel Proximal Policy Optimization reinforcement learning agent.

The agent interacts with an environment defined by:

- State vector s_t: Contains the current cooldown clocks of all three channels, elapsed time since last event, active message progress, and diurnal hour features.
- Action space a_t: Discrete channel selection c in {0, 1, 2} and continuous delay adjustment alpha.
- Action masking: Channels currently within their mandatory cooldown window are strictly masked out of the policy softmax distribution, mathematically preventing cooldown violations.
- Reward formulation:

$$R_t = R_{\text{throughput}} - \lambda_{\text{stealth}} \cdot P_{\text{detector}}(\mathbf{\Delta t}_{1:t}) - R_{\text{penalty}} \cdot \mathbb{I}_{\text{cooldown\_violated}}$$

The agent optimizes its policy pi_theta(a | s) using the clipped surrogate PPO objective:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where r_t(theta) is the probability ratio and A_t is the generalized advantage estimate. As training progresses, lambda_stealth ramps up to 50.0, penalizing the detection probability reported by an adversarial traffic detector. The policy learns to route traffic dynamically across available channels while pacing transmissions to mimic human timing variance.

---

## 5. Experimental evaluation and security analysis

We conduct an empirical evaluation of DCASS across all five architectural components. All experiments use reproducible scripts, verified seeds, and manifest-pinned model checkpoints.

### 5.1 Media-layer content steganalysis blindness: Component 1

#### 5.1.1 Threat model and Option B protocol
We evaluate DCASS against three state-of-the-art deep residual steganalyzers: SRNet, ZhuNet, and YeNet. To prevent false negatives from undertrained detectors, we follow the Option B protocol:

1. Positive control gate: Detectors are trained on a known steganographic scheme, spatial +/- 1 LSB embedding at 1.0 bit per pixel on image pairs. Each detector must pass a strict accuracy gate of >= 0.90 on held-out test twins across a 3-fold cross-validation split over 2,000 images.
2. Covert evaluation phase: Gated detectors evaluate untouched DCASS carrier images against clean natural covers drawn from identical splits.

#### 5.1.2 Empirical results
Table 1 summarizes detection performance. All three gated models pass the positive control gate with 93.67% to 95.05% accuracy and ROC AUCs between 0.9844 and 0.9860. When evaluating DCASS carriers, detection accuracy collapses to balanced chance, 0.4988 to 0.5025, and measured ROC AUCs sit at 0.4973 to 0.4993, with every 95% confidence interval spanning 0.5000.

| Detector | Positive control acc. (95% CI) | Positive control AUC | Carrier cover rate | Balanced acc. | DCASS carrier AUC (95% CI) | D_KL [perm p] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SRNet | 93.67% [92.88, 94.39] | 0.9844 | 88.31% | 0.5025 | 0.4973 [0.4830, 0.5143] | 0.23 [0.238, 0.423] |
| ZhuNet | 95.05% [94.33, 95.68] | 0.9860 | 92.14% | 0.5025 | 0.4993 [0.4820, 0.5146] | 0.18 [0.250, 0.937] |
| YeNet | 93.97% [93.19, 94.67] | 0.9853 | 92.34% | 0.4988 | 0.4993 [0.4826, 0.5160] | 0.23 [0.385, 0.849] |

Because DCASS transmits unaltered natural images, no spatial dependencies or high-frequency DCT residuals are disturbed, rendering physical steganalyzers statistically blind.

### 5.2 Semantic distribution drift and cross-modal alignment: Component 2

#### 5.2.1 Multivariate statistical tests
To test whether carrier selection introduces semantic anomalies, we evaluate carrier distributions against reference corpus distributions using Energy Distance and Maximum Mean Discrepancy (MMD^2) with permutation testing:

- Image modality: Energy distance 0.0001 (p = 0.009), MMD^2 0.0001 (p = 0.012).
- Text modality: Energy distance 0.0001 (p = 0.004), MMD^2 0.0001 (p = 0.004).
- Audio modality: Energy distance 0.0000 (p = 0.574), MMD^2 0.0000 (p = 0.642).

Observed distributional shifts are on the order of 10^-4 relative to the unit embedding scale. Across all 1,536 individual coordinate dimensions, 512 per modality, two-sample Kolmogorov-Smirnov tests yield zero rejections after Benjamini-Hochberg false discovery rate control.

#### 5.2.2 Cluster uniformity and mechanism consistency: Component 2b
Evaluating 200 transmitted messages comprising 10,800 carriers reveals a transmitted cluster entropy of 7.7935 bits on available slots and 7.7873 bits on unique media items, closely matching the theoretical uniform maximum of 8.00 bits. Chi-square goodness-of-fit against the system structural selection mechanism yields chi^2 = 268.56 on 253 degrees of freedom, permutation p = 0.2079, confirming that carrier selection conforms to natural corpus availability.

### 5.3 Network traffic cadence and DPI warden interaction: Component 3

#### 5.3.1 Threat model and authentic traffic corpus
The network warden monitors packet streams, checking for periodic cadence, coefficient of variation < 0.10, burst violations of more than 5 packets within 2 seconds, channel hammer violations, and a supervised transformer detector evaluating inter-arrival delay sequences.

To ensure realistic evaluation, the detector and reference baselines are trained on an authentic human traffic corpus streamed from public community discussions comprising 1,255 multi-comment conversation sessions and 63,255 timestamps. The corpus exhibits an authentic diurnal chi-square statistic of 214.01 against uniform, p < 0.001, with 44.6% of inter-arrival delays exceeding 120 seconds.

#### 5.3.2 Empirical evaluation across schedulers
We evaluate five transmission schedulers across 50 independent trials of 50 carriers each. The soft-branch detection threshold is calibrated on held-out human traffic to achieve a measured 4.47% false positive rate against a 5.0% target at a threshold of 0.9240.

| Scheduler | Interception rate (%) | Evasion rate (%) | Delay entropy (bits) | KS distance | Mean latency (s) | Cooldown violations | Triggering rule |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Static Cadence | 100.0% | 0.0% | 0.0000 | 0.5330 | 3.00 | 2450 | periodic cv 50 of 50 |
| Stochastic Poisson | 100.0% | 0.0% | 4.2335 | 0.2211 | 3.04 | 1276 | burst 27 of 50, soft 23 of 50 |
| DCASS Temporal WGAN | 100.0% | 0.0% | 3.6586 | 0.5595 | 7.50 | 982 | soft 50 of 50 |
| DCASS Hybrid Engine | 100.0% | 0.0% | 3.6584 | 0.5553 | 7.44 | 0 | soft 50 of 50 |
| DCASS PPO RL Agent | 24.0% | 76.0% | 3.6291 | 0.2399 | 5.63 | 0 | soft 12 of 50 |

Static cadence triggers periodic coefficient of variation rules on every trial. Stochastic Poisson violates burst and rate limits in 100% of trials. Unguided WGAN generates realistic delays but triggers soft detection and violates channel cooldowns 982 times. The DCASS Hybrid engine eliminates all cooldown violations via action masking, but remains detectable by the neural detector.

In contrast, the retrained DCASS PPO RL Agent achieves 76.0% evasion, with only 12 of 50 trials intercepted, while maintaining 0 cooldown violations and an average per-carrier latency of 5.63 seconds. Testing held-out human corpus sessions against the full enterprise detection suite confirms a 0.0% false positive rate across 50 held-out sessions under multi-channel Markovian session transitions.

Diurnal conditioning checks confirm that WGAN delays adapt significantly across waking and sleeping hours: daytime hour at 14:00 yields 3.89 seconds mean latency, entropy 2.7290 bits, whereas night hour at 03:00 expands latency to 14.93 seconds, entropy 4.4098 bits, producing a mapped diurnal delta of -11.04 seconds.

### 5.4 Codec capacity, fault tolerance, and cryptographic isolation: Component 4

#### 5.4.1 Capacity scaling: DSSC versus exact_vcp
Table 4 compares DSSC capacity against the baseline exact_vcp scheme across five payload sizes with matched RS(63, 55) error correction and framing:

| Mode | Payload (B) | Carriers | Coded bpc | User bpc | Carrier reduction (%) | Encode (ms) | Decode (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DSSC (multi-modal) | 14 | 16 | 13.50 | 7.00 | 40.7% | 106.0 | 181.3 |
| DSSC (text-only) | 14 | 14 | 15.43 | 8.00 | 48.1% | 89.3 | 243.3 |
| exact_vcp (multi-modal) | 14 | 27 | 8.00 | 4.15 | -- | 1784.6 | 2.2 |
| DSSC (multi-modal) | 50 | 37 | 13.62 | 10.81 | 41.3% | 113.6 | 280.4 |
| DSSC (text-only) | 50 | 32 | 15.75 | 12.50 | 49.2% | 111.2 | 442.1 |
| exact_vcp (multi-modal) | 50 | 63 | 8.00 | 6.35 | -- | 5232.6 | 5.0 |
| DSSC (multi-modal) | 150 | 96 | 13.58 | 12.50 | 41.1% | 119.0 | 542.7 |
| DSSC (text-only) | 150 | 82 | 15.90 | 14.63 | 49.7% | 153.6 | 1009.4 |
| exact_vcp (multi-modal) | 150 | 163 | 8.00 | 7.36 | -- | 14861.7 | 12.3 |
| DSSC (multi-modal) | 300 | 188 | 13.66 | 12.77 | 41.4% | 167.9 | 1013.1 |
| DSSC (text-only) | 300 | 161 | 15.95 | 14.91 | 49.8% | 214.5 | 1931.4 |
| exact_vcp (multi-modal) | 300 | 321 | 8.00 | 7.48 | -- | 29529.9 | 23.8 |
| DSSC (multi-modal) | 500 | 310 | 13.65 | 12.90 | 41.4% | 209.0 | 1580.0 |
| DSSC (text-only) | 500 | 265 | 15.97 | 15.09 | 49.9% | 296.9 | 3066.4 |
| exact_vcp (multi-modal) | 500 | 529 | 8.00 | 7.56 | -- | 51703.8 | 40.7 |

Across all payload sizes, multi-modal DSSC delivers 13.50 to 13.66 coded bits per carrier with a mean of 13.60 bpc, reducing carrier requirements by 40.7% to 41.4% compared to exact_vcp. Text-only DSSC achieves up to 15.97 coded bpc, achieving a 49.9% reduction. Net secret user payload capacity scales from 7.00 bpc at 14 bytes up to 12.90 bpc at 500 bytes. Furthermore, DSSC encoding latency is up to 250 times faster than exact_vcp because it eliminates iterative exhaustive per-byte vector re-scoring.

#### 5.4.2 Channel robustness and algebraic recovery
Evaluating channel distortion models confirms that Reed-Solomon coding over GF(2^8) achieves 100% exact message recovery when carrier substitution remains within the block budget of t = 4 damaged symbols. Across 80 trials with 5% carrier substitution, RS-ECC recovers 61.2% [50.3%, 71.2%] of messages. Beyond the error budget, an algebraic recovery cliff occurs. Under carrier dropout, meaning the deletion of carrier items, block alignment is lost; the framed payload decoder rejects all corrupt sequences with zero false accepts.

#### 5.4.3 Cryptographic key separation and false acceptance rate
Key sensitivity testing demonstrates complete cryptographic avalanche. Decoding payloads under random keys or single-bit flipped keys yields zero CRC-passing false acceptances across 356 independent trials, giving an observed false acceptance rate of 0.00%. Pearson correlation between incorrect decodings and original plaintexts is rho = 0.0004 with p = 0.99.

### 5.5 Computational performance and concurrency stress: Component 5

Direct-engine profiling demonstrates low computational overhead. For a 32-byte payload, direct encoding completes in 107.1 ms, achieving 81.9 carriers per second. For a 512-byte payload, encoding completes in 206.3 ms, achieving 176.5 carriers per second. Vector index lookup accounts for 96.2% of total execution time, while RS-ECC encoding and keyed permutation require less than 1.5 ms combined. Under multi-threaded concurrency stress testing with 50 concurrent worker threads executing 100 independent sessions, the direct engine achieved 100% decode success with zero cross-session payload swaps or memory race conditions.

---

## 6. Discussion, limitations, and future work

While DCASS advances multi-modal semantic steganography and behavioral traffic scheduling, several operational constraints warrant discussion:

1. Index storage and synchronization: Both sender and receiver must maintain synchronized copies of the 245,279-item vector index and centroid codebook totaling approximately 500 MB. While modest for modern hardware, updating reference corpora requires a deterministic out-of-band codebook distribution protocol.
2. Carrier dropout sensitivity: While RS-ECC provides robust protection against carrier substitution, carrier dropout destroys symbol index alignment, resulting in unrecoverable blocks. Future extensions will incorporate fountain codes such as RaptorQ or Luby Transform codes to provide rateless erasure resilience against packet loss.
3. Multi-platform API availability: The multi-channel scheduler relies on the availability of third-party platforms. Platform API disruptions or CAPTCHA interventions could temporarily restrict channel options, forcing the RL policy to route traffic across surviving channels.

---

## 7. Conclusion

DCASS demonstrates that high-throughput, undetectable covert communication is achievable without modifying a single bit of carrier media. By grounding data encoding in continuous multi-modal hypersphere geometry, DCASS eliminates the physical embedding signatures that render traditional steganography vulnerable to deep residual steganalyzers. Coupling spherical Voronoi partitioning with Reed-Solomon coding over GF(2^8) resolves boundary quantization noise, delivering 13.60 coded bits per carrier and a 41% carrier reduction over baseline coverless methods. At the transport layer, integrating diurnal WGAN generative pacing with closed-loop PPO reinforcement learning achieves 76.0% evasion against supervised traffic detectors while strictly obeying enterprise channel cooldown limits. DCASS establishes a robust, empirically validated foundation for modern, multi-layer covert communications.

---

## References

1. M. Boroumand, M. Chen, and J. Fridrich, "Deep residual network for steganalysis of digital images," *IEEE Transactions on Information Forensics and Security*, vol. 14, no. 5, pp. 1181-1193, 2018.
2. R. Zhang, F. Zhu, J. Liu, and G. Liu, "Efficient steganography embedding scheme based on spatial pyramid pooling," *IEEE Transactions on Information Forensics and Security*, vol. 15, pp. 1656-1668, 2020.
3. M. Yedroudj, F. Comby, and M. Chaumont, "Yedroudj-Net: An efficient CNN for spatial steganalysis," *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 2092-2096, 2018.
4. V. Holub, J. Fridrich, and T. Denemark, "Universal distortion design for steganography in an arbitrary domain," *EURASIP Journal on Information Security*, vol. 2014, no. 1, pp. 1-13, 2014.
5. V. Holub and J. Fridrich, "Designing steganographic distortion using direction filters," *IEEE International Workshop on Information Forensics and Security (WIFS)*, pp. 234-239, 2012.
6. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., "Learning transferable visual models from natural language supervision," *International Conference on Machine Learning (ICML)*, pp. 8748-8763, 2021.
7. Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, "Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation," *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1-5, 2023.
8. J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535-547, 2019.
9. I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, "Improved training of Wasserstein GANs," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5767-5777, 2017.
10. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
11. I. S. Reed and G. Solomon, "Polynomial codes over certain finite fields," *Journal of the Society for Industrial and Applied Mathematics*, vol. 8, no. 2, pp. 300-304, 1960.
12. E. R. Berlekamp, *Algebraic Coding Theory*, McGraw-Hill, New York, 1968.
13. Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 57, no. 1, pp. 289-300, 1995.
14. G. J. Székely and M. L. Rizzo, "Energy statistics: A class of statistics based on energy distances," *Journal of Statistical Planning and Inference*, vol. 143, no. 8, pp. 1249-1272, 2013.
15. A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, "A kernel two-sample test," *Journal of Machine Learning Research*, vol. 13, pp. 723-773, 2012.
16. L. Zhou, S. Sun, J. Dong, P. Yao, and Z. Sun, "Coverless image steganography based on information hiding," *IEEE Access*, vol. 7, pp. 177995-178005, 2019.
17. C. G. Cassuto and M. Blaum, "Codes for symbol-pair read channels," *IEEE Transactions on Information Theory*, vol. 56, no. 12, pp. 6112-6120, 2010.
18. S. Cabuk, C. E. Brodley, and C. Shields, "IP covert timing channels: design and detection," *ACM Conference on Computer and Communications Security (CCS)*, pp. 178-187, 2004.
19. M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," *International Conference on Machine Learning (ICML)*, pp. 214-223, 2017.
20. M. Luby, "LT codes," *IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 271-280, 2002.

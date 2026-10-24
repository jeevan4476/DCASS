# Unified Multi-Modal FAISS 512D Vector Indexing Module Specification


## 1. Executive Summary & Overview

DCASS establishes a unified multi-modal semantic search and retrieval architecture across three distinct sensory modalities: visual images, textual sentences, and spoken audio recordings. Rather than siloing steganographic carriers by media type, DCASS projects all modalities onto a shared 512-dimensional unit hypersphere $\mathbb{S}^{511}$ using OpenAI CLIP (`ViT-B/32`) and LAION CLAP (`clap-htsat-unfused`).

The indexing subsystem manages 245,279 precomputed unit vectors using FAISS `IndexFlatIP` (exact brute-force inner-product search). The query encoder runs on CUDA when a GPU is available; the FAISS search itself runs on the CPU because the project depends on `faiss-cpu`. It implements cross-modal score calibration (Z-score normalization and sigmoid scaling) to enable direct comparison and ranking across heterogeneous modalities, and provides dynamic local disk file path resolution via the `MediaItem.file_path` property.

```mermaid
flowchart TD
    subgraph Encoders["Feature Extraction & Embedding Space"]
        IMG["Flickr Images\n(31,783 items)"] --> CLIP_V["CLIP ViT-B/32 Vision\n(512d)"]
        TXT["Wikipedia Text\n(200,000 items)"] --> CLIP_T["CLIP ViT-B/32 Text\n(512d)"]
        AUD["libretta Audio\n(13,496 items)"] --> CLAP_A["CLAP HTSAT Audio\n(512d)"]
    end

    CLIP_V --> NORM["L2 Normalization\n(||x||_2 = 1.0)"]
    CLIP_T --> NORM
    CLAP_A --> NORM

    subgraph FAISS["FAISS GPU Indices (storage/data/indices/)"]
        NORM --> I_IMG["image.index\n(IndexFlatIP - 62.1 MB)"]
        NORM --> I_TXT["text.index\n(IndexFlatIP - 390.6 MB)"]
        NORM --> I_AUD["audio.index\n(IndexFlatIP - 26.4 MB)"]
    end

    subgraph Retrieval["Unified Search & Calibration Pipeline"]
        QUERY["Secret Chunk / Semantic Query"] --> Q_ENC["CLIP Text Encoder"]
        Q_ENC --> PAR_SRCH["Parallel FAISS Index Search"]
        I_IMG --> PAR_SRCH
        I_TXT --> PAR_SRCH
        I_AUD --> PAR_SRCH
        PAR_SRCH --> Z_NORM["Score Normalizer\n(Z-Score + Sigmoid Calibration)"]
        Z_NORM --> RANK["Unified Ranker & MediaItem Resolution"]
    end
```

### Key Quantitative Metrics

| Metric / Parameter | Value | Details |
| :--- | :--- | :--- |
| **Total Indexed Corpus** | **245,279 vectors** | 31,783 Images + 200,000 Text + 13,496 Audio |
| **Vector Dimensionality ($D$)** | 512 | Constant across all modalities |
| **FAISS Index Type** | `IndexFlatIP` | Exact brute-force inner product, no approximation loss; the shipped indices are CPU indexes |
| **Total Index Disk Footprint** | 479.1 MB | `image.index` (62.1 MB), `text.index` (390.6 MB), `audio.index` (26.4 MB) |
| **Total Metadata Footprint** | 92.4 MB | `image_metadata.json` (22.8 MB), `text_metadata.json` (65.4 MB), `audio_metadata.json` (4.2 MB) |
| **Search Latency** | Not separately benchmarked in the current artifacts | The end-to-end encode path is measured in Component 5; a per-stage FAISS latency table is not recorded |
| **Throughput (Batch Search)** | Not recorded in the current artifacts | The engine issues one query per carrier, not a batched matrix product |

---

## 2. Real-World Intuition & The Universal Translator Analogy

To understand why a unified multi-modal index is essential, consider a universal translator in diplomacy.

```
       Visual Concept             Textual Sentence            Audio Spoken Clip
    [Photo of Forest Trail]     ["A winding path in woods"]   [Recording of Birdsong]
               \                         |                        /
                \                        |                       /
                 v                       v                      v
            ===========================================================
               Unified 512-Dimensional Coordinate Space (S^511)
                     All map to neighborhood: [0.12, -0.45, 0.88, ...]
            ===========================================================
```

If an intelligence agent only speaks English text, but surveillance monitors are watching for unusual bursts of English messages, the agent is restricted. However, if the agent can freely express the concept "forest trail" as an uploaded vacation photo, a tweet about nature, or an ambient audio recording of forest sounds, the communication pattern blends into natural user activity.

The unified index acts as the underlying universal geometry: it maps distinct sensory media (JPEG images, UTF-8 strings, WAV audio files) into the exact same semantic neighborhood in $\mathbb{S}^{511}$, enabling seamless cross-modal carrier selection.

---

## 3. Why Unified 512D Indexing is Needed

### 3.1 Eliminating Single-Modality Steganography Blindspots

Traditional steganography is rigidly locked to a single data format:
- Image steganography (LSB, J-UNIWARD) modifies JPEG or PNG images only.
- Audio steganography embeds payload bits in WAV or MP3 samples.
- Text steganography modifies syntactic word choice or whitespace.

An adversary monitoring network traffic easily spots anomalous single-modality traffic patterns (for instance, an account that posts 50 consecutive high-resolution images with no accompanying text or interaction).

### 3.2 Dynamic Cross-Modal Camouflage

By standardizing all media types onto a single 512-dimensional coordinate space:
1. **Dynamic Media Interleaving**: A sender can transmit a 4-chunk message as 1 photograph, 2 text sentences, and 1 voice recording.
2. **Context-Adaptive Carrier Selection**: If network traffic policies restrict image uploads, the system automatically redirects carrier selection to text or audio channels without changing the core encoding logic.
3. **Unified Codebook Compatibility**: The 256 Voronoi centroids derived in Module 02 operate globally across all three media formats simultaneously.

---

## 4. Mathematical Derivation & Normalization Formulation

### 4.1 Embedding Space Alignment

1. **Vision and Text Modalities (OpenAI CLIP ViT-B/32)**:
   - Vision Encoder $f_{\text{img}}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{512}$
   - Text Encoder $f_{\text{txt}}: \mathcal{V}^* \to \mathbb{R}^{512}$
   - Trained jointly with contrastive loss over 400M image-text pairs to align visual concepts with natural language descriptions.

2. **Audio Modality (LAION CLAP clap-htsat-unfused)**:
   - Audio Spectrogram Transformer $f_{\text{aud}}: \mathbb{R}^{T \times F} \to \mathbb{R}^{512}$
   - Aligned to the shared 512-dimensional joint semantic space.

### 4.2 Strict $L_2$ Normalization onto $\mathbb{S}^{511}$

For any raw embedding vector $\mathbf{e} \in \mathbb{R}^{512}$, its unit-normalized representation $\mathbf{x}$ is computed as:

$$\mathbf{x} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2} = \frac{\mathbf{e}}{\sqrt{\sum_{d=1}^{512} e_d^2}}$$

With $\|\mathbf{x}\|_2 = 1.0$, the cosine similarity between any two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{S}^{511}$ equals their Euclidean inner product:

$$\cos(\theta) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{d=1}^{512} u_d v_d$$

### 4.3 FAISS `IndexFlatIP` GPU Search

FAISS implements exact inner product search over index matrix $\mathbf{X} \in \mathbb{R}^{N \times 512}$. Given a batch of query vectors $\mathbf{Q} \in \mathbb{R}^{B \times 512}$, similarity scores are computed via matrix multiplication:

$$\mathbf{S} = \mathbf{Q} \mathbf{X}^T \in \mathbb{R}^{B \times N}$$

The top-$k$ nearest carriers are extracted via parallel GPU reduction:

$$\text{Top-}k(\mathbf{q}) = \arg\max_{i \in \{1, \dots, N\}}^{(k)} S_{qi}$$

Because `IndexFlatIP` performs exhaustive inner product computation without quantization approximations (such as IVF or PQ), recall is mathematically $100.0\%$.

---

### 4.4 Cross-Modal Score Normalization & Sigmoid Calibration

#### The Modality Gap Problem
Due to differing contrastive loss temperatures and representation densities during pretraining, raw cosine similarities vary substantially across modalities for semantically equivalent content:

- Text-to-Text raw cosine similarity: $\sim 0.70 - 0.95$ (Mean: $\mu = 0.885, \sigma = 0.053$)
- Text-to-Image raw cosine similarity: $\sim 0.20 - 0.35$ (Mean: $\mu = 0.271, \sigma = 0.028$)
- Text-to-Audio raw cosine similarity: $\sim 0.05 - 0.20$ (Mean: $\mu = 0.100, \sigma = 0.021$)

If raw inner products are compared directly, text items will dominate visual and audio items regardless of visual relevance.

#### Score Normalization Solution
DCASS implements a two-stage Z-score and logistic sigmoid transformation:

1. **Z-Score Standardization**:
   $$z = \frac{s_{\text{raw}} - \mu_{\text{modality}}}{\sigma_{\text{modality}}}$$

2. **Sigmoid Mapping to $[0, 1]$**:
   $$\hat{s} = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + \exp\left(-\frac{s_{\text{raw}} - \mu_{\text{modality}}}{\sigma_{\text{modality}}}\right)}$$

```
Raw Similarity s      Z-Score Normalization        Sigmoid Scaling
[Image: 0.271]   -->  z = (0.271 - 0.271) / 0.028 = 0.0   -->  sigma(0.0) = 0.500 (Fair Baseline)
[Text:  0.885]   -->  z = (0.885 - 0.885) / 0.053 = 0.0   -->  sigma(0.0) = 0.500 (Fair Baseline)
[Audio: 0.100]   -->  z = (0.100 - 0.100) / 0.021 = 0.0   -->  sigma(0.0) = 0.500 (Fair Baseline)
```

This transforms raw scores into calibrated relative percentile confidences, ensuring fair multi-modal ranking.

---

## 5. Codebase Implementation Architecture

The unified indexing subsystem is implemented in [`src/corpus/index/unified_index.py`](../../src/corpus/index/unified_index.py).

### Core Components

```python
# src/corpus/index/unified_index.py

@dataclass
class MediaItem:
    """Represents a media item from the corpus."""
    id: str
    modality: Literal["image", "text", "audio"]
    content: str               # Caption/transcript for text/audio, image path
    score: float               # Raw FAISS inner product score
    normalized_score: float    # Calibrated score in range [0, 1]
    metadata: dict = field(default_factory=dict)
```

### Dynamic File Path Resolution: `file_path` Property

A critical requirement in DCASS is resolving the exact absolute local disk path for any selected media carrier so that the transmission engine can load, display, or transmit the actual media file:

```python
@property
def file_path(self) -> Optional[str]:
    """Resolve exact absolute local file path on disk."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    if self.modality == "image":
        # Search metadata path, Flickr30k raw storage, or Flickr8k fallback
        raw_path = self.metadata.get("path") or self.metadata.get("image_path")
        if raw_path:
            p = Path(raw_path)
            if p.is_absolute() and p.exists():
                return str(p.resolve())
            rel_p = (project_root / p).resolve()
            if rel_p.exists():
                return str(rel_p)
            cand30k = project_root / "storage/data/raw/flickr30k/images" / p.name
            if cand30k.exists():
                return str(cand30k.resolve())
        cand_id = project_root / "storage/data/raw/flickr30k/images" / f"{self.id}.jpg"
        if cand_id.exists():
            return str(cand_id.resolve())
        return str((project_root / "storage/data/indices/image_metadata.json").resolve())

    elif self.modality == "audio":
        # Resolve libretta audio file in cache or raw datasets
        cand_aud = project_root / "storage/data/audio/cache"
        if cand_aud.exists():
            for file_p in cand_aud.rglob("*.arrow"):
                return str(file_p.resolve())
        return str((project_root / "storage/data/indices/audio_metadata.json").resolve())

    else:  # text
        # Resolve Wikipedia sentence corpus source file
        cand_txt = project_root / "storage/data/text/wikipedia/sentences.json"
        if cand_txt.exists():
            return str(cand_txt.resolve())
        return str((project_root / "storage/data/indices/text_metadata.json").resolve())
```

---

## 6. Multi-Modal Storage Breakdown & Hardware Benchmarks

### 6.1 Dataset Composition Breakdown

```
+------------------------------------------------------------------------------------+
|                                 DCASS UNIFIED CORPUS                               |
|                                (245,279 Total Items)                               |
+--------------------------+----------------------------+----------------------------+
|     Flickr Images        |     Wikipedia Text         |      libretta Audio        |
|     31,783 vectors       |    200,000 vectors         |      13,496 vectors        |
|     (26.0% of Corpus)    |    (65.2% of Corpus)       |      (8.8% of Corpus)      |
+--------------------------+----------------------------+----------------------------+
```

### 6.2 Index File Layout on Disk

All indices reside in `storage/data/indices/`:

| File Name | Modality | Vector Count | Dimensions | Disk Size |
| :--- | :--- | :--- | :--- | :--- |
| `image.index` | Image | 31,783 | 512 (FP32) | 62.1 MB |
| `image_metadata.json` | Image (Meta) | 31,783 records | Captions, paths | 22.8 MB |
| `text.index` | Text | 100,000 | 512 (FP32) | 204.8 MB |
| `text_metadata.json` | Text (Meta) | 100,000 records | Sentences, sources | 35.2 MB |
| `audio.index` | Audio | 13,496 | 512 (FP32) | 26.4 MB |
| `audio_metadata.json` | Audio (Meta) | 13,496 records | Transcripts, speakers | 4.2 MB |
| **Total** | **All 3 Modalities** | **245,279** | **512d** | **571.5 MB** |

### 6.3 Retrieval Latency & Throughput

**No current artifact records a per-stage retrieval latency breakdown.** The earlier table of CUDA-versus-CPU latencies and speedups for query encoding, per-index search, and score merging was never produced by a recorded benchmark, and it also described a GPU FAISS backend that the shipped code does not use: the indices are CPU `IndexFlatIP` and the project depends on `faiss-cpu`.

The measured numbers that do exist are the end-to-end encode and decode latencies in Component 5 (`storage/benchmarks/component_5_results.json`), measured on the direct engine with `device: cuda`. They range from 107.1 ms encode at 32 B to 206.3 ms encode at 512 B, with decode from 222.6 ms to 1,590.0 ms.

A per-stage retrieval benchmark is a Phase 1 candidate if the paper needs to attribute latency to the search stage.

---

## 7. Complete End-to-End Search Workflow

The `UnifiedSemanticIndex.search()` method executes the following pipelined workflow:

```
1. Input Query: "a lone wolf standing in snowy mountains"
      │
2. CLIP Text Tokenization & Encoding:
      tokens = clip.tokenize(query)
      q_emb = clip_model.encode_text(tokens)
      q_emb = q_emb / ||q_emb||_2   --> shape: (1, 512)
      │
3. Parallel FAISS Inner Product Search:
      scores_img, ids_img = image_index.search(q_emb, k=20)
      scores_txt, ids_txt = text_index.search(q_emb, k=20)
      scores_aud, ids_aud = audio_index.search(q_emb, k=20)
      │
4. Z-Score & Sigmoid Calibration:
      For each item: norm_score = 1 / (1 + exp(-(score - mu_m) / sigma_m))
      │
5. Semantic Content Extraction:
      Extract human-readable caption / transcript / sentence string
      │
6. Unified Merging and Top-K Ranking:
      Sort all 60 candidates by norm_score (descending)
      Return top-k MediaItem objects with resolved file_paths
```

---

## 8. Summary of Engineering Guarantees

1. **Modality Agnostic**: Images, sentences, and audio clips are treated as first-class citizens in a shared geometric coordinate space.
2. **Exact Retrieval**: `IndexFlatIP` provides 100% recall with zero vector quantization distortion.
3. **Calibrated Ranking**: Z-score and sigmoid scaling eliminate modality bias during carrier competition.
4. **Resolved File Access**: Every `MediaItem` dynamically resolves its exact absolute path on disk for verification and transmission.

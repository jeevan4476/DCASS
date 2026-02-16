# DCASS System Architecture

## High-Level Architecture (4 Layers)

```mermaid
flowchart TB
    subgraph CLI["CLI Layer"]
        CLI_ENCODE["encode <message>"]
        CLI_DECODE["decode <ids>"]
        CLI_DIST["distribute <message>"]
        CLI_DEMO["demo <message>"]
    end

    subgraph ENGINE["Engine Layer"]
        ENCODER["SemanticEncoder"]
        DECODER["SemanticDecoder"]
        CHUNKER["SemanticChunker"]
        
        ENCODER --> CHUNKER
        ENCODER --> |"search"| CORPUS
        DECODER --> |"lookup"| CORPUS
    end

    subgraph CORPUS["Corpus Layer"]
        UNIFIED["UnifiedSemanticIndex"]
        NORMALIZER["ScoreNormalizer"]
        
        subgraph EMBEDDERS["Embedders"]
            CLIP["CLIPEmbedder<br/>(image/text)"]
            CLAP["AudioEmbedder<br/>(audio)"]
        end
        
        subgraph INDICES["FAISS Indices"]
            IMG_IDX["image.index<br/>(512-dim)"]
            TXT_IDX["text.index<br/>(512-dim)"]
            AUD_IDX["audio.index<br/>(512-dim)"]
        end
        
        subgraph METADATA["Metadata"]
            IMG_META["image_metadata.json"]
            TXT_META["text_metadata.json"]
            AUD_META["audio_metadata.json"]
        end
        
        UNIFIED --> NORMALIZER
        UNIFIED --> CLIP
        UNIFIED --> CLAP
        UNIFIED --> INDICES
        UNIFIED --> METADATA
    end

    subgraph DISTRIBUTION["Distribution Layer"]
        DISPATCHER["Dispatcher"]
        SCHEDULER["Scheduler"]
        NOISE["NoiseController"]
        
        subgraph CHANNELS["Channels"]
            CONSOLE["ConsoleChannel"]
            FOLDER["LocalFolderChannel"]
            FUTURE["Future: Social, Email..."]
        end
        
        NOISE --> SCHEDULER
        SCHEDULER --> DISPATCHER
        DISPATCHER --> CHANNELS
    end

    CLI_ENCODE --> ENCODER
    CLI_DECODE --> DECODER
    CLI_DEMO --> ENCODER
    CLI_DEMO --> DECODER
    CLI_DIST --> ENCODER
    CLI_DIST --> DISTRIBUTION
    
    ENCODER --> |"MediaSequence"| DISTRIBUTION
```

## Data Flow Overview

```mermaid
flowchart LR
    subgraph INPUT
        MSG["Secret Message"]
    end
    
    subgraph ENCODE
        CHUNK["Chunk Message"]
        SEARCH["Search Corpus"]
        SELECT["Select Media"]
    end
    
    subgraph TRANSMIT
        NOISE["Add Noise"]
        SCHEDULE["Schedule"]
        DISPATCH["Dispatch"]
    end
    
    subgraph OUTPUT
        CHAN["Channels"]
    end
    
    subgraph RECEIVE
        IDS["Media IDs"]
    end
    
    subgraph DECODE
        LOOKUP["Lookup IDs"]
        VERIFY["Verify"]
        EXTRACT["Extract Content"]
    end
    
    subgraph RESULT
        MEANING["Reconstructed<br/>Meaning"]
    end
    
    MSG --> CHUNK --> SEARCH --> SELECT
    SELECT --> |"Media IDs"| NOISE --> SCHEDULE --> DISPATCH --> CHAN
    CHAN -.-> |"observed"| IDS
    IDS --> LOOKUP --> VERIFY --> EXTRACT --> MEANING
```

## Component Details

### 1. CLI Layer
- **Purpose**: User interface for all DCASS operations
- **Commands**: encode, decode, demo, distribute
- **Implementation**: `src/cli/main.py`

### 2. Engine Layer
- **Purpose**: Core encoding/decoding logic
- **Components**:
  - `SemanticEncoder`: Message → Media sequence
  - `SemanticDecoder`: Media IDs → Reconstructed meaning
  - `SemanticChunker`: Message → Semantic chunks
- **Implementation**: `src/engine/`

### 3. Corpus Layer
- **Purpose**: Multi-modal semantic search
- **Components**:
  - `UnifiedSemanticIndex`: Unified search across modalities
  - `ScoreNormalizer`: Cross-modal score normalization
  - `CLIPEmbedder`: Image/text embeddings (512-dim)
  - `AudioEmbedder`: Audio embeddings via CLAP (512-dim)
- **Implementation**: `src/corpus/`

### 4. Distribution Layer
- **Purpose**: Human-like content distribution
- **Components**:
  - `NoiseController`: Adds jitter, skips, gaps
  - `Scheduler`: Timed dispatch
  - `Dispatcher`: Channel selection (round-robin, etc.)
  - Channels: Console, LocalFolder, (future: Social, Email)
- **Implementation**: `src/distribution/`

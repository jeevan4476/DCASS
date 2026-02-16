# DCASS Architecture Diagram

## System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        MSG[/"Secret Message"/]
        MEDIA[/"Received Media Sequence"/]
    end

    subgraph Corpus["Corpus & Indexing Layer"]
        subgraph Loaders["Data Loaders"]
            FL[FlickrLoader]
            WL[WikipediaLoader]
            AL[AudioLoader<br/>NOT IMPLEMENTED]
        end
        
        subgraph Embedders["Embedders"]
            IE[ImageEmbedder<br/>CLIP ViT-B/32]
            TE[TextEmbedder<br/>CLIP Text Encoder]
            AE[AudioEmbedder<br/>NOT IMPLEMENTED]
        end
        
        subgraph Index["FAISS Index"]
            UI[UnifiedSemanticIndex]
            MI_IMG[ModalityIndex<br/>image.index]
            MI_TXT[ModalityIndex<br/>text.index]
            MI_AUD[ModalityIndex<br/>audio.index<br/>NOT IMPLEMENTED]
            SN[ScoreNormalizer]
        end
        
        FL --> IE
        WL --> TE
        AL -.-> AE
        IE --> MI_IMG
        TE --> MI_TXT
        AE -.-> MI_AUD
        MI_IMG --> UI
        MI_TXT --> UI
        MI_AUD -.-> UI
        UI --> SN
    end

    subgraph Engine["Encoding/Decoding Engine"]
        subgraph Encoding["Encoding"]
            SC[SemanticChunker]
            SE[SemanticEncoder]
            EC[EnhancedChunk]
        end
        
        subgraph Decoding["Decoding"]
            SD[SemanticDecoder]
            DM[DecodedMessage]
        end
        
        MSG --> SC
        SC --> EC
        EC --> SE
        SE --> |"search"| UI
        UI --> |"SearchResult[]"| SE
        SE --> EM[EncodedMessage]
        
        MEDIA --> SD
        SD --> |"lookup"| UI
        SD --> DM
    end

    subgraph Stealth["Stealth & Distribution Layer"]
        subgraph StealthAI["AI Stealth (NOT IMPLEMENTED)"]
            GAN[GANScheduler<br/>Behavioral Mimicry]
            RL[RLPolicyAgent<br/>Adaptive Decisions]
        end
        
        subgraph Distribution["Distribution"]
            SCHED[Scheduler]
            DISP[Dispatcher]
            
            subgraph Channels["Channels"]
                CC[ConsoleChannel]
                LFC[LocalFolderChannel]
                EC2[EmailChannel<br/>NOT IMPLEMENTED]
                SC2[SocialChannel<br/>NOT IMPLEMENTED]
            end
        end
        
        EM --> GAN
        GAN -.-> |"schedule"| SCHED
        EM --> SCHED
        RL -.-> |"policy"| DISP
        SCHED --> DISP
        DISP --> CC
        DISP --> LFC
        DISP -.-> EC2
        DISP -.-> SC2
    end

    subgraph Analysis["Analysis Layer (NOT IMPLEMENTED)"]
        METRICS[StealthMetrics]
        BENCH[Benchmarks]
        ADV[AdversarialTesting]
    end

    subgraph Output["Output Layer"]
        OUT_SEQ[/"Media Sequence<br/>[img, txt, img, ...]"/]
        OUT_MSG[/"Reconstructed Message"/]
    end

    CC --> OUT_SEQ
    LFC --> OUT_SEQ
    DM --> OUT_MSG

    %% Styling
    classDef implemented fill:#90EE90,stroke:#228B22,color:#000
    classDef notImplemented fill:#FFB6C1,stroke:#DC143C,color:#000
    classDef partial fill:#FFE4B5,stroke:#FF8C00,color:#000
    
    class FL,WL,IE,TE,UI,MI_IMG,MI_TXT,SN,SC,SE,EC,SD,DM,SCHED,DISP,CC,LFC implemented
    class AL,AE,MI_AUD,GAN,RL,EC2,SC2,METRICS,BENCH,ADV notImplemented
```

## Layer Descriptions

### 1. Corpus & Indexing Layer
- **Loaders**: Load raw data from datasets (Flickr8k images, Wikipedia text)
- **Embedders**: Generate CLIP embeddings (512-dim vectors)
- **Index**: FAISS-based similarity search with score normalization

### 2. Encoding/Decoding Engine
- **SemanticChunker**: Splits messages into semantic chunks with synonym expansion
- **SemanticEncoder**: Maps chunks to media using FAISS search
- **SemanticDecoder**: Reverses the process to reconstruct messages

### 3. Stealth & Distribution Layer
- **GANScheduler** (NOT IMPLEMENTED): Generate human-like transmission schedules
- **RLPolicyAgent** (NOT IMPLEMENTED): Adaptive decision-making based on threat level
- **Dispatcher**: Routes media to channels using configurable policies
- **Channels**: Output destinations (console, local folder, etc.)

### 4. Analysis Layer (NOT IMPLEMENTED)
- **StealthMetrics**: Measure detectability of encoded sequences
- **Benchmarks**: Accuracy, latency, capacity measurements
- **AdversarialTesting**: Test against detection algorithms

## Implementation Status

| Component | Status |
|-----------|--------|
| FlickrLoader | Implemented |
| WikipediaLoader | Implemented |
| AudioLoader | Not Implemented |
| ImageEmbedder (CLIP) | Implemented |
| TextEmbedder (CLIP) | Implemented |
| AudioEmbedder | Not Implemented |
| UnifiedSemanticIndex | Implemented |
| ScoreNormalizer | Implemented |
| SemanticChunker | Implemented |
| SemanticEncoder | Implemented |
| SemanticDecoder | Implemented |
| Scheduler | Implemented (basic) |
| Dispatcher | Implemented |
| GANScheduler | Not Implemented |
| RLPolicyAgent | Not Implemented |
| Analysis Layer | Not Implemented |

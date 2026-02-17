# DCASS Sequence Diagrams

## 1. Basic Encoding Flow

```mermaid
sequenceDiagram
    autonumber
    participant Alice as Alice (Sender)
    participant CLI as CLI
    participant Encoder as SemanticEncoder
    participant Chunker as SemanticChunker
    participant Index as UnifiedSemanticIndex
    participant Normalizer as ScoreNormalizer
    participant CLIP as ImageEmbedder (CLIP)

    Alice->>CLI: encode "Secret meeting at dawn"
    CLI->>Encoder: encode(message, modality="auto")
    
    Note over Encoder: Step 1: Chunk the message
    Encoder->>Chunker: chunk(message)
    Chunker-->>Encoder: ["secret meeting", "at dawn"]
    
    Note over Encoder: Step 2: Encode each chunk
    loop For each chunk
        Encoder->>Index: search(chunk, modality="auto", k=1)
        Index->>CLIP: encode_text([chunk])
        CLIP-->>Index: query_embedding (512-dim)
        
        Note over Index: Search ALL modalities
        Index->>Index: search text_index
        Index->>Index: search image_index
        
        Index->>Normalizer: normalize_cross_modal(results)
        Normalizer-->>Index: normalized_results
        
        Index-->>Encoder: SearchResult (best match)
    end
    
    Note over Encoder: Step 3: Build EncodedMessage
    Encoder-->>CLI: EncodedMessage
    CLI-->>Alice: Display results<br/>[image: whisper.jpg, text: "sunrise over..."]
```

## 2. Hierarchical Encoding Flow (with Synonyms)

```mermaid
sequenceDiagram
    autonumber
    participant Alice as Alice
    participant Encoder as SemanticEncoder
    participant Chunker as SemanticChunker
    participant Index as UnifiedSemanticIndex

    Alice->>Encoder: encode_hierarchical("Secret meeting at the bank")
    
    Note over Encoder: Enhanced chunking with expansions
    Encoder->>Chunker: chunk_enhanced(message)
    
    Note over Chunker: Create EnhancedChunks
    Chunker->>Chunker: normalize text
    Chunker->>Chunker: split into clauses
    
    loop For each chunk
        Chunker->>Chunker: expand_synonyms()
        Note right of Chunker: "secret" → ["hidden", "private", "quiet"]
        Chunker->>Chunker: decompose_abstract()
        Note right of Chunker: "meeting" → ["people together", "gathering"]
    end
    
    Chunker-->>Encoder: [EnhancedChunk, EnhancedChunk, ...]
    
    Note over Encoder: Search with ALL variants
    loop For each EnhancedChunk
        Encoder->>Encoder: variants = chunk.all_variants()
        Note right of Encoder: ["secret meeting", "hidden gathering",<br/>"private meeting", "people whispering"]
        
        loop For each variant
            Encoder->>Index: search(variant, modality="auto")
            Index-->>Encoder: SearchResult
        end
        
        Encoder->>Encoder: Select best result across all variants
    end
    
    Encoder-->>Alice: EncodedMessage (with metadata.matched_query)
```

## 3. Decoding Flow

```mermaid
sequenceDiagram
    autonumber
    participant Bob as Bob (Receiver)
    participant CLI as CLI
    participant Decoder as SemanticDecoder
    participant Index as UnifiedSemanticIndex

    Bob->>CLI: decode [media_sequence]
    CLI->>Decoder: decode(["img_001.jpg", "text_042", "img_103.jpg"])
    
    Note over Decoder: Build metadata lookup
    Decoder->>Decoder: _build_metadata_lookup()
    
    loop For each media_id
        Decoder->>Decoder: _decode_single(media_id)
        
        alt Found in metadata
            Decoder->>Index: lookup metadata[media_id]
            Index-->>Decoder: {caption: "sunrise over mountains", ...}
            Decoder->>Decoder: Extract semantic content
        else Not found
            Decoder->>Decoder: Return "[unknown:media_id]"
        end
    end
    
    Note over Decoder: Reconstruct message
    Decoder->>Decoder: Join semantic chunks
    Decoder-->>CLI: DecodedMessage
    CLI-->>Bob: "sunrise mountains meeting point"
```

## 4. Verification Flow

```mermaid
sequenceDiagram
    autonumber
    participant User as User
    participant Decoder as SemanticDecoder
    participant Encoder as EncodedMessage

    User->>Decoder: verify_encoding("encoded.json")
    
    Decoder->>Encoder: load("encoded.json")
    Encoder-->>Decoder: EncodedMessage
    
    Note over Decoder: Get original info
    Decoder->>Encoder: .original_message
    Encoder-->>Decoder: "Secret meeting at dawn"
    Decoder->>Encoder: .chunks
    Encoder-->>Decoder: ["secret meeting", "at dawn"]
    Decoder->>Encoder: .media_ids
    Encoder-->>Decoder: ["img_001", "text_042"]
    
    Note over Decoder: Decode the sequence
    Decoder->>Decoder: decode(media_ids)
    Decoder-->>Decoder: DecodedMessage
    
    Note over Decoder: Compare chunks
    loop For each chunk pair
        Decoder->>Decoder: _simple_similarity(original, decoded)
        Note right of Decoder: Word overlap: intersection/union
    end
    
    Decoder-->>User: {<br/>  original: "Secret meeting at dawn",<br/>  reconstructed: "whisper sunrise",<br/>  avg_match_score: 0.45<br/>}
```

## 5. Distribution Flow

```mermaid
sequenceDiagram
    autonumber
    participant Alice as Alice
    participant Scheduler as Scheduler
    participant Dispatcher as Dispatcher
    participant Channel1 as ConsoleChannel
    participant Channel2 as LocalFolderChannel

    Alice->>Scheduler: run(["img_001", "text_042", "img_103"])
    
    Note over Scheduler: Process each item with delay
    loop For each media_id (i)
        Scheduler->>Scheduler: sleep(delays[i])
        Note right of Scheduler: Wait 2.5 seconds (human-like)
        
        Scheduler->>Dispatcher: dispatch_one(media_id, i)
        
        Dispatcher->>Dispatcher: _select_channel(i)
        Note right of Dispatcher: Round-robin policy
        
        alt i % 2 == 0
            Dispatcher->>Channel1: send(media_id)
            Channel1-->>Dispatcher: {channel: "console", ...}
        else i % 2 == 1
            Dispatcher->>Channel2: send(media_id)
            Channel2-->>Dispatcher: {channel: "local_folder", ...}
        end
        
        Dispatcher-->>Scheduler: dispatch_log
    end
    
    Scheduler-->>Alice: [log1, log2, log3]
```

## 6. Full Pipeline Flow (End-to-End)

```mermaid
sequenceDiagram
    autonumber
    participant Alice as Alice
    participant Encoder as SemanticEncoder
    participant Index as UnifiedSemanticIndex
    participant Scheduler as Scheduler
    participant Dispatcher as Dispatcher
    participant Channel as Channel
    participant Bob as Bob
    participant Decoder as SemanticDecoder

    Note over Alice,Bob: === ENCODING PHASE ===
    Alice->>Encoder: encode("Secret meeting at dawn")
    Encoder->>Index: search chunks
    Index-->>Encoder: media matches
    Encoder-->>Alice: EncodedMessage<br/>[img_001, text_042, img_103]
    
    Note over Alice,Bob: === DISTRIBUTION PHASE ===
    Alice->>Scheduler: run(media_sequence, delays)
    
    loop For each media
        Scheduler->>Dispatcher: dispatch_one(media_id)
        Dispatcher->>Channel: send(media_id)
        Channel-->>Bob: Receive media
    end
    
    Note over Alice,Bob: === DECODING PHASE ===
    Bob->>Decoder: decode(received_sequence)
    Decoder->>Index: lookup metadata
    Index-->>Decoder: semantic content
    Decoder-->>Bob: DecodedMessage<br/>"sunrise gathering morning"
    
    Note over Bob: Interpret semantic meaning
```

## 7. GAN Scheduler Flow (PLANNED - NOT IMPLEMENTED)

```mermaid
sequenceDiagram
    autonumber
    participant Alice as Alice
    participant GAN as GANScheduler
    participant Gen as Generator
    participant Disc as Discriminator
    participant Scheduler as Scheduler

    Note over GAN: Training Phase (offline)
    GAN->>GAN: Load human activity data
    loop Training epochs
        GAN->>Gen: Generate fake schedule
        Gen-->>GAN: fake_schedule
        GAN->>Disc: Discriminate(real, fake)
        Disc-->>GAN: loss
        GAN->>GAN: Update weights
    end
    
    Note over Alice,Scheduler: Inference Phase
    Alice->>GAN: generate_schedule(sequence_length=5)
    GAN->>Gen: forward(noise)
    Gen-->>GAN: delays = [2.3, 5.1, 1.8, 4.2, 3.5]
    GAN-->>Alice: Human-like schedule
    
    Alice->>Scheduler: run(media_sequence, delays)
    Note right of Scheduler: Transmission looks like<br/>natural human behavior
```

## 8. RL Policy Agent Flow (PLANNED - NOT IMPLEMENTED)

```mermaid
sequenceDiagram
    autonumber
    participant System as DCASS System
    participant RL as RLPolicyAgent
    participant Monitor as StateMonitor
    participant Policy as PolicyNetwork
    participant Dispatcher as Dispatcher

    Note over System,Dispatcher: Continuous monitoring loop
    
    loop Every dispatch decision
        System->>Monitor: Get current state
        Monitor->>Monitor: Check network conditions
        Monitor->>Monitor: Analyze traffic patterns
        Monitor-->>RL: state = {threat_level: 0.7, ...}
        
        RL->>Policy: select_action(state)
        
        alt threat_level > 0.8
            Policy-->>RL: action = "pause"
            RL-->>Dispatcher: Wait for safer conditions
        else threat_level > 0.5
            Policy-->>RL: action = "slow_down"
            RL-->>Dispatcher: Increase delays
        else threat_level < 0.3
            Policy-->>RL: action = "normal"
            RL-->>Dispatcher: Proceed normally
        end
        
        Dispatcher->>Dispatcher: Execute action
        Dispatcher-->>RL: reward (stealth_score)
        RL->>Policy: update_policy(reward)
    end
```

## Sequence Diagram Summary

| Diagram | Description | Status |
|---------|-------------|--------|
| Basic Encoding | Standard encoding flow | Implemented |
| Hierarchical Encoding | Encoding with synonym expansion | Implemented |
| Decoding | Message reconstruction | Implemented |
| Verification | Round-trip verification | Implemented |
| Distribution | Multi-channel dispatch | Implemented |
| Full Pipeline | End-to-end flow | Implemented |
| GAN Scheduler | Human behavior mimicry | Not Implemented |
| RL Policy Agent | Adaptive decision making | Not Implemented |

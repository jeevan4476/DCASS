# Entity-Relationship Diagram

## Data Model

```mermaid
erDiagram
    IMAGE_INDEX ||--o{ IMAGE_METADATA : "indexes"
    TEXT_INDEX ||--o{ TEXT_METADATA : "indexes"
    AUDIO_INDEX ||--o{ AUDIO_METADATA : "indexes"
    
    IMAGE_METADATA ||--o{ TEXT_METADATA : "has captions"
    
    IMAGE_METADATA {
        string id PK "flickr8k_00000"
        string content "path to image file"
        string caption "primary caption"
        array captions "all 5 captions"
        string modality "image"
    }
    
    TEXT_METADATA {
        string id PK "flickr8k_00000_cap0"
        string text "caption or sentence"
        string content "same as text"
        string source "flickr8k or wikipedia"
        string image_id FK "links to image"
        string article "wikipedia article (if wiki)"
        string modality "text"
    }
    
    AUDIO_METADATA {
        string id PK "audio_00000"
        string content "path to audio file"
        string description "audio description"
        string transcript "speech transcript"
        string source "dataset name"
        string modality "audio"
    }
    
    IMAGE_INDEX {
        binary vectors "FAISS IndexFlatIP"
        int dimension "512 (CLIP)"
        int count "number of vectors"
    }
    
    TEXT_INDEX {
        binary vectors "FAISS IndexFlatIP"
        int dimension "512 (CLIP)"
        int count "number of vectors"
    }
    
    AUDIO_INDEX {
        binary vectors "FAISS IndexFlatIP"
        int dimension "512 (CLAP)"
        int count "number of vectors"
    }
```

## Encoding Result Structure

```mermaid
erDiagram
    ENCODING_RESULT ||--|{ SEMANTIC_CHUNK : contains
    ENCODING_RESULT ||--|{ ENCODED_CHUNK : contains
    ENCODED_CHUNK ||--|| SEMANTIC_CHUNK : "encodes"
    ENCODED_CHUNK ||--|| MEDIA_ITEM : "maps to"
    ENCODED_CHUNK ||--o{ MEDIA_ITEM : "alternatives"
    
    ENCODING_RESULT {
        string original_message
        list chunks
        list encoded
    }
    
    SEMANTIC_CHUNK {
        string text "expanded query"
        string original "original text"
        int index "position"
    }
    
    ENCODED_CHUNK {
        SemanticChunk chunk
        MediaItem media
        list alternatives
    }
    
    MEDIA_ITEM {
        string id
        string modality
        string content
        float score
        float normalized_score
        dict metadata
    }
```

## Decoding Result Structure

```mermaid
erDiagram
    DECODING_RESULT ||--|{ DECODED_ITEM : contains
    
    DECODING_RESULT {
        list media_ids "input IDs"
        list decoded "decoded items"
        bool all_verified "computed"
        float verification_rate "computed"
        string reconstructed_meaning "computed"
    }
    
    DECODED_ITEM {
        string media_id
        string modality "nullable"
        string content "extracted"
        bool verified
        dict metadata
    }
```

## File System Structure

```mermaid
erDiagram
    PROJECT_ROOT ||--|| DATA : contains
    PROJECT_ROOT ||--|| SRC : contains
    PROJECT_ROOT ||--|| DOCS : contains
    
    DATA ||--|| RAW : contains
    DATA ||--|| INDICES : contains
    
    RAW ||--|| FLICKR8K : contains
    RAW ||--|| WIKIPEDIA : contains
    
    INDICES ||--|| IMAGE_FILES : contains
    INDICES ||--|| TEXT_FILES : contains
    INDICES ||--|| AUDIO_FILES : contains
    
    FLICKR8K {
        folder images "*.jpg files"
        file Flickr8k_token_txt "captions"
    }
    
    WIKIPEDIA {
        file sentences_json "extracted sentences"
    }
    
    IMAGE_FILES {
        file image_index "FAISS binary"
        file image_metadata_json "JSON metadata"
    }
    
    TEXT_FILES {
        file text_index "FAISS binary"
        file text_metadata_json "JSON metadata"
    }
    
    AUDIO_FILES {
        file audio_index "FAISS binary"
        file audio_metadata_json "JSON metadata"
    }
```

## Metadata JSON Schemas

### Image Metadata
```json
{
  "id": "flickr8k_00000",
  "content": "data/raw/flickr8k/images/flickr8k_00000.jpg",
  "caption": "A black dog is running after a white dog in the snow.",
  "captions": [
    "A black dog is running after a white dog in the snow.",
    "Black dog chasing brown dog through snow",
    "Two dogs chase each other across the snowy ground.",
    "Two dogs play together in the snow.",
    "Two dogs running through a low lying body of water."
  ],
  "modality": "image"
}
```

### Text Metadata (Flickr Caption)
```json
{
  "id": "flickr8k_00000_cap0",
  "text": "A black dog is running after a white dog in the snow.",
  "content": "A black dog is running after a white dog in the snow.",
  "source": "flickr8k",
  "image_id": "flickr8k_00000",
  "modality": "text"
}
```

### Text Metadata (Wikipedia)
```json
{
  "id": "wiki_000000",
  "text": "April is the fourth month of the year.",
  "content": "April is the fourth month of the year.",
  "source": "wikipedia",
  "article": "April",
  "modality": "text"
}
```

### Audio Metadata (Planned)
```json
{
  "id": "audio_00000",
  "content": "data/raw/audio/audio_00000.wav",
  "description": "A dog barking loudly",
  "transcript": "Woof woof woof",
  "source": "libretta",
  "modality": "audio"
}
```

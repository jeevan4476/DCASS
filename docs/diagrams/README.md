# DCASS Architecture Diagrams

This directory contains Mermaid diagrams documenting the DCASS system architecture for Phase 3 Review.

## Quick Reference

| Diagram | Description |
|---------|-------------|
| [architecture.md](architecture.md) | High-level system architecture (4 layers) |
| [class_diagram.md](class_diagram.md) | Class relationships and dependencies |
| [sequence_encode.md](sequence_encode.md) | Encoding flow sequence |
| [sequence_decode.md](sequence_decode.md) | Decoding flow sequence |
| [sequence_distribute.md](sequence_distribute.md) | Distribution pipeline flow |
| [er_diagram.md](er_diagram.md) | Data entity relationships |

## How to View

These diagrams use [Mermaid](https://mermaid.js.org/) syntax. You can view them:

1. **GitHub/GitLab**: Renders automatically in markdown preview
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Paste into [Mermaid Live Editor](https://mermaid.live/)

## Architecture Overview

DCASS (Dynamic Context-Aware Semantic Steganography) encodes secret messages by selecting sequences of **unmodified media** from a corpus, making it resistant to traditional steganalysis.

```
Message → Chunker → Encoder → Media Sequence → Distribution → Channels
                                    ↓
                              Receiver
                                    ↓
                    Media IDs → Decoder → Reconstructed Meaning
```

# DCASS System Diagrams

This directory contains all Mermaid diagrams for the DCASS (Dynamic Context-Aware Semantic Steganography) system, created for **Phase 3 Review**.

## Diagram Index

| Diagram | File | Description |
|---------|------|-------------|
| Architecture | [architecture.md](./architecture.md) | Full system architecture with all layers |
| Class Diagram | [class_diagram.md](./class_diagram.md) | All classes, attributes, methods, relationships |
| ER Diagram | [er_diagram.md](./er_diagram.md) | Data entities, relationships, storage mapping |
| Use Cases | [use_cases.md](./use_cases.md) | Actors and use cases with implementation status |
| Sequences | [sequences.md](./sequences.md) | Sequence diagrams for all major flows |

## How to View

These diagrams use **Mermaid** syntax. To render them:

1. **GitHub**: Mermaid renders automatically in markdown preview
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)
4. **CLI**: Use `mmdc` (Mermaid CLI) to export as PNG/SVG

## Implementation Status Legend

| Color | Meaning |
|-------|---------|
| Green | Implemented |
| Red/Pink | Not Implemented |
| Orange | Partial/Basic |

## Quick Links

### Core Flows
- [Basic Encoding](./sequences.md#1-basic-encoding-flow)
- [Hierarchical Encoding](./sequences.md#2-hierarchical-encoding-flow-with-synonyms)
- [Decoding](./sequences.md#3-decoding-flow)
- [Full Pipeline](./sequences.md#6-full-pipeline-flow-end-to-end)

### Planned Features
- [GAN Scheduler](./sequences.md#7-gan-scheduler-flow-planned---not-implemented)
- [RL Policy Agent](./sequences.md#8-rl-policy-agent-flow-planned---not-implemented)

## Summary Statistics

| Category | Total | Implemented | Not Implemented |
|----------|-------|-------------|-----------------|
| Classes | 20 | 16 | 4 |
| Use Cases | 23 | 16 | 7 |
| Sequence Flows | 8 | 6 | 2 |

---

*Generated for DCASS Capstone Project - Phase 3 Review*

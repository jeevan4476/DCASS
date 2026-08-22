# src/engine/__init__.py
"""
DCASS Engine - Core encoding and decoding logic.

This module provides the main steganographic encoding and decoding
capabilities for the DCASS system.

Components:
- SemanticEncoder: Encodes messages into media sequences
- SemanticDecoder: Decodes media sequences back to semantic meaning
- SemanticChunker: Splits messages into semantic units
"""

from .encoder import SemanticEncoder, EncodingResult, EncodedChunk, encode_message
from .decoder import SemanticDecoder, DecodingResult, DecodedItem, decode_media_sequence
from .chunker import SemanticChunker, SemanticChunk, chunk_message
from .vcp_payload import VCPPayloadMapper, PayloadCarrier

__all__ = [
    # Encoder
    "SemanticEncoder",
    "EncodingResult",
    "EncodedChunk",
    "encode_message",

    # Decoder
    "SemanticDecoder",
    "DecodingResult",
    "DecodedItem",
    "decode_media_sequence",

    # Chunker
    "SemanticChunker",
    "SemanticChunk",
    "chunk_message",

    # Exact payload mapping
    "VCPPayloadMapper",
    "PayloadCarrier",
]

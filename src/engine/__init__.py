"""
DCASS Engine Package - Encoding and Decoding.

This package contains the core encoding/decoding logic for DCASS.

Main components:
- SemanticEncoder: Encodes messages into media sequences
- SemanticDecoder: Decodes media sequences back to semantic meaning
- SemanticChunker: Splits messages into semantic chunks
"""

from src.engine.encoder import SemanticEncoder, EncodedMessage, encode_message
from src.engine.decoder import SemanticDecoder, DecodedMessage, decode_sequence
from src.engine.chunker import SemanticChunker

__all__ = [
    "SemanticEncoder",
    "EncodedMessage", 
    "encode_message",
    "SemanticDecoder",
    "DecodedMessage",
    "decode_sequence",
    "SemanticChunker",
]

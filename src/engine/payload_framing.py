# src/engine/payload_framing.py
"""
Framed payload for the exact_vcp channel (Decision 4 / WP-6).

Layout inside the RS-protected region:

    [1 byte  version/flags ]   0x01 = framed v1
    [2 bytes plaintext len ]   big-endian, UTF-8 byte length
    [2 bytes CRC-16 of plaintext ]  CRC-CCITT (poly 0x1021, init 0xFFFF)
    [N bytes plaintext     ]

Cost: 5 extra carriers per message. Gains: an explicit length (no reliance on
trailing bytes), a version field for future format changes, and - critically -
an integrity check independent of Reed-Solomon. The decoder's epoch search
accepts a candidate epoch only when RS succeeds AND the CRC verifies, which
makes wrong-key/wrong-epoch acceptance negligible instead of merely unlikely.

Legacy compatibility: unframed (raw) payloads never start with version byte
0x01 followed by plausible framing, so the decoder sniffs the version byte:
0x01 -> parse frame; anything else -> treat entire codeword as raw UTF-8.
"""

from __future__ import annotations

import struct

FRAME_VERSION = 0x01
HEADER_LEN = 5  # version(1) + length(2) + crc16(2)


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE. Deterministic, dependency-free integrity check."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def frame_payload(message: str) -> bytes:
    """
    Encode a plaintext string into the framed wire format.
    Raises ValueError if the message exceeds the 2-byte length field.
    """
    raw = message.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ValueError(f"message too long for framed channel: {len(raw)} > 65535 bytes")
    head = struct.pack(">BH", FRAME_VERSION, len(raw))
    crc = crc16_ccitt(raw)
    return head + bytes([(crc >> 8) & 0xFF, crc & 0xFF]) + raw


class FrameError(Exception):
    """Raised when a codeword is not validly framed."""


def unframe_payload(codeword: bytes) -> tuple[str, bool]:
    """
    Decode a post-RS codeword into (plaintext, was_framed).

    Returns (text, True) when the version byte and CRC verify.
    Falls back to legacy raw decoding (text, False) when the first byte is
    not the frame version marker. Raises FrameError only when the payload
    CLAIMS to be framed but fails validation - that indicates corruption or
    a wrong epoch and must not be silently accepted.
    """
    if not codeword:
        raise FrameError("empty payload")
    if codeword[0] != FRAME_VERSION:
        # Legacy unframed transport
        return codeword.decode("utf-8", errors="replace"), False

    if len(codeword) < HEADER_LEN:
        raise FrameError("framed payload shorter than header")
    version, length = struct.unpack(">BH", codeword[:3])
    crc_received = (codeword[3] << 8) | codeword[4]
    body = codeword[HEADER_LEN : HEADER_LEN + length]
    if len(body) != length:
        raise FrameError(f"truncated frame: expected {length} bytes, got {len(body)}")
    if crc16_ccitt(body) != crc_received:
        raise FrameError(
            f"CRC mismatch: computed {crc16_ccitt(body):04x} != "
            f"received {crc_received:04x} (wrong key or corrupted beyond RS capacity)"
        )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise FrameError(f"framed body is not valid UTF-8: {e}") from e
    return text, True


def is_probably_framed(codeword: bytes) -> bool:
    """Cheap sniff without full validation."""
    return bool(codeword) and codeword[0] == FRAME_VERSION

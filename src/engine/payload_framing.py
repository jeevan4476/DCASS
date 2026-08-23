# src/engine/payload_framing.py
"""
Framed payload for the exact_vcp channel (Decision 4 / WP-6).

Layout inside the RS-protected region:

CRC frame (version 0x01) — integrity / corruption detection only:
    [1 byte  version/flags ]   0x01 = framed v1
    [2 bytes plaintext len ]   big-endian, UTF-8 byte length
    [2 bytes CRC-16 of plaintext ]  CRC-CCITT (poly 0x1021, init 0xFFFF)
    [N bytes plaintext     ]

HMAC frame (version 0x02) — when a context secret is present:
    [1 byte  version/flags ]   0x02 = HMAC-framed v1
    [2 bytes plaintext len ]   big-endian UTF-8 length
    [32 bytes HMAC-SHA256(secret, plaintext)]
    [N bytes plaintext     ]

Cost: 5 extra carriers (CRC) or 35 (HMAC) per message. Gains: an explicit
length, a version field, and an integrity check independent of Reed-Solomon.

Honest scope: CRC-16 is NOT authentication (~2^-16 false accept). The
decoder's epoch search uses CRC to reject wrong epochs with high probability
but not cryptographic certainty. When a shared secret is available, use the
HMAC frame so wrong-key/wrong-epoch acceptance is negligible.

Legacy compatibility: unframed (raw) payloads never start with version byte
0x01/0x02 followed by plausible framing, so the decoder sniffs the version:
0x01/0x02 -> parse frame; anything else -> treat entire codeword as raw UTF-8.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Optional

FRAME_VERSION = 0x01
FRAME_VERSION_HMAC = 0x02
HEADER_LEN = 5  # version(1) + length(2) + crc16(2)
HMAC_LEN = 32
HMAC_HEADER_LEN = 1 + 2 + HMAC_LEN  # version + length + tag


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


def frame_payload(message: str, secret: Optional[bytes] = None) -> bytes:
    """
    Encode a plaintext string into the framed wire format.

    When *secret* is set, emit an HMAC-authenticated frame (version 0x02).
    Otherwise emit a CRC-16 integrity frame (version 0x01).

    Raises ValueError if the message exceeds the 2-byte length field.
    """
    raw = message.encode("utf-8")
    if len(raw) > 0xFFFF:
        raise ValueError(f"message too long for framed channel: {len(raw)} > 65535 bytes")
    if secret:
        tag = hmac.new(secret, raw, hashlib.sha256).digest()
        head = struct.pack(">BH", FRAME_VERSION_HMAC, len(raw))
        return head + tag + raw
    head = struct.pack(">BH", FRAME_VERSION, len(raw))
    crc = crc16_ccitt(raw)
    return head + bytes([(crc >> 8) & 0xFF, crc & 0xFF]) + raw


class FrameError(Exception):
    """Raised when a codeword is not validly framed."""


def unframe_payload(codeword: bytes, secret: Optional[bytes] = None) -> tuple[str, bool]:
    """
    Decode a post-RS codeword into (plaintext, was_framed).

    Returns (text, True) when the version byte and integrity/auth check verify.
    Falls back to legacy raw decoding (text, False) when the first byte is
    not a frame version marker. Raises FrameError only when the payload
    CLAIMS to be framed but fails validation - that indicates corruption or
    a wrong epoch and must not be silently accepted.
    """
    if not codeword:
        raise FrameError("empty payload")

    if codeword[0] == FRAME_VERSION_HMAC:
        if not secret:
            raise FrameError("HMAC-framed payload requires secret")
        if len(codeword) < HMAC_HEADER_LEN:
            raise FrameError("HMAC-framed payload shorter than header")
        _version, length = struct.unpack(">BH", codeword[:3])
        tag = codeword[3 : 3 + HMAC_LEN]
        body = codeword[HMAC_HEADER_LEN : HMAC_HEADER_LEN + length]
        if len(body) != length:
            raise FrameError(f"truncated HMAC frame: expected {length} bytes, got {len(body)}")
        expected = hmac.new(secret, body, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise FrameError("HMAC mismatch (wrong key or corrupted beyond RS capacity)")
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            raise FrameError(f"framed body is not valid UTF-8: {e}") from e
        return text, True

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
    return bool(codeword) and codeword[0] in (FRAME_VERSION, FRAME_VERSION_HMAC)

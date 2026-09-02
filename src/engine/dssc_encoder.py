"""
DSSC Encoder - Dynamic Semantic State-Space Coding Encoder.

Encodes messages by:
1. Framing payload with CRC integrity and optional secret HMAC
2. Protecting payload with Reed-Solomon Error Correction Code (RS-ECC)
3. Decomposing message into semantic chunks to determine admissible semantic families
4. Building session-keyed dynamic carrier state spaces inside those families
5. Encoding exact payload bits into carrier selections
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Optional

from src.corpus.index.unified_index import UnifiedSemanticIndex, MediaItem, Modality
from src.engine.chunker import SemanticChunker, SemanticChunk
from src.engine.dssc_state_space import (
    DSSCStateSpace,
    SemanticFamilyManager,
    derive_session_permutation,
)
from src.engine.vcp_payload import VCPPayloadMapper
from src.engine.ecc import RSErrorCorrection
from src.engine.payload_framing import frame_payload


@dataclass
class DSSCEncodedCarrier:
    """A carrier item selected by DSSC."""
    media_id: str
    media: Optional[MediaItem]
    symbol: int
    bits: int
    family: str
    chunk_index: int


@dataclass
class DSSCEncodingResult:
    """Result of encoding a message with DSSC."""
    original_message: str
    carrier_ids: list[str]
    encoded_carriers: list[DSSCEncodedCarrier]
    total_bits: int
    parity_bytes: int
    session_key_id: str

    @property
    def media_sequence(self) -> list[MediaItem]:
        return [c.media for c in self.encoded_carriers if c.media is not None]

    @property
    def bits_per_carrier_avg(self) -> float:
        if not self.encoded_carriers:
            return 0.0
        return self.total_bits / len(self.encoded_carriers)


class BitStreamReader:
    """Helper to consume variable numbers of bits from a byte sequence."""
    def __init__(self, data: bytes):
        self.data = data
        self.total_bits = len(data) * 8
        self.bit_offset = 0

    @property
    def has_bits(self) -> bool:
        return self.bit_offset < self.total_bits

    @property
    def remaining_bits(self) -> int:
        return max(0, self.total_bits - self.bit_offset)

    def read_bits(self, n_bits: int) -> int:
        """Read n_bits as an integer symbol. Zero-pads if past end."""
        if n_bits <= 0:
            return 0

        val = 0
        for _ in range(n_bits):
            if self.bit_offset < self.total_bits:
                byte_idx = self.bit_offset // 8
                bit_in_byte = 7 - (self.bit_offset % 8)
                bit = (self.data[byte_idx] >> bit_in_byte) & 1
                val = (val << 1) | bit
                self.bit_offset += 1
            else:
                val = val << 1
                self.bit_offset += 1
        return val


class DSSCEncoder:
    """
    Dynamic Semantic State-Space Coding Encoder.
    """
    def __init__(
        self,
        index: UnifiedSemanticIndex,
        chunker: SemanticChunker = None,
        family_manager: SemanticFamilyManager = None,
        vcp_mapper: VCPPayloadMapper = None,
    ):
        self.index = index
        self.chunker = chunker or SemanticChunker()
        self.family_manager = family_manager or SemanticFamilyManager()
        self.vcp_mapper = vcp_mapper or VCPPayloadMapper(index)

    def encode(
        self,
        message: str,
        session_key: bytes,
        ecc_parity_bytes: int = 8,
        modalities: list[Modality] = None,
    ) -> DSSCEncodingResult:
        if not message:
            raise ValueError("Message cannot be empty")
        if not session_key:
            raise ValueError("Session key required for DSSC")

        # 1. Framing and Reed-Solomon error correction
        framed = frame_payload(message)
        rs = RSErrorCorrection(parity_bytes=ecc_parity_bytes)
        codeword = rs.encode(framed)
        bitstream = BitStreamReader(codeword)
        total_bits = len(codeword) * 8

        # 2. Semantic decomposition
        chunks = self.chunker.chunk(message)
        if not chunks:
            chunks = [SemanticChunk(text=message, original=message, index=0)]

        # Collect candidate pool from index
        all_ids = []
        for mod, meta_list in self.index.metadata.items():
            if modalities and mod not in modalities:
                continue
            for m in meta_list:
                mid = m.get("id")
                if mid:
                    all_ids.append(mid)

        all_ids.sort()
        if len(all_ids) < 16:
            raise RuntimeError(f"Corpus too small for DSSC ({len(all_ids)} items available)")

        encoded_carriers: list[DSSCEncodedCarrier] = []
        chunk_idx = 0

        while bitstream.has_bits:
            # Deterministic, HMAC-keyed family selection (topic-safe: never derives from message text)
            family_digest = hmac.new(
                session_key,
                f"family:{chunk_idx}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
            family_idx = int.from_bytes(family_digest[:4], "big") % len(self.family_manager.families)
            primary_family = self.family_manager.families[family_idx]

            # Build candidate set: all corpus IDs whose VCP cluster falls in
            # the family's cluster range. This replaces the modulo-hash partition.
            allowed_clusters = set(primary_family.cluster_ids)
            candidates = [
                cid for cid in all_ids
                if self.vcp_mapper.symbol_for_media_id(cid) in allowed_clusters
            ]
            if len(candidates) < 8:
                candidates = all_ids[:256]

            # Capacity
            bits_per_carrier = max(2, min(16, int(math.floor(math.log2(len(candidates))))))
            perm = derive_session_permutation(
                len(candidates),
                session_key,
                context_salt=f"dssc:{chunk_idx}:{primary_family.name}",
            )

            state_space = DSSCStateSpace(
                chunk_index=chunk_idx,
                family_name=primary_family.name,
                candidate_media_ids=candidates,
                permuted_indices=perm,
                bits_per_carrier=bits_per_carrier,
            )

            # Consume bits
            symbol = bitstream.read_bits(bits_per_carrier)
            selected_id = state_space.symbol_to_media_id(symbol)
            media_item = self.index.get_by_id(selected_id)

            encoded_carriers.append(
                DSSCEncodedCarrier(
                    media_id=selected_id,
                    media=media_item,
                    symbol=symbol,
                    bits=bits_per_carrier,
                    family=primary_family.name,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

        key_id = hashlib.sha256(session_key).hexdigest()[:8]

        return DSSCEncodingResult(
            original_message=message,
            carrier_ids=[c.media_id for c in encoded_carriers],
            encoded_carriers=encoded_carriers,
            total_bits=total_bits,
            parity_bytes=ecc_parity_bytes,
            session_key_id=key_id,
        )

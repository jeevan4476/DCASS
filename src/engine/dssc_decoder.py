"""
DSSC Decoder - Dynamic Semantic State-Space Coding Decoder.

Decodes carrier sequences by:
1. Reconstructing deterministic candidate state spaces using session key and corpus index
2. Inverting the dynamic session permutation to recover payload symbols
3. Reconstructing the protected bitstream
4. Applying Reed-Solomon Error Correction to repair any corrupted carrier symbols
5. Validating CRC integrity frame to ensure 100% exact message reconstruction
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field
from typing import Optional

from src.corpus.index.unified_index import UnifiedSemanticIndex, Modality
from src.engine.dssc_state_space import (
    DSSCStateSpace,
    SemanticFamilyManager,
    derive_session_permutation,
)
from src.engine.vcp_payload import VCPPayloadMapper
from src.engine.ecc import RSErrorCorrection
from src.engine.payload_framing import unframe_payload, FrameError


@dataclass
class DSSCDecodingResult:
    """Result of decoding a carrier sequence with DSSC."""
    media_ids: list[str]
    reconstructed_message: Optional[str]
    success: bool
    verification_rate: float
    ecc_fixed_errors: list[int] = field(default_factory=list)
    raw_payload_bytes: list[int] = field(default_factory=list)
    total_bits_recovered: int = 0
    parity_bytes: int = 0


class BitStreamWriter:
    """Helper to assemble a byte sequence from variable-width bit symbols."""
    def __init__(self):
        self.bits: list[int] = []

    def append_bits(self, value: int, n_bits: int):
        """Append n_bits of integer value to bit list (MSB first)."""
        for i in range(n_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.bits.append(bit)

    def to_bytes(self) -> bytes:
        """Convert accumulated bits to bytes (trims trailing partial bits)."""
        num_bytes = len(self.bits) // 8
        out = bytearray(num_bytes)
        for byte_idx in range(num_bytes):
            byte_val = 0
            for b in range(8):
                byte_val = (byte_val << 1) | self.bits[byte_idx * 8 + b]
            out[byte_idx] = byte_val
        return bytes(out)


class DSSCDecoder:
    """
    Dynamic Semantic State-Space Coding Decoder.
    """
    def __init__(
        self,
        index: UnifiedSemanticIndex,
        vcp_mapper: VCPPayloadMapper = None,
        family_manager: SemanticFamilyManager = None,
    ):
        self.index = index
        self.vcp_mapper = vcp_mapper or VCPPayloadMapper(index)
        self.family_manager = family_manager or SemanticFamilyManager()

    def decode(
        self,
        carrier_ids: list[str],
        session_key: bytes,
        ecc_parity_bytes: int = 8,
        modalities: list[Modality] = None,
    ) -> DSSCDecodingResult:
        if not carrier_ids:
            return DSSCDecodingResult(
                media_ids=[],
                reconstructed_message=None,
                success=False,
                verification_rate=0.0,
            )

        # 1. Collect identical canonical candidate pool from index
        all_ids = []
        for mod, meta_list in self.index.metadata.items():
            if modalities and mod not in modalities:
                continue
            for m in meta_list:
                mid = m.get("id")
                if mid:
                    all_ids.append(mid)

        all_ids.sort()
        bit_writer = BitStreamWriter()

        verified_count = 0

        for chunk_idx, carrier_id in enumerate(carrier_ids):
            # Check if item exists in corpus
            item = self.index.get_by_id(carrier_id)
            if item is not None:
                verified_count += 1

            # Deterministic, HMAC-keyed family selection (identically matches encoder logic)
            family_digest = hmac.new(
                session_key,
                f"family:{chunk_idx}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
            family_idx = int.from_bytes(family_digest[:4], "big") % len(self.family_manager.families)
            primary_family = self.family_manager.families[family_idx]

            # Candidate set: all IDs whose VCP cluster in family range (mirrors encoder)
            allowed_clusters = set(primary_family.cluster_ids)
            candidates = [
                cid for cid in all_ids
                if self.vcp_mapper.symbol_for_media_id(cid) in allowed_clusters
            ]
            if len(candidates) < 8:
                candidates = all_ids[:256]

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

            symbol = state_space.media_id_to_symbol(carrier_id)
            if symbol is None:
                # Symbol error / missing carrier
                symbol = 0

            bit_writer.append_bits(symbol, bits_per_carrier)

        codeword = bit_writer.to_bytes()
        rs = RSErrorCorrection(parity_bytes=ecc_parity_bytes)

        fixed_errors = []
        data, ok, fixed = rs.decode_bytes(codeword)
        if ok:
            fixed_errors = fixed

        reconstructed_text = None
        success = False
        if ok:
            try:
                reconstructed_text, _ = unframe_payload(data)
                success = True
            except FrameError:
                # Corruption beyond RS capacity
                success = False

        ver_rate = verified_count / len(carrier_ids) if carrier_ids else 0.0

        return DSSCDecodingResult(
            media_ids=carrier_ids,
            reconstructed_message=reconstructed_text,
            success=success,
            ecc_fixed_errors=fixed_errors,
            total_bits_recovered=len(bit_writer.bits),
            verification_rate=ver_rate,
        )

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
from typing import Optional, Literal

from src.corpus.index.unified_index import UnifiedSemanticIndex, Modality
from src.engine.dssc_state_space import (
    DSSCStateSpace,
    MODALITY_FAMILY_MAP,
    SemanticFamilyManager,
    adaptive_ecc_parity,
    derive_session_permutation,
)
from src.engine.vcp_payload import VCPPayloadMapper
from src.engine.ecc import RSErrorCorrection
from src.engine.payload_framing import HEADER_LEN, unframe_payload, FrameError


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
        carrier_ids: list[Optional[str]],
        session_key: bytes,
        ecc_parity_bytes: Optional[int] = 8,
        modalities: list[Modality] = None,
        diversity_mode: Literal["best", "round_robin", "balanced"] = "round_robin",
        erasure_positions: Optional[list[int]] = None,
    ) -> DSSCDecodingResult:
        if not carrier_ids:
            return DSSCDecodingResult(
                media_ids=[],
                reconstructed_message=None,
                success=False,
                verification_rate=0.0,
            )

        erased_carriers: set[int] = set()
        if erasure_positions:
            for pos in erasure_positions:
                idx = int(pos)
                if 0 <= idx < len(carrier_ids):
                    erased_carriers.add(idx)
        for idx, cid in enumerate(carrier_ids):
            if cid is None or cid == "":
                erased_carriers.add(idx)

        # 1. Collect identical canonical candidate pool from index
        all_ids_by_modality: dict[Modality, list[str]] = {}
        for mod, meta_list in self.index.metadata.items():
            if modalities and mod not in modalities:
                continue
            ids = []
            for m in meta_list:
                mid = m.get("id")
                if mid:
                    ids.append(mid)
            ids.sort()
            all_ids_by_modality[mod] = ids

        # Combined sorted list for fallback
        all_ids = []
        for ids in all_ids_by_modality.values():
            all_ids.extend(ids)
        all_ids.sort()

        # Available modalities present in the index
        available_modalities = [m for m in ["text", "image", "audio"] if all_ids_by_modality.get(m)]
        if not available_modalities:
            available_modalities = [m for m in all_ids_by_modality.keys() if all_ids_by_modality[m]]
        if not available_modalities:
            available_modalities = ["image"]

        # MODALITY_FAMILY_MAP is imported from dssc_state_space (single source
        # of truth shared with the encoder) so the mapping can never diverge.

        bit_writer = BitStreamWriter()
        erased_byte_positions: set[int] = set()
        # Total bits appended BEFORE the final carrier (0 when M <= 1). Used
        # below to bound the true codeword byte length L against phantom
        # trailing zero-padding bytes (see the phantom-byte guard).
        prev_bit_count = 0

        verified_count = 0

        candidate_pool_cache: dict[tuple[str, str], list[str]] = {}
        modality_usage: dict[str, int] = {m: 0 for m in available_modalities}
        normalized_carrier_ids: list[str] = []

        for chunk_idx, carrier_id in enumerate(carrier_ids):
            is_erased = chunk_idx in erased_carriers
            normalized_carrier_ids.append(carrier_id if carrier_id else "[DROPPED]")

            # Check if item exists in corpus (dropped slots do not count as verified)
            if not is_erased and carrier_id:
                item = self.index.get_by_id(carrier_id)
                if item is not None:
                    verified_count += 1

            # Deterministic, HMAC-keyed family selection (identically matches encoder logic)
            family_digest = hmac.new(
                session_key,
                f"family:{chunk_idx}".encode("utf-8"),
                hashlib.sha256,
            ).digest()

            # Modality selection governed by diversity_mode (matches encoder)
            if diversity_mode == "round_robin":
                modality_for_chunk = available_modalities[chunk_idx % len(available_modalities)]
            elif diversity_mode == "balanced":
                modality_for_chunk = min(
                    available_modalities,
                    key=lambda m: (modality_usage.get(m, 0), available_modalities.index(m)),
                )
            elif diversity_mode == "best":
                modality_for_chunk = available_modalities[0]
            else:
                modality_for_chunk = available_modalities[chunk_idx % len(available_modalities)]

            modality_usage[modality_for_chunk] = modality_usage.get(modality_for_chunk, 0) + 1
            modality_ids = all_ids_by_modality.get(modality_for_chunk, [])

            # Modality-conditioned family selection: pick from families native to this modality
            native_names = MODALITY_FAMILY_MAP.get(modality_for_chunk, [])
            matching_families = [fam for fam in self.family_manager.families if fam.name in native_names]
            if matching_families:
                admissible_families = matching_families
            else:
                admissible_families = self.family_manager.families

            family_idx = int.from_bytes(family_digest[:4], "big") % len(admissible_families)
            primary_family = admissible_families[family_idx]

            # Candidate set: all IDs whose VCP cluster in family range (cached per pool)
            cache_key = (modality_for_chunk, primary_family.name)
            if cache_key not in candidate_pool_cache:
                allowed_clusters = set(primary_family.cluster_ids)
                cand = [
                    cid for cid in modality_ids
                    if self.vcp_mapper.symbol_for_media_id(cid) in allowed_clusters
                ]
                if len(cand) < 8:
                    cand = [
                        cid for cid in all_ids
                        if self.vcp_mapper.symbol_for_media_id(cid) in allowed_clusters
                    ]
                if len(cand) < 8:
                    cand = all_ids[:256]  # Ultimate safety fallback
                candidate_pool_cache[cache_key] = cand

            candidates = candidate_pool_cache[cache_key]

            bits_per_carrier = max(2, min(16, int(math.floor(math.log2(len(candidates))))))
            bit_start = len(bit_writer.bits)
            prev_bit_count = bit_start

            if is_erased:
                symbol = 0
                bit_end = bit_start + bits_per_carrier - 1
                for byte_idx in range(bit_start // 8, (bit_end // 8) + 1):
                    erased_byte_positions.add(byte_idx)
            else:
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

        total_bits = len(bit_writer.bits)
        raw_bytes = bit_writer.to_bytes()
        max_len = len(raw_bytes)
        ver_rate = verified_count / len(carrier_ids) if carrier_ids else 0.0
        fallback_parity = ecc_parity_bytes if ecc_parity_bytes is not None else 0

        if max_len == 0:
            # Fail closed: no codeword bytes at all.
            return DSSCDecodingResult(
                media_ids=normalized_carrier_ids,
                reconstructed_message=None,
                success=False,
                ecc_fixed_errors=[],
                raw_payload_bytes=[],
                total_bits_recovered=total_bits,
                parity_bytes=fallback_parity,
                verification_rate=ver_rate,
            )

        # Phantom-byte guard (bits_per_carrier > 8): BitStreamReader-style
        # zero-padding means the final carrier can append >= 8 padding bits,
        # which makes BitStreamWriter.to_bytes() emit an extra trailing 0x00
        # byte at index L beyond the encoded RS codeword length L. The encoder
        # stops at `while bitstream.has_bits`, so the true codeword byte length
        # L must satisfy prev_bit_count < 8 * L <= total_bits, i.e.
        # L in [min_len, max_len] with min_len = min(max_len, prev_bit_count//8 + 1)
        # (at most 2 candidate lengths for bpc <= 16). Candidates are tried
        # shortest first, so the true L is preferred over the phantom L + 1.
        min_len = min(max_len, (prev_bit_count // 8) + 1)
        candidate_lengths = list(range(min_len, max_len + 1))

        reconstructed_text: Optional[str] = None
        success = False
        fixed_errors: list[int] = []
        accepted_codeword = raw_bytes[:min_len]
        accepted_parity = fallback_parity

        for cand_len in candidate_lengths:
            cw_cand = raw_bytes[:cand_len]
            sorted_byte_erasures = sorted(
                b for b in erased_byte_positions if 0 <= b < cand_len
            )
            if ecc_parity_bytes is not None:
                cand_parities = [ecc_parity_bytes]
            else:
                # Adaptive parity (encoder called with ecc_parity_bytes=None):
                # infer every R for which some framed length L_framed >= HEADER_LEN
                # satisfies L_framed + adaptive_ecc_parity(L_framed) == cand_len.
                cand_parities = sorted(
                    {
                        cand_len - l_framed
                        for l_framed in range(HEADER_LEN, cand_len)
                        if l_framed + adaptive_ecc_parity(l_framed) == cand_len
                    }
                )
            for cand_parity in cand_parities:
                if cand_parity > 0:
                    rs = RSErrorCorrection(parity_bytes=cand_parity)
                    if sorted_byte_erasures:
                        data, ok, fixed = rs.decode_bytes(
                            cw_cand, erase_pos=sorted_byte_erasures
                        )
                    else:
                        data, ok, fixed = rs.decode_bytes(cw_cand)
                    if not ok:
                        continue
                else:
                    data = cw_cand
                    fixed = []
                try:
                    # unframe_payload validates the CRC-16 integrity frame. DSSC
                    # frames are NOT HMAC-authenticated: the encoder calls
                    # frame_payload(message) without a secret deliberately, to
                    # avoid the 32-byte HMAC tag. The session key protects the
                    # carrier state-space permutation, not these frame bytes, so
                    # no secret belongs here.
                    #
                    # The encoder ALWAYS emits a CRC-16 frame regardless of
                    # ecc_parity_bytes, so framing is always required here: a
                    # missing/bad frame means a wrong session key or corruption
                    # beyond repair, and the decode must fail closed (never fall
                    # back to accepting raw garbage as the message).
                    text, was_framed = unframe_payload(data, require_framed=True)
                except FrameError:
                    # Wrong key / corruption beyond RS capacity
                    continue
                # The DSSC encoder never appends trailing padding after
                # frame_payload(message), so the framed length must match the
                # decoded text length EXACTLY. This uniquely rejects the phantom
                # L + 1 candidate (whose extra 0x00 byte would leave
                # len(data) == HEADER_LEN + len(text) + 1).
                if not was_framed or len(data) != HEADER_LEN + len(text.encode("utf-8")):
                    continue
                reconstructed_text = text
                success = True
                fixed_errors = fixed
                accepted_codeword = cw_cand
                accepted_parity = cand_parity
                break
            if success:
                break

        if not success:
            # No candidate length/parity produced a valid CRC frame: fail
            # closed, reporting only the minimal (true-length) raw bytes.
            return DSSCDecodingResult(
                media_ids=normalized_carrier_ids,
                reconstructed_message=None,
                success=False,
                ecc_fixed_errors=[],
                raw_payload_bytes=list(raw_bytes[:min_len]),
                total_bits_recovered=total_bits,
                parity_bytes=fallback_parity,
                verification_rate=ver_rate,
            )

        return DSSCDecodingResult(
            media_ids=normalized_carrier_ids,
            reconstructed_message=reconstructed_text,
            success=success,
            ecc_fixed_errors=fixed_errors,
            raw_payload_bytes=list(accepted_codeword),
            total_bits_recovered=total_bits,
            parity_bytes=accepted_parity,
            verification_rate=ver_rate,
        )

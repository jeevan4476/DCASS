# src/engine/ecc.py
"""
Reed-Solomon Error Correction Code (RS-ECC) Module for DCASS.

Provides algebraic block error correction over Galois Field GF(2^8)
to eliminate vector quantization noise and semantic drift during carrier reconstruction.

Key Features:
- Appends R parity bytes to message payloads before FAISS vector search
- Fixes up to t = floor(R / 2) arbitrary corrupted media items or vector mismatches
- Correctness is guaranteed only when the number of corrupted bytes is <= t
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, List
import reedsolo


class RSErrorCorrection:
    """
    Reed-Solomon Error Correction wrapper over GF(2^8).
    """

    def __init__(self, parity_bytes: int = 8):
        """
        Initialize RS-ECC codec.

        Args:
            parity_bytes: Number of parity bytes (R).
                         Can correct up to t = floor(R / 2) unknown byte errors,
                         up to R known-position erasures, or any combination
                         satisfying 2 * e + v <= R.
        """
        self.parity_bytes = parity_bytes
        self._codec = reedsolo.RSCodec(parity_bytes)

    @property
    def max_correctable_errors(self) -> int:
        """Maximum number of unknown-position byte errors correctable."""
        return self.parity_bytes // 2

    @property
    def max_correctable_erasures(self) -> int:
        """Maximum number of known-position byte erasures correctable."""
        return self.parity_bytes

    def encode(self, data: str | bytes) -> bytes:
        """
        Encode raw string or bytes with Reed-Solomon parity.

        Args:
            data: Input message payload (string or bytes)

        Returns:
            Codeword bytes (Data + Parity bytes)
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = bytes(data)

        return bytes(self._codec.encode(data_bytes))

    def decode(
        self,
        codeword: bytes,
        erase_pos: Optional[Sequence[int]] = None,
    ) -> Tuple[str, bool, List[int]]:
        """
        Decode a received codeword using Berlekamp-Massey + Forney algorithm.

        Args:
            codeword: Received codeword bytes (potentially corrupted or erased)
            erase_pos: Optional 0-based byte indices of known erasures (dropped packets)

        Returns:
            Tuple of (decoded_str, is_success, list_of_fixed_error_positions)
        """
        data, ok, fixed = self.decode_bytes(codeword, erase_pos=erase_pos)
        return data.decode("utf-8", errors="replace"), ok, fixed

    def decode_bytes(
        self,
        codeword: bytes,
        erase_pos: Optional[Sequence[int]] = None,
    ) -> Tuple[bytes, bool, List[int]]:
        """
        Decode a received codeword to raw BYTES.

        Unlike :meth:`decode`, no UTF-8 conversion is applied - framed
        payloads contain arbitrary header bytes and must round-trip exactly.

        When *erase_pos* is provided, known erasure positions are passed to
        Reed-Solomon decoding so up to ``R`` erased bytes (or ``2*e + v <= R``
        combined unknown errors and known erasures) can be reconstructed.
        """
        clean_erase_pos: Optional[List[int]] = None
        if erase_pos is not None:
            clean_erase_pos = sorted(
                {int(pos) for pos in erase_pos if 0 <= int(pos) < len(codeword)}
            )
            if len(clean_erase_pos) > self.parity_bytes:
                raw_data = (
                    codeword[: -self.parity_bytes]
                    if len(codeword) > self.parity_bytes
                    else codeword
                )
                return raw_data, False, []

        try:
            if clean_erase_pos:
                decoded_bytes, _, errata_pos = self._codec.decode(
                    bytearray(codeword), erase_pos=clean_erase_pos
                )
            else:
                decoded_bytes, _, errata_pos = self._codec.decode(bytearray(codeword))
            return bytes(decoded_bytes), True, list(errata_pos)
        except (reedsolo.ReedSolomonError, ValueError, IndexError):
            # Uncorrectable corruption: return raw slice, flagged as failure
            raw_data = (
                codeword[: -self.parity_bytes] if len(codeword) > self.parity_bytes else codeword
            )
            return raw_data, False, []

    def __repr__(self) -> str:
        return (
            f"RSErrorCorrection(parity_bytes={self.parity_bytes}, "
            f"max_fixable_errors={self.max_correctable_errors}, "
            f"max_fixable_erasures={self.max_correctable_erasures})"
        )

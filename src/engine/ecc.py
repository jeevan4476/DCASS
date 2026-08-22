# src/engine/ecc.py
"""
Reed-Solomon Error Correction Code (RS-ECC) Module for DCASS.

Provides algebraic block error correction over Galois Field GF(2^8)
to eliminate vector quantization noise and semantic drift during covert transmission.

Key Features:
- Appends R parity bytes to secret payloads before FAISS vector search
- Fixes up to t = floor(R / 2) arbitrary corrupted media items or vector mismatches
- Correctness is guaranteed only when the number of corrupted bytes is <= t
"""

from __future__ import annotations
from typing import Tuple, List
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
                         Can correct up to t = floor(R / 2) byte errors.
        """
        self.parity_bytes = parity_bytes
        self._codec = reedsolo.RSCodec(parity_bytes)

    @property
    def max_correctable_errors(self) -> int:
        """Maximum number of byte errors correctable."""
        return self.parity_bytes // 2

    def encode(self, data: str | bytes) -> bytes:
        """
        Encode raw string or bytes with Reed-Solomon parity.

        Args:
            data: Secret payload (string or bytes)

        Returns:
            Codeword bytes (Data + Parity bytes)
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = bytes(data)

        return bytes(self._codec.encode(data_bytes))

    def decode(self, codeword: bytes) -> Tuple[str, bool, List[int]]:
        """
        Decode a received codeword using Berlekamp-Massey algorithm.

        Args:
            codeword: Received codeword bytes (potentially corrupted)

        Returns:
            Tuple of (decoded_str, is_success, list_of_fixed_error_positions)
        """
        data, ok, fixed = self.decode_bytes(codeword)
        return data.decode("utf-8", errors="replace"), ok, fixed

    def decode_bytes(self, codeword: bytes) -> Tuple[bytes, bool, List[int]]:
        """
        Decode a received codeword to raw BYTES.

        Unlike :meth:`decode`, no UTF-8 conversion is applied - framed
        payloads contain arbitrary header bytes and must round-trip exactly.
        """
        try:
            decoded_bytes, _, errata_pos = self._codec.decode(bytearray(codeword))
            return bytes(decoded_bytes), True, list(errata_pos)
        except reedsolo.ReedSolomonError:
            # Uncorrectable corruption: return raw slice, flagged as failure
            raw_data = (
                codeword[: -self.parity_bytes] if len(codeword) > self.parity_bytes else codeword
            )
            return raw_data, False, []

    def __repr__(self) -> str:
        return f"RSErrorCorrection(parity_bytes={self.parity_bytes}, max_fixable_errors={self.max_correctable_errors})"

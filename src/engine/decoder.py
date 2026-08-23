# src/engine/decoder.py
"""
SemanticDecoder - Core decoding engine for DCASS.

Decodes media sequences back into semantic meaning by:
1. Looking up each media item in the corpus
2. Extracting semantic content (caption/text)
3. Verifying items exist in the corpus (tamper detection)
4. Reconstructing the original semantic meaning

Architecture:
    Media Sequence -> Lookup Each ID -> Extract Content -> Verify -> Semantic Meaning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import time

from src.corpus.index.unified_index import (
    UnifiedSemanticIndex,
    Modality,
    extract_semantic_content,
)
from src.engine.ecc import RSErrorCorrection
from src.engine.payload_framing import FrameError, unframe_payload
from src.engine.vcp_payload import VCPPayloadMapper

PayloadMode = Literal["semantic_legacy", "exact_vcp"]


@dataclass
class DecodedItem:
    """Represents a decoded media item."""

    media_id: str
    modality: Modality
    content: str  # The semantic content (text/caption)
    verified: bool  # Whether item was found in corpus
    metadata: dict = field(default_factory=dict)
    file_path: Optional[str] = None
    payload_byte: Optional[int] = None
    cluster_id: Optional[int] = None

    def __repr__(self) -> str:
        status = "VERIFIED" if self.verified else "UNVERIFIED"
        return f"DecodedItem({self.modality}:{self.media_id}, {status})"


@dataclass
class DecodingResult:
    """Complete result of decoding a media sequence."""

    media_ids: list[str]
    decoded: list[DecodedItem]
    ecc_success: bool = True
    ecc_errors_fixed: list[int] = field(default_factory=list)
    ecc_payload: Optional[str] = None
    payload_mode: PayloadMode = "semantic_legacy"
    payload_symbols: list[int] = field(default_factory=list)
    # Set when dynamic context keying was used and an epoch was confirmed.
    context_epoch_id: Optional[str] = None

    @property
    def file_paths(self) -> list[str]:
        """Get file paths for decoded items."""
        return [d.file_path for d in self.decoded if d.file_path]

    @property
    def file_path(self) -> Optional[str]:
        """Get primary file path if available."""
        paths = self.file_paths
        return paths[0] if paths else None

    @property
    def contents(self) -> list[str]:
        """Get the semantic content from each decoded item."""
        return [d.content for d in self.decoded if d.content]

    @property
    def reconstructed_meaning(self) -> str:
        """Reconstruct the semantic meaning from all items (with RS-ECC recovery if active)."""
        if self.ecc_success and self.ecc_payload is not None and self.ecc_payload.strip():
            return self.ecc_payload
        return " | ".join(self.contents)

    @property
    def verification_rate(self) -> float:
        """Percentage of items that were verified in corpus."""
        if not self.decoded:
            return 0.0
        verified = sum(1 for d in self.decoded if d.verified)
        return verified / len(self.decoded)

    @property
    def all_verified(self) -> bool:
        """Check if all items were verified."""
        return all(d.verified for d in self.decoded)

    def summary(self) -> str:
        """Human-readable summary of decoding."""
        lines = [
            f"Media IDs: {self.media_ids}",
            f"Decoded items: {len(self.decoded)}",
            f"Verification rate: {self.verification_rate:.1%}",
            "",
            "Decoded content:",
        ]
        for i, item in enumerate(self.decoded, 1):
            status = "✓" if item.verified else "✗"
            lines.append(f"  {i}. [{status}] {item.modality}:{item.media_id}")
            lines.append(
                f'      "{item.content[:60]}..."'
                if len(item.content) > 60
                else f'      "{item.content}"'
            )

        lines.append("")
        lines.append(f'Reconstructed: "{self.reconstructed_meaning}"')
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DecodingResult(items={len(self.decoded)}, verified={self.verification_rate:.0%})"


class SemanticDecoder:
    """
    Core semantic decoder for DCASS steganography.

    Transforms sequences of media IDs back into semantic meaning
    by looking up items in the unified corpus index.

    Key Features:
    - Multi-modal decoding (images, text, audio)
    - Corpus verification (tamper detection)
    - Semantic content extraction

    Usage:
        # Initialize with index
        decoder = SemanticDecoder()
        decoder.load()  # Loads indices

        # Decode a media sequence
        result = decoder.decode(["flickr8k_00123", "wiki_00456"])

        # Get reconstructed meaning
        print(result.reconstructed_meaning)

        # Check verification
        if result.all_verified:
            print("All items verified in corpus")

    Attributes:
        index: UnifiedSemanticIndex for corpus lookup
    """

    def __init__(
        self, index: UnifiedSemanticIndex = None, base_path: Path = None, device: str = None
    ):
        """
        Initialize the decoder.

        Args:
            index: Pre-configured UnifiedSemanticIndex (created if None)
            base_path: Base path for indices (passed to index if created)
            device: Device for models (passed to index if created)
        """
        self._index = index
        self._base_path = base_path
        self._device = device
        self._loaded = False
        self._payload_mapper: Optional[VCPPayloadMapper] = None

    @property
    def index(self) -> UnifiedSemanticIndex:
        """Get the unified index (creates if needed)."""
        if self._index is None:
            self._index = UnifiedSemanticIndex(base_path=self._base_path, device=self._device)
        return self._index

    def load(self, modalities: list[Modality] = None) -> dict[str, bool]:
        """
        Load the underlying indices.

        Args:
            modalities: Specific modalities to load

        Returns:
            Dict mapping modality to load success status
        """
        print("Loading semantic decoder...")
        status = self.index.load(modalities)
        self._loaded = any(status.values())
        return status

    def is_loaded(self) -> bool:
        """Check if decoder is ready for use."""
        return self._loaded

    @property
    def payload_mapper(self) -> VCPPayloadMapper:
        """Get the exact VCP payload mapper."""
        if self._payload_mapper is None:
            self._payload_mapper = VCPPayloadMapper(self.index)
        return self._payload_mapper

    def decode(
        self,
        media_ids: list[str],
        use_ecc: bool = False,
        ecc_parity_bytes: int = 8,
        raw_codeword: Optional[bytes] = None,
        payload_mode: PayloadMode = "semantic_legacy",
        context_manager=None,
        context_epoch_hint: Optional[str] = None,
    ) -> DecodingResult:
        """
        Decode a sequence of media IDs into semantic meaning.

        Args:
            media_ids: List of media IDs to decode
            use_ecc: If True, decodes codeword using Reed-Solomon Error Correction
            ecc_parity_bytes: Number of RS parity bytes (default 8)
            raw_codeword: Legacy side-channel codeword bytes to decode via Berlekamp-Massey
            payload_mode: "exact_vcp" reconstructs bytes from media ID clusters
            context_manager: Optional ContextKeyManager. When the sender used
                dynamic context keying, the receiver must derive the same
                per-epoch permutation.
            context_epoch_hint: Epoch id from the encode result, if available.
                Skips candidate search and derives exactly that epoch.

        Returns:
            DecodingResult with decoded items and reconstructed meaning
        """
        if not self._loaded:
            raise RuntimeError("Decoder not loaded. Call load() first.")

        decoded_items = []
        id_symbols: dict[str, Optional[int]] = {}

        # Resolve carrier clusters once (exact_vcp only); with keying these
        # are PERMUTED cluster ids, not payload bytes.
        if payload_mode == "exact_vcp":
            for media_id in media_ids:
                id_symbols[media_id] = self.payload_mapper.symbol_for_media_id(media_id)

        # Dynamic context keying: try candidate epochs until RS verifies
        # (or, without ECC, until a hint-confirmed epoch is used).
        context_used_epoch: Optional[str] = None
        if context_manager is not None and payload_mode == "exact_vcp":
            if context_epoch_hint:
                # Validated parse + re-fetch external source materials.
                # Raises ValueError on malformed hints (API maps to 400).
                candidates = [context_manager.resolve_epoch_hint(context_epoch_hint)]
            else:
                candidates = context_manager.candidate_epochs(time.time())
            frame_secret = context_manager.secret
            codeword, missing_ids = self.payload_mapper.decode_symbols(media_ids)
            if not missing_ids:
                rs_ecc = RSErrorCorrection(parity_bytes=ecc_parity_bytes) if use_ecc else None
                for epoch in candidates:
                    inv = context_manager.derive_inverse_permutation(epoch)
                    recovered = bytes(int(inv[int(s)]) for s in codeword)
                    if rs_ecc is not None:
                        data, ok, fixed = rs_ecc.decode_bytes(recovered)
                        if ok:
                            # Decision 4: require frame integrity/auth
                            # (CRC or HMAC) before accepting this epoch.
                            try:
                                text, _ = unframe_payload(data, secret=frame_secret)
                            except FrameError:
                                continue  # integrity failed -> wrong epoch
                            context_used_epoch = epoch.epoch_id
                            return self._build_result(
                                media_ids=media_ids,
                                id_symbols=id_symbols,
                                codeword=bytes(codeword),
                                inverse_permutation=inv,
                                text=text,
                                ecc_success=True,
                                ecc_errors_fixed=fixed,
                                parity_bytes=ecc_parity_bytes,
                                context_epoch=context_used_epoch,
                            )
                    else:
                        # No integrity check available; require an explicit
                        # hint rather than guessing epochs silently.
                        if context_epoch_hint:
                            context_used_epoch = epoch.epoch_id
                            return self._build_result(
                                media_ids=media_ids,
                                id_symbols=id_symbols,
                                codeword=bytes(codeword),
                                inverse_permutation=inv,
                                text=recovered.decode("utf-8", errors="replace"),
                                ecc_success=True,
                                ecc_errors_fixed=[],
                                parity_bytes=ecc_parity_bytes,
                                context_epoch=context_used_epoch,
                            )
            # Keyed decode failed all candidates - fall through so the caller
            # sees a normal failure instead of garbage success.
            return self._build_result(
                media_ids=media_ids,
                id_symbols=id_symbols,
                codeword=bytes(codeword),
                inverse_permutation=None,
                text=None,
                ecc_success=False,
                ecc_errors_fixed=[],
                parity_bytes=ecc_parity_bytes,
                context_epoch=None,
                missing_ids=missing_ids,
            )

        for media_id in media_ids:
            # Look up item in corpus
            item = self.index.get_by_id(media_id)
            payload_byte = id_symbols.get(media_id)

            if item:
                # Item found - extract content
                content = extract_semantic_content(item.metadata, item.modality)

                decoded_items.append(
                    DecodedItem(
                        media_id=media_id,
                        modality=item.modality,
                        content=content,
                        verified=True,
                        metadata=item.metadata,
                        file_path=item.file_path,
                        payload_byte=payload_byte,
                        cluster_id=payload_byte,
                    )
                )
            else:
                # Item not found - unverified
                decoded_items.append(
                    DecodedItem(
                        media_id=media_id,
                        modality="text",  # Default
                        content=f"[UNVERIFIED: {media_id}]",
                        verified=False,
                        metadata={},
                        payload_byte=payload_byte,
                        cluster_id=payload_byte,
                    )
                )

        ecc_payload = None
        ecc_success = True
        ecc_errors_fixed = []
        payload_symbols = []

        if payload_mode == "exact_vcp":
            codeword, missing_ids = self.payload_mapper.decode_symbols(media_ids)
            payload_symbols = list(codeword)
            rs_ecc = RSErrorCorrection(parity_bytes=ecc_parity_bytes)
            if missing_ids:
                ecc_success = False
                ecc_payload = codeword.decode("utf-8", errors="replace")
            elif use_ecc:
                data, ecc_success, ecc_errors_fixed = rs_ecc.decode_bytes(codeword)
                try:
                    # Framed transport (current encoder default). Integrity
                    # failure here means corruption beyond RS capacity.
                    ecc_payload, _framed = unframe_payload(data)
                except FrameError:
                    ecc_success = False
                    ecc_payload = data.decode("utf-8", errors="replace")
            else:
                # Legacy raw or framed-without-ECC: sniff the version byte.
                try:
                    ecc_payload, _framed = unframe_payload(bytes(codeword))
                except FrameError:
                    ecc_success = False
                    ecc_payload = codeword.decode("utf-8", errors="replace")
        elif use_ecc and raw_codeword:
            rs_ecc = RSErrorCorrection(parity_bytes=ecc_parity_bytes)
            ecc_payload, ecc_success, ecc_errors_fixed = rs_ecc.decode(raw_codeword)
            payload_symbols = list(raw_codeword)

        return DecodingResult(
            media_ids=media_ids,
            decoded=decoded_items,
            ecc_success=ecc_success,
            ecc_errors_fixed=ecc_errors_fixed,
            ecc_payload=ecc_payload,
            payload_mode=payload_mode,
            payload_symbols=payload_symbols,
        )

    def _build_result(
        self,
        media_ids: list[str],
        id_symbols: dict,
        codeword: bytes,
        inverse_permutation,
        text: Optional[str],
        ecc_success: bool,
        ecc_errors_fixed: list,
        parity_bytes: int,
        context_epoch: Optional[str],
        missing_ids=None,
    ) -> DecodingResult:
        """Assemble a DecodingResult for a keyed exact_vcp decode."""
        decoded_items = []
        for media_id in media_ids:
            item = self.index.get_by_id(media_id)
            raw_symbol = id_symbols.get(media_id)
            payload_byte = (
                int(inverse_permutation[int(raw_symbol)])
                if inverse_permutation is not None and raw_symbol is not None
                else raw_symbol
            )
            if item:
                content = extract_semantic_content(item.metadata, item.modality)
                decoded_items.append(
                    DecodedItem(
                        media_id=media_id,
                        modality=item.modality,
                        content=content,
                        verified=True,
                        metadata=item.metadata,
                        file_path=item.file_path,
                        payload_byte=payload_byte,
                        cluster_id=raw_symbol,
                    )
                )
            else:
                decoded_items.append(
                    DecodedItem(
                        media_id=media_id,
                        modality="text",
                        content=f"[UNVERIFIED: {media_id}]",
                        verified=False,
                        metadata={},
                        payload_byte=payload_byte,
                        cluster_id=raw_symbol,
                    )
                )

        verification_rate = (
            sum(1 for d in decoded_items if d.verified) / len(decoded_items)
            if decoded_items
            else 0.0
        )
        reconstructed = text if text is not None else " | ".join(d.content for d in decoded_items)
        return DecodingResult(
            media_ids=media_ids,
            decoded=decoded_items,
            ecc_success=ecc_success and not missing_ids,
            ecc_errors_fixed=ecc_errors_fixed,
            ecc_payload=reconstructed,
            payload_mode="exact_vcp",
            payload_symbols=list(codeword),
            context_epoch_id=context_epoch,
        )
        reconstructed = text if text is not None else " | ".join(d.content for d in decoded_items)
        return DecodingResult(
            media_ids=media_ids,
            decoded=decoded_items,
            ecc_success=ecc_success and not missing_ids,
            ecc_errors_fixed=ecc_errors_fixed,
            ecc_payload=reconstructed,
            all_verified=bool(decoded_items) and all(d.verified for d in decoded_items),
            verification_rate=verification_rate,
            payload_mode="exact_vcp",
            payload_symbols=list(codeword),
            context_epoch_id=context_epoch,
        )

    def decode_to_text(self, media_ids: list[str]) -> str:
        """
        Decode media IDs and return just the reconstructed text.

        Convenience method for simple decoding.

        Args:
            media_ids: List of media IDs

        Returns:
            Reconstructed semantic meaning as string
        """
        result = self.decode(media_ids)
        return result.reconstructed_meaning

    def verify_sequence(self, media_ids: list[str]) -> tuple[bool, float]:
        """
        Verify that all media IDs exist in the corpus.

        Args:
            media_ids: List of media IDs to verify

        Returns:
            Tuple of (all_verified, verification_rate)
        """
        result = self.decode(media_ids)
        return result.all_verified, result.verification_rate

    def status(self) -> dict:
        """Get decoder status."""
        return {
            "loaded": self._loaded,
            "index": self.index.status() if self._loaded else "not loaded",
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"SemanticDecoder({status})"


# Convenience function for quick decoding
def decode_media_sequence(media_ids: list[str], base_path: Path = None) -> DecodingResult:
    """
    Quick decode a media sequence.

    Creates decoder, loads indices, and decodes in one call.

    Args:
        media_ids: Media IDs to decode
        base_path: Custom index path

    Returns:
        DecodingResult
    """
    decoder = SemanticDecoder(base_path=base_path)
    decoder.load()
    return decoder.decode(media_ids)

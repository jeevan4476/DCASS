"""
SemanticEngine — Unified encode/decode facade for DCASS.

Wraps both exact_vcp (SemanticEncoder/SemanticDecoder) and DSSC
(DSSCEncoder/DSSCDecoder) behind a single interface so callers
never need to import or instantiate the sub-engines directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.corpus.index.unified_index import UnifiedSemanticIndex, Modality
from src.engine.chunker import SemanticChunker
from src.engine.vcp_payload import VCPPayloadMapper
from src.engine.encoder import SemanticEncoder, EncodingResult, DiversityMode
from src.engine.decoder import SemanticDecoder, DecodingResult
from src.engine.dssc_encoder import DSSCEncoder, DSSCEncodingResult
from src.engine.dssc_decoder import DSSCDecoder, DSSCDecodingResult


EngineMode = Literal["exact_vcp", "dssc"]


@dataclass
class UnifiedEncodingResult:
    """Union result for any encoding mode."""
    mode: EngineMode
    media_ids: list[str]
    carrier_count: int
    bits_per_carrier: float
    ecc_parity_bytes: int
    payload_bytes: list[int]
    context_info: dict
    exact_vcp_result: Optional[EncodingResult] = None
    dssc_result: Optional[DSSCEncodingResult] = None


@dataclass
class UnifiedDecodingResult:
    """Union result for any decoding mode."""
    mode: EngineMode
    reconstructed_message: Optional[str]
    success: bool
    verification_rate: float
    ecc_fixed_errors: list[int]
    exact_vcp_result: Optional[DecodingResult] = None
    dssc_result: Optional[DSSCDecodingResult] = None


class SemanticEngine:
    """
    Unified DCASS engine — routes encode/decode calls to exact_vcp or DSSC.

    One instance is created per server process; sub-engines share the same
    UnifiedSemanticIndex and VCPPayloadMapper to avoid double-loading FAISS.
    """

    def __init__(
        self,
        index: UnifiedSemanticIndex = None,
        chunker: SemanticChunker = None,
        base_path: Path = None,
        device: str = None,
        default_modalities: list[Modality] = None,
    ):
        self._index = index
        self._chunker = chunker
        self._base_path = base_path
        self._device = device
        self.default_modalities = default_modalities or ["image", "text", "audio"]
        self._loaded = False
        self._vcp_mapper: Optional[VCPPayloadMapper] = None
        self._exact_encoder: Optional[SemanticEncoder] = None
        self._exact_decoder: Optional[SemanticDecoder] = None
        self._dssc_encoder: Optional[DSSCEncoder] = None
        self._dssc_decoder: Optional[DSSCDecoder] = None

    @property
    def index(self) -> UnifiedSemanticIndex:
        if self._index is None:
            self._index = UnifiedSemanticIndex(
                base_path=self._base_path,
                device=self._device,
                enabled_modalities=self.default_modalities,
            )
        return self._index

    def load(self, modalities: list[Modality] = None) -> dict[str, bool]:
        """Load FAISS indices and wire sub-engines (called once on startup)."""
        status = self.index.load(modalities)
        self._loaded = any(status.values())

        # Shared VCP mapper
        self._vcp_mapper = VCPPayloadMapper(self.index)

        # exact_vcp sub-engines
        self._exact_encoder = SemanticEncoder(
            index=self.index,
            chunker=self._chunker,
            default_modalities=self.default_modalities,
            base_path=self._base_path,
            device=self._device,
        )
        self._exact_encoder._loaded = self._loaded
        self._exact_encoder._payload_mapper = self._vcp_mapper

        self._exact_decoder = SemanticDecoder(
            index=self.index,
            base_path=self._base_path,
            device=self._device,
        )
        self._exact_decoder._loaded = self._loaded
        self._exact_decoder._payload_mapper = self._vcp_mapper

        # DSSC sub-engines (share same index + mapper)
        self._dssc_encoder = DSSCEncoder(
            index=self.index,
            chunker=self._chunker or SemanticChunker(),
            vcp_mapper=self._vcp_mapper,
        )
        self._dssc_decoder = DSSCDecoder(
            index=self.index,
            vcp_mapper=self._vcp_mapper,
        )

        return status

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    def encode(
        self,
        message: str,
        mode: EngineMode = "exact_vcp",
        session_key: bytes = None,
        modalities: list[Modality] = None,
        use_ecc: bool = True,
        ecc_parity_bytes: int = 8,
        diversity_mode: DiversityMode = "best",
        context_manager=None,
        cover_story: str = None,
    ) -> UnifiedEncodingResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")
        modalities = modalities or self.default_modalities

        if mode == "exact_vcp":
            if self._exact_encoder is None:
                self.load()
            result = self._exact_encoder.encode(
                message=message,
                modalities=modalities,
                use_ecc=use_ecc,
                ecc_parity_bytes=ecc_parity_bytes,
                diversity_mode=diversity_mode,
                context_manager=context_manager,
                cover_story=cover_story,
            )
            return UnifiedEncodingResult(
                mode="exact_vcp",
                media_ids=result.media_ids,
                carrier_count=len(result.media_ids),
                bits_per_carrier=8.0,
                ecc_parity_bytes=result.ecc_parity_bytes,
                payload_bytes=result.payload_symbols,
                context_info=result.context_info,
                exact_vcp_result=result,
            )

        elif mode == "dssc":
            if not session_key:
                raise ValueError(
                    "session_key (bytes) is required for DSSC mode. "
                    "Generate a random key per session: os.urandom(32)."
                )
            if self._dssc_encoder is None:
                self.load()
            result = self._dssc_encoder.encode(
                message=message,
                session_key=session_key,
                ecc_parity_bytes=ecc_parity_bytes,
                modalities=modalities,
            )
            avg_bits = result.bits_per_carrier_avg
            return UnifiedEncodingResult(
                mode="dssc",
                media_ids=result.carrier_ids,
                carrier_count=len(result.carrier_ids),
                bits_per_carrier=round(avg_bits, 2),
                ecc_parity_bytes=result.parity_bytes,
                payload_bytes=list(result.encoded_carriers[i].symbol for i in range(len(result.encoded_carriers))),
                context_info={"session_key_id": result.session_key_id},
                dssc_result=result,
            )

        else:
            raise ValueError(f"Unknown mode {mode!r}. Valid: 'exact_vcp', 'dssc'.")

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def decode(
        self,
        media_ids: list[str],
        mode: EngineMode = "exact_vcp",
        session_key: bytes = None,
        modalities: list[Modality] = None,
        use_ecc: bool = True,
        ecc_parity_bytes: int = 8,
        context_manager=None,
        context_epoch_hint: str = None,
    ) -> UnifiedDecodingResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        if mode == "exact_vcp":
            if self._exact_decoder is None:
                self.load()
            result = self._exact_decoder.decode(
                media_ids=media_ids,
                use_ecc=use_ecc,
                ecc_parity_bytes=ecc_parity_bytes,
                context_manager=context_manager,
                context_epoch_hint=context_epoch_hint,
            )
            return UnifiedDecodingResult(
                mode="exact_vcp",
                reconstructed_message=result.reconstructed_meaning,
                success=result.ecc_success,
                verification_rate=result.verification_rate,
                ecc_fixed_errors=result.ecc_errors_fixed,
                exact_vcp_result=result,
            )

        elif mode == "dssc":
            if not session_key:
                raise ValueError(
                    "session_key (bytes) is required for DSSC mode."
                )
            if self._dssc_decoder is None:
                self.load()
            result = self._dssc_decoder.decode(
                carrier_ids=media_ids,
                session_key=session_key,
                ecc_parity_bytes=ecc_parity_bytes,
                modalities=modalities,
            )
            return UnifiedDecodingResult(
                mode="dssc",
                reconstructed_message=result.reconstructed_message,
                success=result.success,
                verification_rate=result.verification_rate,
                ecc_fixed_errors=result.ecc_fixed_errors,
                dssc_result=result,
            )

        else:
            raise ValueError(f"Unknown mode {mode!r}. Valid: 'exact_vcp', 'dssc'.")

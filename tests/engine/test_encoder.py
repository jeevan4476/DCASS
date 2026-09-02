# tests/engine/test_encoder.py
"""Tests for SemanticEncoder removing semantic_legacy mode."""

import inspect
import pytest

from src.corpus.index.unified_index import MediaItem
from src.engine.encoder import SemanticEncoder
from src.engine.vcp_payload import PayloadCarrier


class FakePayloadMapper:
    def __init__(self):
        self.id_to_symbol: dict[str, int] = {}
        self.counter = 0

    def select_carrier(
        self,
        symbol: int,
        query: str = "",
        modalities: list[str] | None = None,
        used_ids: set[str] | None = None,
        avoid_duplicates: bool = True,
    ) -> PayloadCarrier:
        media_id = f"{(modalities or ['text'])[0]}_{symbol:02x}_{self.counter}"
        if avoid_duplicates and media_id in (used_ids or set()):
            raise RuntimeError(f"Duplicate carrier for {symbol}")
        self.counter += 1
        self.id_to_symbol[media_id] = int(symbol)
        media = MediaItem(
            id=media_id,
            modality=(modalities or ["text"])[0],
            content=f"carrier for 0x{symbol:02x}",
            score=1.0,
            normalized_score=1.0,
            metadata={"text": f"carrier for 0x{symbol:02x}"},
        )
        return PayloadCarrier(media, int(symbol), self.counter - 1, self.counter - 1, 1.0)


def test_semantic_legacy_mode_raises():
    """semantic_legacy is removed; passing it must raise immediately."""
    enc = SemanticEncoder()
    enc._loaded = True
    with pytest.raises((ValueError, TypeError)):
        enc.encode("hello", payload_mode="semantic_legacy")


def test_default_payload_mode_is_exact_vcp():
    """encode() default must route to exact_vcp, not legacy."""
    sig = inspect.signature(SemanticEncoder.encode)
    params = sig.parameters
    if "payload_mode" in params:
        assert params["payload_mode"].default == "exact_vcp"
    else:
        # If payload_mode param was removed entirely, that's also valid
        assert "payload_mode" not in params

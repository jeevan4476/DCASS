"""
Regression tests for exact VCP payload transport.

These tests prove the core research claim locally: exact mode recovers payload
bytes from media IDs alone, while the legacy semantic path does not.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.corpus.index.unified_index import MediaItem
from src.engine.decoder import SemanticDecoder
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
        media_id = f"carrier_{symbol:02x}_{self.counter}"
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

    def decode_symbols(self, media_ids: list[str]) -> tuple[bytes, list[str]]:
        missing = [media_id for media_id in media_ids if media_id not in self.id_to_symbol]
        return bytes(self.id_to_symbol.get(media_id, 0) for media_id in media_ids), missing

    def symbol_for_media_id(self, media_id: str) -> int | None:
        return self.id_to_symbol.get(media_id)


@pytest.fixture
def fake_index():
    index = Mock()
    index.load.return_value = {"text": True}
    index.status.return_value = {"loaded": True}

    def search(query, k=5, modalities=None, min_score=0.0):
        return [
            MediaItem(
                id=f"legacy_{i}",
                modality="text",
                content=f"semantic content for {query}",
                score=1.0,
                normalized_score=1.0,
                metadata={"text": f"semantic content for {query}"},
            )
            for i in range(k)
        ]

    def get_by_id(media_id):
        return MediaItem(
            id=media_id,
            modality="text",
            content=f"content for {media_id}",
            score=1.0,
            normalized_score=1.0,
            metadata={"text": f"content for {media_id}"},
        )

    index.search.side_effect = search
    index.get_by_id.side_effect = get_by_id
    return index


def test_legacy_mode_rejected(fake_index):
    encoder = SemanticEncoder(index=fake_index)
    encoder._loaded = True
    with pytest.raises((ValueError, TypeError)):
        encoder.encode("alpha bravo", payload_mode="semantic_legacy")


def test_exact_vcp_round_trip_recovers_payload_without_raw_codeword(fake_index):
    mapper = FakePayloadMapper()
    encoder = SemanticEncoder(index=fake_index)
    encoder._loaded = True
    encoder._payload_mapper = mapper
    decoder = SemanticDecoder(index=fake_index)
    decoder._loaded = True
    decoder._payload_mapper = mapper

    result = encoder.encode(
        "alpha bravo",
        use_ecc=True,
        ecc_parity_bytes=4,
    )
    decoded = decoder.decode(
        result.media_ids,
        use_ecc=True,
        ecc_parity_bytes=4,
    )

    assert decoded.ecc_success is True
    assert decoded.reconstructed_meaning == "alpha bravo"
    assert decoded.payload_symbols == result.payload_symbols
    assert [item.payload_byte for item in decoded.decoded] == result.payload_symbols


def test_exact_vcp_reports_failure_when_rs_capacity_is_exceeded(fake_index):
    mapper = FakePayloadMapper()
    encoder = SemanticEncoder(index=fake_index)
    encoder._loaded = True
    encoder._payload_mapper = mapper
    decoder = SemanticDecoder(index=fake_index)
    decoder._loaded = True
    decoder._payload_mapper = mapper

    result = encoder.encode(
        "alpha bravo",
        use_ecc=True,
        ecc_parity_bytes=4,
    )

    corrupted_ids = list(result.media_ids)
    for position in range(3):
        wrong_symbol = (result.payload_symbols[position] + 17) % 256
        corrupted_ids[position] = mapper.select_carrier(wrong_symbol).media.id

    decoded = decoder.decode(
        corrupted_ids,
        use_ecc=True,
        ecc_parity_bytes=4,
    )

    assert decoded.ecc_success is False
    assert decoded.reconstructed_meaning != "alpha bravo"

# src/engine/vcp_payload.py
"""
Exact payload mapping through Voronoi Codebook Partitioning.

This module is the bridge between byte-oriented ECC codewords and unmodified
media IDs. It uses the saved VCP cluster assignments to select media from the
cluster matching each payload byte, and to recover bytes from received IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.corpus.cluster.voronoi_codebook import VoronoiCodebook
from src.corpus.index.unified_index import (
    MediaItem,
    Modality,
    UnifiedSemanticIndex,
    extract_semantic_content,
)


CANONICAL_MODALITIES: tuple[Modality, ...] = ("image", "text", "audio")


@dataclass(frozen=True)
class PayloadCarrier:
    """A selected carrier and the byte symbol it represents."""

    media: MediaItem
    symbol: int
    global_index: int
    local_index: int
    semantic_score: float


class VCPPayloadMapper:
    """
    Maps payload bytes to media IDs and media IDs back to payload bytes.

    The global row order must match the fitting script:
    image.index, then text.index, then audio.index.
    """

    def __init__(
        self,
        index: UnifiedSemanticIndex,
        codebook: Optional[VoronoiCodebook] = None,
        codebook_path: Optional[Path] = None,
    ):
        self.index = index
        self.codebook = codebook
        self.codebook_path = codebook_path
        self._loaded = False
        self._symbol_to_globals: dict[int, list[int]] = {}
        self._global_to_entry: dict[int, tuple[Modality, int, dict]] = {}
        self._id_to_symbol: dict[str, int] = {}
        self._id_to_entry: dict[str, tuple[Modality, int, dict]] = {}
        self._query_embeddings: dict[str, np.ndarray] = {}

    def load(self) -> None:
        """Load codebook metadata and build lookup tables."""
        if self._loaded:
            return

        missing_modalities = [m for m in CANONICAL_MODALITIES if m not in self.index.indices]
        if missing_modalities:
            raise RuntimeError(
                "Exact VCP payload mode requires all canonical indices to be loaded: "
                + ", ".join(CANONICAL_MODALITIES)
            )

        if self.codebook is None:
            path = self.codebook_path or (self.index.base_path / "voronoi_codebook.npz")
            self.codebook = VoronoiCodebook()
            self.codebook.load(path)

        if self.codebook.cluster_assignments is None:
            raise RuntimeError("Voronoi codebook has no cluster assignments")

        assignments = self.codebook.cluster_assignments
        self._symbol_to_globals = {i: [] for i in range(self.codebook.num_clusters)}
        self._global_to_entry = {}
        self._id_to_symbol = {}
        self._id_to_entry = {}

        global_offset = 0
        for modality in CANONICAL_MODALITIES:
            faiss_index = self.index.indices[modality]
            meta_list = self.index.metadata.get(modality, [])
            count = min(faiss_index.ntotal, len(meta_list))

            for local_index in range(count):
                global_index = global_offset + local_index
                if global_index >= len(assignments):
                    raise RuntimeError(
                        "Voronoi assignments are shorter than the loaded FAISS corpus"
                    )

                meta = meta_list[local_index]
                media_id = meta.get("id", f"{modality}_{local_index}")
                symbol = int(assignments[global_index])

                self._symbol_to_globals.setdefault(symbol, []).append(global_index)
                self._global_to_entry[global_index] = (modality, local_index, meta)
                self._id_to_symbol[media_id] = symbol
                self._id_to_entry[media_id] = (modality, local_index, meta)

            global_offset += faiss_index.ntotal

        self._loaded = True

    def select_carrier(
        self,
        symbol: int,
        query: str = "",
        modalities: Optional[list[Modality]] = None,
        used_ids: Optional[set[str]] = None,
        avoid_duplicates: bool = True,
    ) -> PayloadCarrier:
        """Select a carrier whose VCP cluster maps exactly to `symbol`."""
        self.load()

        allowed_modalities = set(modalities or CANONICAL_MODALITIES)
        used_ids = used_ids or set()
        candidates = []

        for global_index in self._symbol_to_globals.get(int(symbol), []):
            modality, local_index, meta = self._global_to_entry[global_index]
            if modality not in allowed_modalities:
                continue

            media_id = meta.get("id", f"{modality}_{local_index}")
            if avoid_duplicates and media_id in used_ids:
                continue

            media = self._make_media_item(
                modality=modality,
                local_index=local_index,
                meta=meta,
                query=query,
            )
            score = self._score_candidate(query, media, local_index)
            candidates.append(PayloadCarrier(media, int(symbol), global_index, local_index, score))

        if not candidates:
            raise RuntimeError(
                f"No available VCP carrier found for byte 0x{int(symbol):02x} "
                f"in modalities {sorted(allowed_modalities)}"
            )

        candidates.sort(key=lambda c: (c.semantic_score, c.media.normalized_score, c.media.id), reverse=True)
        return candidates[0]

    def decode_symbols(self, media_ids: list[str]) -> tuple[bytes, list[str]]:
        """Recover the VCP byte stream represented by media IDs."""
        self.load()

        symbols = bytearray()
        missing = []
        for media_id in media_ids:
            symbol = self._id_to_symbol.get(media_id)
            if symbol is None:
                missing.append(media_id)
                symbols.append(0)
            else:
                symbols.append(symbol)

        return bytes(symbols), missing

    def symbol_for_media_id(self, media_id: str) -> Optional[int]:
        """Return the byte symbol for a media ID, if it exists in the corpus."""
        self.load()
        return self._id_to_symbol.get(media_id)

    def _make_media_item(
        self,
        modality: Modality,
        local_index: int,
        meta: dict,
        query: str,
    ) -> MediaItem:
        content = extract_semantic_content(meta, modality)
        score = self._vector_score(query, modality, local_index)
        normalized = self.index.normalizer.normalize(score, modality)
        return MediaItem(
            id=meta.get("id", f"{modality}_{local_index}"),
            modality=modality,
            content=content,
            score=score,
            normalized_score=normalized,
            metadata=meta,
        )

    def _score_candidate(self, query: str, media: MediaItem, local_index: int) -> float:
        if query:
            return self._vector_score(query, media.modality, local_index)
        return media.normalized_score

    def _vector_score(self, query: str, modality: Modality, local_index: int) -> float:
        """
        Compute cosine similarity between the query and a specific FAISS vector.

        Routes through the correct encoder for each modality:
        - image/text: CLIP text encoder
        - audio: CLAP text encoder (same space as the CLAP audio FAISS index)
        """
        if not query or not hasattr(self.index, "_encode_query"):
            return 1.0

        try:
            # Key the cache on (query, modality) because different encoders
            # produce different embeddings for the same query text.
            cache_key = f"{modality}:{query}"
            query_embedding = self._query_embeddings.get(cache_key)
            if query_embedding is None:
                query_embedding = self.index._encode_query(query, modality)[0]
                self._query_embeddings[cache_key] = query_embedding

            vector = self.index.indices[modality].reconstruct(local_index)
            vector = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return float(np.dot(query_embedding, vector))
        except Exception:
            return 1.0

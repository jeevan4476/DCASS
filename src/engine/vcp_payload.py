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

import json

import numpy as np

from src.corpus.cluster.voronoi_codebook import VoronoiCodebook
from src.corpus.index.unified_index import (
    MediaItem,
    Modality,
    UnifiedSemanticIndex,
    extract_semantic_content,
    resolve_indices_base_path,
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

        codebook_from_disk = False
        if self.codebook is None:
            # Allow the index to carry a pre-fitted/shared codebook instance
            # (used by tests and by callers that fit or load it themselves).
            shared = getattr(self.index, "_vcp_codebook", None)
            if shared is not None:
                self.codebook = shared
            else:
                path = self.codebook_path or (self.index.base_path / "voronoi_codebook.npz")
                self.codebook = VoronoiCodebook()
                self.codebook.load(path)
                codebook_from_disk = True

        # ------------------------------------------------------------------
        # Codebook <-> index binding check (WP-3): refuse to run when the
        # sidecar exists but the live indices no longer match, so a rebuilt
        # index produces a loud failure instead of silently wrong bytes.
        # Only applies when the codebook came from disk - injected in-memory
        # codebooks (tests) have no certified pairing.
        # ------------------------------------------------------------------
        if not codebook_from_disk:
            pass  # skip binding verification for shared/injected codebooks
        else:
            base = getattr(self.index, "base_path", None) or resolve_indices_base_path()
            sidecar_path = Path(base) / "voronoi_codebook.meta.json"
            if sidecar_path.exists():
                try:
                    import hashlib

                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        sidecar = json.load(f)
                    expected = sidecar.get("index_fingerprints", {})
                    for modality in ("image", "text", "audio"):
                        idx = self.index.indices.get(modality)
                        exp = expected.get(modality, {}).get("fingerprint")
                        if idx is None or not exp or not hasattr(idx, "reconstruct"):
                            continue
                        ntotal = int(idx.ntotal)
                        v0 = np.asarray(idx.reconstruct(0), dtype=np.float32).tobytes()
                        vn = (
                            np.asarray(idx.reconstruct(ntotal - 1), dtype=np.float32).tobytes()
                            if ntotal > 1
                            else b""
                        )
                        live_fp = hashlib.sha256(v0 + vn + str(ntotal).encode()).hexdigest()[:16]
                        if live_fp != exp:
                            raise RuntimeError(
                                f"Codebook/index binding BROKEN for '{modality}': "
                                f"live fingerprint {live_fp} != certified {exp}. "
                                f"An index was rebuilt without re-fitting the "
                                f"codebook - decoding now would produce WRONG "
                                f"bytes. Re-fit the codebook and re-bless via "
                                f"scripts/cluster/bless_codebook.py --bless."
                            )
                except RuntimeError:
                    raise
                except Exception as e:
                    print(f"[VCPPayloadMapper] binding check skipped: {e}")
            else:
                print(
                    "[VCPPayloadMapper] note: no voronoi_codebook.meta.json sidecar; "
                    "run scripts/cluster/bless_codebook.py --bless to certify the pairing."
                )

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

        # Warn about byte symbols with no carriers: encoding any message that
        # contains such a byte will fail at select_carrier() time.
        empty_symbols = [s for s, g in sorted(self._symbol_to_globals.items()) if not g]
        if empty_symbols:
            print(
                f"[VCPPayloadMapper] WARNING: {len(empty_symbols)} byte symbols have "
                f"zero carriers in the corpus: "
                f"{[f'0x{s:02x}' for s in empty_symbols[:16]]}"
                f"{'...' if len(empty_symbols) > 16 else ''}"
            )

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
        if not candidates and modalities:
            # Graceful fallback: if the requested modality has no carriers in this cluster,
            # fall back to any available modality in the cluster rather than failing.
            for global_index in self._symbol_to_globals.get(int(symbol), []):
                modality, local_index, meta = self._global_to_entry[global_index]
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
            available = len(self._symbol_to_globals.get(int(symbol), []))
            if available == 0:
                raise RuntimeError(
                    f"Cannot encode byte 0x{int(symbol):02x}: its VCP cluster has "
                    f"NO carriers in the corpus. "
                    f"Re-fit the codebook or rebuild indices so every byte symbol "
                    f"is populated."
                )
            raise RuntimeError(
                f"Cannot encode byte 0x{int(symbol):02x}: all {available} carriers in "
                f"its cluster are already used. This usually means the message repeats "
                f"this byte more often than the corpus can supply distinct carriers; "
                f"retry with avoid_duplicates=False or use a larger corpus."
            )

        candidates.sort(
            key=lambda c: (c.semantic_score, c.media.normalized_score, c.media.id),
            reverse=True,
        )
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

    def usable_clusters(self, min_carriers: int = 1) -> list[int]:
        """
        Cluster ids with at least *min_carriers* carriers (Decision 3 / WP-6).

        The keyed bijection requires |usable| == 256 so every byte maps to a
        distinct cluster. The Phase-0 measurement on the shipped corpus gives
        256 usable clusters with min density 48, so this passes trivially;
        it exists to fail LOUDLY on a future corpus where empty/thin clusters
        would otherwise corrupt keyed traffic.
        """
        self.load()
        return [
            s
            for s, globals_list in sorted(self._symbol_to_globals.items())
            if len(globals_list) >= min_carriers
        ]

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

        Returns 0.0 (neutral) when scoring is unavailable so that a failure
        never artificially promotes a candidate to the top of the ranking.
        """
        if not query or not hasattr(self.index, "_encode_query"):
            return self._fallback_score(modality, local_index)

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
        except Exception as e:
            print(f"[VCPPayloadMapper] vector scoring failed for {modality}[{local_index}]: {e}")
            return self._fallback_score(modality, local_index)

    def _fallback_score(self, modality: Modality, local_index: int) -> float:
        """Neutral score derived from stored normalized score, not a constant."""
        meta_list = self.index.metadata.get(modality, [])
        if 0 <= local_index < len(meta_list):
            media_id = meta_list[local_index].get("id", f"{modality}_{local_index}")
        else:
            media_id = f"{modality}_{local_index}"
        entry = self._id_to_entry.get(media_id)
        if entry is None:
            return 0.0
        _, _, meta = entry
        return float(meta.get("normalized_score", 0.0))

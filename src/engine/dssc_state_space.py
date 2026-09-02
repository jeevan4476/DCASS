"""
DSSC State Space and Semantic Family Management.

Implements the Dynamic Semantic State-Space Coding (DSSC) state space:
1. Groups VCP clusters into high-level semantic families
2. Filters admissible carrier candidates for a given semantic context
3. Applies a session-keyed pseudo-random permutation to build a private, dynamic codebook
4. Maps exact payload bit chunks into deterministically indexed carrier states
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class SemanticFamily:
    """A high-level semantic domain grouping multiple VCP clusters."""
    name: str
    description: str
    cluster_ids: list[int]
    keywords: list[str] = field(default_factory=list)


# Standard default semantic families mapped over 256 VCP clusters
DEFAULT_SEMANTIC_FAMILIES = [
    SemanticFamily(
        name="nature_outdoor",
        description="Natural landscapes, outdoor scenes, trees, mountains, ocean",
        cluster_ids=list(range(0, 42)),
        keywords=["nature", "outdoor", "tree", "forest", "mountain", "ocean", "beach", "sky", "river", "sun"],
    ),
    SemanticFamily(
        name="urban_architecture",
        description="City streets, buildings, vehicles, modern infrastructure",
        cluster_ids=list(range(42, 85)),
        keywords=["city", "street", "building", "car", "traffic", "road", "bridge", "architecture", "urban"],
    ),
    SemanticFamily(
        name="people_interaction",
        description="Human activities, conversations, sports, crowds, portraits",
        cluster_ids=list(range(85, 128)),
        keywords=["people", "person", "crowd", "man", "woman", "child", "talking", "playing", "meeting", "face"],
    ),
    SemanticFamily(
        name="objects_indoor",
        description="Everyday indoor objects, tools, furniture, food, kitchen",
        cluster_ids=list(range(128, 170)),
        keywords=["room", "furniture", "food", "kitchen", "table", "chair", "cooking", "book", "home", "indoor"],
    ),
    SemanticFamily(
        name="technology_science",
        description="Computers, electronics, documents, science, abstract concepts",
        cluster_ids=list(range(170, 213)),
        keywords=["computer", "technology", "phone", "network", "system", "screen", "science", "error", "code"],
    ),
    SemanticFamily(
        name="sound_atmosphere",
        description="Audio ambiance, music, environmental soundscapes, weather",
        cluster_ids=list(range(213, 256)),
        keywords=["music", "sound", "rain", "thunder", "audio", "acoustic", "melody", "noise", "wind", "bells"],
    ),
]


def family_for_cluster(
    cluster_id: int,
    families: list[SemanticFamily] | None = None,
) -> SemanticFamily:
    """
    Return the SemanticFamily whose cluster_ids range contains cluster_id.

    This is the single source of truth used by both DSSCEncoder and
    DSSCDecoder to resolve a VCP cluster ID → semantic family, replacing
    the broken text-keyword matching that caused encoder/decoder divergence.

    Returns DEFAULT_SEMANTIC_FAMILIES[0] if cluster_id is not covered
    (safe fallback; should not happen with a valid 256-cluster codebook).
    """
    families = families or DEFAULT_SEMANTIC_FAMILIES
    for fam in families:
        if cluster_id in fam.cluster_ids:
            return fam
    return families[0]



def derive_session_permutation(num_candidates: int, session_key: bytes, context_salt: str = "") -> np.ndarray:
    """
    Derive a deterministic, cryptographically keyed permutation of [0, num_candidates-1].
    Uses HMAC-SHA256 in counter mode to generate pseudo-random ranking weights.
    """
    if num_candidates <= 1:
        return np.arange(num_candidates)

    scores = []
    for i in range(num_candidates):
        msg = f"{context_salt}:{i}".encode("utf-8")
        h = hmac.new(session_key, msg, hashlib.sha256).digest()
        score = int.from_bytes(h[:8], "big")
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])
    return np.array([idx for _, idx in scores], dtype=np.int32)


@dataclass
class DSSCStateSpace:
    """
    Represents the active carrier state space for a semantic chunk.
    """
    chunk_index: int
    family_name: str
    candidate_media_ids: list[str]  # Canonically sorted candidate IDs
    permuted_indices: np.ndarray    # Dynamic permutation for this session
    bits_per_carrier: int           # Capacity floor(log2(N))

    @property
    def capacity(self) -> int:
        """Usable capacity in bits."""
        return self.bits_per_carrier

    @property
    def state_count(self) -> int:
        """Total number of usable deterministic states (2^capacity)."""
        return 1 << self.bits_per_carrier

    def symbol_to_media_id(self, symbol: int) -> str:
        """Map a payload integer symbol (0 <= symbol < state_count) to a carrier media ID."""
        if symbol < 0 or symbol >= self.state_count:
            raise ValueError(f"Symbol {symbol} out of range [0, {self.state_count})")
        carrier_idx = int(self.permuted_indices[symbol])
        return self.candidate_media_ids[carrier_idx]

    def media_id_to_symbol(self, media_id: str) -> Optional[int]:
        """Map a carrier media ID back to its payload integer symbol."""
        try:
            carrier_idx = self.candidate_media_ids.index(media_id)
        except ValueError:
            return None

        # Invert permutation: find symbol where permuted_indices[symbol] == carrier_idx
        where = np.where(self.permuted_indices == carrier_idx)[0]
        if len(where) == 0 or where[0] >= self.state_count:
            return None
        return int(where[0])


class SemanticFamilyManager:
    """
    Manages semantic family definitions and selects families for input text.
    """
    def __init__(self, families: list[SemanticFamily] = None):
        self.families = families or DEFAULT_SEMANTIC_FAMILIES

    def match_families(self, text: str) -> list[SemanticFamily]:
        """Match text against semantic families using lexical keyword scoring."""
        text_lower = text.lower()
        scored = []
        for fam in self.families:
            score = sum(1 for kw in fam.keywords if kw in text_lower)
            scored.append((score, fam))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Return matched families or top default
        top_score = scored[0][0]
        if top_score > 0:
            return [fam for score, fam in scored if score >= max(1, top_score // 2)]
        return [self.families[0]]

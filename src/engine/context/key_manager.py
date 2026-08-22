# src/engine/context/key_manager.py
"""
Dynamic Context Key Derivation for DCASS.

The exact_vcp mapping from media ID to payload byte is fully determined by
the public corpus and the Voronoi codebook. Without a key, anyone holding
the same public artifacts can decode intercepted traffic: the system then
provides carrier unobtrusiveness, NOT confidentiality.

This module closes that gap with a minimum viable keyed-steganography layer:

1. Both Alice and Bob independently derive a per-epoch key from a shared,
   predictable context source - a time bucket (default 1 hour) optionally
   mixed with a slowly-varying public signal (crypto price) and an optional
   out-of-band secret.
2. The key material deterministically permutes the cluster -> byte mapping:
       carrier cluster = P[byte]        (encoder)
       byte = P^-1[carrier cluster]     (decoder)
3. Because the permutation rotates every bucket, intercepting traffic in
   one epoch reveals nothing about mappings in another, and the codebook
   alone is no longer sufficient to decode.

Honest scope (see docs/modules/08_CAPACITY_AND_TRAFFIC_COST.md):
- With secret=None this is *time-bucket obfuscation*: an attacker who knows
  the scheme can try candidate buckets. It defeats casual static decoding,
  not a determined adversary.
- Passing a `secret` (shared out of band) upgrades this to real keyed
  steganography: without the secret, permutations are unpredictable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class ContextEpoch:
    """A single derivation epoch (one permutation)."""

    bucket_start: int  # unix seconds of bucket start
    bucket_seconds: int
    sources: tuple[str, ...]
    source_materials: dict[str, str] = field(default_factory=dict)

    @property
    def epoch_id(self) -> str:
        """Canonical, shareable identifier for this epoch."""
        return f"{self.bucket_start}|{self.bucket_seconds}|{','.join(self.sources)}"

    def canonical_material(self, secret: Optional[bytes]) -> bytes:
        """Canonical bytes fed into the KDF."""
        parts = [str(self.bucket_start).encode(), str(self.bucket_seconds).encode()]
        for name in self.sources:
            # The time source is implicit in bucket_start; external sources
            # must carry their fetched material.
            material = (
                str(self.bucket_start) if name == "time" else self.source_materials.get(name, "")
            )
            parts.append(name.encode())
            parts.append(material.encode())
        blob = b"|".join(parts)
        if secret:
            blob = hmac.new(secret, blob, hashlib.sha256).digest()
        return blob


# Fetchers for optional external sources. Each returns a short stable string;
# any failure must degrade gracefully to time-only derivation.


def _fetch_crypto_price(timeout: float = 5.0) -> str:
    """Slowly-varying public signal: BTC spot price rounded to $100."""
    import urllib.request

    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        price = float(json.loads(resp.read())["bitcoin"]["usd"])
    # Quantise coarsely so both sides agree despite fetch-time skew.
    return f"btc_usd_{int(price // 100)}"


SourceFetcher = Callable[[float], str]

SOURCE_FETCHERS: dict[str, SourceFetcher] = {
    "coingecko": _fetch_crypto_price,
}


class ContextKeyManager:
    """
    Derives per-epoch permutations of the 256-symbol VCP mapping.

    Args:
        bucket_seconds: Width of one derivation epoch (default 3600).
            Must match on sender and receiver; shorter buckets rotate keys
            faster but tighten clock-sync requirements.
        secret: Optional out-of-band secret. Without it the scheme is
            time-bucket obfuscation only (documented honestly above).
        sources: Context sources mixed into the key. "time" is implicit
            and always included. Currently supports "coingecko".
        fetch_timeout: Timeout for external source fetches.
    """

    NUM_SYMBOLS = 256

    def __init__(
        self,
        bucket_seconds: int = 3600,
        secret: Optional[bytes] = None,
        sources: tuple[str, ...] = ("time",),
        fetch_timeout: float = 5.0,
        _source_fetchers: Optional[dict[str, SourceFetcher]] = None,
    ):
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")
        unknown = [s for s in sources if s != "time" and s not in SOURCE_FETCHERS]
        if unknown:
            raise ValueError(f"Unknown context sources: {unknown}")
        self.bucket_seconds = int(bucket_seconds)
        self.secret = bytes(secret) if secret else None
        self.sources = ("time",) + tuple(s for s in sources if s != "time")
        self.fetch_timeout = fetch_timeout
        # Injectable for tests / custom signals
        self._source_fetchers = dict(_source_fetchers or SOURCE_FETCHERS)

    # ------------------------------------------------------------------
    # Epoch handling
    # ------------------------------------------------------------------

    def epoch_at(self, timestamp: float) -> ContextEpoch:
        """Derive the epoch containing *timestamp*."""
        bucket_start = int(timestamp // self.bucket_seconds) * self.bucket_seconds
        materials: dict[str, str] = {"time": str(bucket_start)}
        for name in self.sources:
            if name == "time":
                continue
            fetcher = self._source_fetchers.get(name)
            if fetcher is None:
                continue
            try:
                materials[name] = fetcher(self.fetch_timeout)
            except Exception as e:
                # Degrade to time-only rather than fail the channel.
                print(
                    f"[ContextKeyManager] source '{name}' unavailable ({e}); "
                    "deriving from remaining sources"
                )
        return ContextEpoch(
            bucket_start=bucket_start,
            bucket_seconds=self.bucket_seconds,
            sources=self.sources,
            source_materials=materials,
        )

    def current_epoch(self) -> ContextEpoch:
        """Derive the current wall-clock epoch."""
        return self.epoch_at(time.time())

    def candidate_epochs(self, timestamp: float, tolerance_buckets: int = 1) -> list[ContextEpoch]:
        """
        Epochs to attempt when decoding, ordered most-likely first.

        Covers clock skew and packets sent just before a bucket boundary.
        """
        base = self.epoch_at(timestamp).bucket_start
        offsets = [0] + [d for k in range(1, tolerance_buckets + 1) for d in (k, -k)]
        epochs = [self.epoch_at(base + d * self.bucket_seconds) for d in offsets]
        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in epochs:
            if e.epoch_id not in seen:
                seen.add(e.epoch_id)
                unique.append(e)
        return unique

    # ------------------------------------------------------------------
    # Permutation derivation
    # ------------------------------------------------------------------

    def derive_permutation(self, epoch: ContextEpoch) -> np.ndarray:
        """
        Deterministic permutation P for *epoch* where sending byte b uses
        cluster P[b].
        """
        seed_bytes = epoch.canonical_material(self.secret)
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.NUM_SYMBOLS)
        return perm.astype(np.int64)

    def derive_inverse_permutation(self, epoch: ContextEpoch) -> np.ndarray:
        """Inverse of :meth:`derive_permutation`: byte b was carried by cluster inv[b]."""
        perm = self.derive_permutation(epoch)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(self.NUM_SYMBOLS, dtype=np.int64)
        return inv


def apply_symbol_permutation(payload_bytes, permutation: np.ndarray) -> list[int]:
    """Map payload bytes -> carrier clusters via P."""
    return [int(permutation[int(b)]) for b in payload_bytes]


def invert_symbol_permutation(carrier_symbols, inverse_permutation: np.ndarray) -> list[int]:
    """Map observed carrier clusters -> payload bytes via P^-1."""
    return [int(inverse_permutation[int(s)]) for s in carrier_symbols]

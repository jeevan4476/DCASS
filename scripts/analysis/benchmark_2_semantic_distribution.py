#!/usr/bin/env python3
"""
Component 2: Semantic Distribution and Selection-Bias Benchmark (schema v2).

Protocol
--------
1. Draws: D independent DSSC message batches with fresh session keys. The
   transmitted stream is the union over draws; per-draw rows give a draw
   variance band for every headline statistic.
2. Disjoint reference: the corpus reference sample is drawn uniformly from
   FAISS offsets that exclude every media id any carrier used, so the two
   samples never share an item. The legacy artifact sampled the full index
   and shared a large part of the carrier pool with the reference.
3. Multivariate headline: unbiased energy distance and unbiased RBF-kernel
   MMD^2 (median heuristic bandwidth, off-diagonal within-group pairs) on
   pooled embeddings, calibrated by a label permutation test on a shared
   pooled distance matrix with balanced groups. The U-statistic denominators
   keep the headline and the smaller per-draw samples on one scale.
4. Secondary: W1 on the first principal coordinate of a pooled PCA fit, with
   its explained variance ratio. This is a one-dimensional projection and is
   not a multivariate transport cost.
5. KS family: per-dimension two-sample KS with Benjamini-Hochberg FDR and a
   family-wise permutation p from the max ECDF-gap statistic under joint
   label permutation, which preserves cross-dimension correlation.
6. Fail-loud accounting: every message and carrier is either scored or
   dropped with a recorded reason; unexpected drops fail the run before the
   artifact is written.
7. Repeat rate: media reuse across messages and draws, with cluster entropy
   recomputed on unique media.

Outputs:
    storage/benchmarks/component_2_results.json
    storage/benchmarks/tables/table2_distributional_drift.tex
    storage/benchmarks/plots/fig_manifold_tsne.png
    storage/benchmarks/plots/fig2_distributional_stats.png
    storage/benchmarks/plots/fig_ks_heatmap.png

Usage:
    .venv/bin/python scripts/analysis/benchmark_2_semantic_distribution.py
    .venv/bin/python scripts/analysis/benchmark_2_semantic_distribution.py --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    recorded_commit,
    scale_table_to_textwidth,
    write_manifest,
)

from src.engine.semantic_engine import SemanticEngine  # noqa: E402

SEED = 42
MODALITIES = ["image", "text", "audio"]
NUM_CLUSTERS = 256

DEFAULT_DRAWS = 10
DEFAULT_MESSAGES_PER_DRAW = 50
DEFAULT_REFERENCE_SIZE = 5000
DEFAULT_PERMUTATIONS = 1000
DEFAULT_FAMILY_PERMUTATIONS = 500
BH_Q = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

BASE_TOPICS = [
    "The quick brown fox jumps over the lazy dog in the sunny park.",
    "Deep convolutional neural networks achieve state-of-the-art accuracy.",
    "A peaceful mountain stream flowing through an ancient evergreen forest.",
    "Global economic growth indicators show resilient consumer spending.",
    "A classical symphony performance in the grand concert hall tonight.",
    "Astronomers discover high-energy gamma ray emissions from a distant galaxy.",
    "The bustling downtown farmer's market offers fresh local produce.",
    "Autonomous electric vehicles navigate complex urban traffic safely.",
    "Gentle rain falls on the cobblestone street under vintage gas lamps.",
    "Innovative renewable energy technologies reduce carbon emissions worldwide.",
]

# Counters that invalidate the artifact if nonzero (fail-loud contract).
REQUIRED_ZERO_COUNTERS = (
    "messages_failed",
    "messages_without_carriers",
    "carriers_missing_embedding",
    "carriers_unmapped_cluster",
    "carriers_zero_norm",
    "carriers_modality_mismatch",
)


def provenance() -> Dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    return {
        "git_commit": commit,
        "torch": torch.__version__,
        "device": str(DEVICE),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class DropLedger:
    """Fail-loud accounting for every message and carrier in the pipeline."""

    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.failure_samples: List[str] = []

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def note_failure(self, text: str) -> None:
        if len(self.failure_samples) < 10:
            self.failure_samples.append(text)

    def dirty(self) -> List[str]:
        return [key for key in REQUIRED_ZERO_COUNTERS if self.counters.get(key, 0) > 0]

    def raise_if_dirty(self, context: str) -> None:
        dirty = self.dirty()
        if dirty:
            detail = ", ".join(f"{key}={self.counters[key]}" for key in dirty)
            samples = "; ".join(self.failure_samples)
            raise RuntimeError(
                f"fail-loud accounting violation in {context}: {detail}"
                + (f" ({samples})" if samples else "")
            )

    def to_dict(self) -> Dict:
        return {
            "counters": dict(sorted(self.counters.items())),
            "failure_samples": list(self.failure_samples),
        }


def entropy_bits(counts: np.ndarray) -> float:
    """Plug-in Shannon entropy of a count vector, in bits."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def bh_fdr(p_values: np.ndarray, q: float = BH_Q) -> Tuple[np.ndarray, int, np.ndarray]:
    """Benjamini-Hochberg step-up procedure. Returns (reject mask, count, q-values)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    k = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if k > 0:
        reject[order[:k]] = True
    qvals = np.empty(m, dtype=np.float64)
    qvals[order] = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    qvals = np.minimum(qvals, 1.0)
    return reject, k, qvals


def disjoint_offsets(
    media_ids: Sequence[Optional[str]],
    excluded_ids: set,
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Pick FAISS offsets uniformly, excluding every id in ``excluded_ids``."""
    candidates = [
        offset
        for offset, media_id in enumerate(media_ids)
        if media_id is not None and media_id not in excluded_ids
    ]
    n_missing = sum(1 for media_id in media_ids if media_id is None)
    n_excluded = len(media_ids) - n_missing - len(candidates)
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"only {len(candidates)} candidate offsets after excluding "
            f"{n_excluded} carrier ids; requested {sample_size}"
        )
    chosen = np.sort(rng.choice(np.asarray(candidates, dtype=np.int64), size=sample_size, replace=False))
    digest = hashlib.sha256(chosen.tobytes()).hexdigest()[:16]
    stats = {
        "n_total": len(media_ids),
        "n_missing_id": n_missing,
        "n_excluded_carrier_media": n_excluded,
        "n_candidates": len(candidates),
        "n_sampled": sample_size,
        "offsets_sha256_16": digest,
    }
    return chosen, stats

# ----------------------------------------------------------------------------
# Two-sample statistics: energy distance, MMD^2, permutation calibration
# ----------------------------------------------------------------------------

def median_heuristic_sigma2(
    squared_distances: torch.Tensor,
    max_pairs: int = 2_000_000,
    seed: int = SEED,
) -> float:
    """Median of off-diagonal squared distances on a random pair subsample."""
    n = int(squared_distances.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, n, (max_pairs,), generator=gen)
    j = torch.randint(0, n, (max_pairs,), generator=gen)
    mask = i != j
    device = squared_distances.device
    vals = squared_distances[i[mask].to(device), j[mask].to(device)]
    sigma2 = float(vals.median())
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise RuntimeError("median heuristic produced a degenerate bandwidth")
    return sigma2


def energy_mmd_observed(D: torch.Tensor, K: torch.Tensor, n_a: int) -> Tuple[float, float]:
    """Observed energy distance and MMD^2, both unbiased U-statistics.

    Group A is the first ``n_a`` rows of the pooled matrices, group B the rest.
    ``D`` holds pooled Euclidean distances (zero diagonal), ``K`` an RBF kernel
    with unit diagonal. Within-group means use ``n(n-1)`` denominators over
    off-diagonal pairs and the cross-group mean uses all ``n_a * n_b`` pairs, so
    both estimators are unbiased with respect to sample size. This matters
    because the V-statistic convention carries bias proportional to ``1/n``,
    which would put headline samples and the smaller per-draw samples on
    different scales. Both statistics are permutation calibrated in
    ``two_sample_permutation_test``.
    """
    n = int(D.shape[0])
    n_b = n - n_a
    if not 0 < n_a < n:
        raise ValueError("n_a must be strictly between 0 and n")
    if n_a < 2 or n_b < 2:
        raise ValueError("both groups need at least two samples")
    total_d = D.sum()
    total_k = K.sum()
    row_d = D.sum(dim=1)
    row_k = K.sum(dim=1)
    a_idx = torch.arange(n_a, device=D.device)
    d_aa = D[a_idx][:, a_idx].sum()
    d_ab = row_d[a_idx].sum() - d_aa
    d_bb = total_d - d_aa - 2.0 * d_ab
    energy = (
        2.0 * d_ab / (n_a * n_b)
        - d_aa / (n_a * (n_a - 1))
        - d_bb / (n_b * (n_b - 1))
    )
    k_aa_total = K[a_idx][:, a_idx].sum()
    k_aa_off = k_aa_total - n_a  # drop the unit diagonal
    k_ab = row_k[a_idx].sum() - k_aa_total
    k_bb_off = total_k - k_aa_total - 2.0 * k_ab - n_b
    mmd = (
        k_aa_off / (n_a * (n_a - 1))
        + k_bb_off / (n_b * (n_b - 1))
        - 2.0 * k_ab / (n_a * n_b)
    )
    return float(energy), float(mmd)


def two_sample_permutation_test(
    D: torch.Tensor,
    K: torch.Tensor,
    n_a: int,
    n_perm: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Label-permutation calibration for energy distance and MMD^2.

    Group sizes are held fixed across permutations and the pooled matrices are
    reused, so every permutation is a pure relabeling of the same pooled sample.
    """
    n = int(D.shape[0])
    n_b = n - n_a
    if not 0 < n_a < n:
        raise ValueError("n_a must be strictly between 0 and n")
    if n_a < 2 or n_b < 2:
        raise ValueError("both groups need at least two samples")
    obs_energy, obs_mmd = energy_mmd_observed(D, K, n_a)

    total_d = D.sum()
    total_k = K.sum()
    row_d = D.sum(dim=1)
    row_k = K.sum(dim=1)
    device = D.device

    gen = torch.Generator(device="cpu").manual_seed(seed)
    null_energy = np.empty(n_perm, dtype=np.float64)
    null_mmd = np.empty(n_perm, dtype=np.float64)

    for b in range(n_perm):
        a_idx = torch.randperm(n, generator=gen)[:n_a].to(device)
        d_aa = D[a_idx][:, a_idx].sum()
        d_ab = row_d[a_idx].sum() - d_aa
        d_bb = total_d - d_aa - 2.0 * d_ab
        energy = (
            2.0 * d_ab / (n_a * n_b)
            - d_aa / (n_a * (n_a - 1))
            - d_bb / (n_b * (n_b - 1))
        )
        k_aa_total = K[a_idx][:, a_idx].sum()
        k_aa_off = k_aa_total - n_a
        k_ab = row_k[a_idx].sum() - k_aa_total
        k_bb_off = total_k - k_aa_total - 2.0 * k_ab - n_b
        mmd = (
            k_aa_off / (n_a * (n_a - 1))
            + k_bb_off / (n_b * (n_b - 1))
            - 2.0 * k_ab / (n_a * n_b)
        )
        null_energy[b] = float(energy)
        null_mmd[b] = float(mmd)

    def summarize(stat: float, null: np.ndarray) -> Dict[str, float]:
        return {
            "stat": stat,
            "p_perm": (1.0 + float((null >= stat).sum())) / (n_perm + 1.0),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95.0)),
            "n_perm": n_perm,
        }

    return {
        "energy": summarize(obs_energy, null_energy),
        "mmd": summarize(obs_mmd, null_mmd),
    }


class KSFamPermutation:
    """Two-sample KS over all embedding dimensions with a family-wise max test.

    The per-dimension statistic is the maximum ECDF gap evaluated on the pooled
    order statistics (block ends only, so ties match scipy's convention). One
    shared label permutation is applied across all dimensions, so the
    family-wise max statistic inherits the cross-dimension dependence induced
    by correlated embeddings and repeated carrier media.
    """

    def __init__(self, reference: torch.Tensor, carriers: torch.Tensor) -> None:
        if reference.ndim != 2 or carriers.ndim != 2:
            raise ValueError("embeddings must be 2-D")
        if reference.shape[1] != carriers.shape[1]:
            raise ValueError("dimension mismatch between reference and carriers")
        pooled = torch.cat([reference, carriers], dim=0)
        self.device = pooled.device
        self.n_ref = int(reference.shape[0])
        self.n_car = int(carriers.shape[0])
        self.n = int(pooled.shape[0])
        self.dims = int(pooled.shape[1])
        vals, order = torch.sort(pooled, dim=0)
        block_end = torch.ones_like(vals, dtype=torch.bool)
        if self.n > 1:
            block_end[:-1] = vals[1:] != vals[:-1]
        self.block_end = block_end
        self.order = order
        self.k = torch.arange(1, self.n + 1, device=self.device, dtype=torch.float32).unsqueeze(1)
        flags = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        flags[self.n_ref:] = 1.0
        self.flags = flags

    def per_dimension(self, flags: Optional[torch.Tensor] = None) -> torch.Tensor:
        if flags is None:
            flags = self.flags
        cum = flags[self.order].cumsum(dim=0)
        gap = (cum / self.n_car - (self.k - cum) / self.n_ref).abs()
        gap = torch.where(self.block_end, gap, torch.full_like(gap, -1.0))
        return gap.max(dim=0).values

    def run(self, n_perm: int, seed: int) -> Dict[str, object]:
        observed = self.per_dimension()
        observed_max = float(observed.max())
        gen = torch.Generator(device="cpu").manual_seed(seed)
        null = np.empty(n_perm, dtype=np.float64)
        for b in range(n_perm):
            perm = torch.randperm(self.n, generator=gen).to(self.device)
            null[b] = float(self.per_dimension(self.flags[perm]).max())
        return {
            "per_dimension": observed.cpu().numpy(),
            "familywise_max": observed_max,
            "familywise_p": (1.0 + float((null >= observed_max).sum())) / (n_perm + 1.0),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95.0)),
            "n_perm": n_perm,
            "dims": self.dims,
            "n_ref": self.n_ref,
            "n_carriers": self.n_car,
        }


def ks_family_scipy(reference: np.ndarray, carriers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension two-sample KS via scipy (float64)."""
    dims = min(reference.shape[1], carriers.shape[1])
    stats = np.empty(dims, dtype=np.float64)
    pvals = np.empty(dims, dtype=np.float64)
    for d in range(dims):
        stat, p = ks_2samp(reference[:, d], carriers[:, d])
        stats[d] = stat
        pvals[d] = p
    return stats, pvals

# ----------------------------------------------------------------------------
# Carrier generation and disjoint reference sampling
# ----------------------------------------------------------------------------

def generate_carriers_for_draw(
    engine: SemanticEngine,
    draw_index: int,
    num_messages: int,
    rng: np.random.Generator,
    ledger: DropLedger,
) -> List[Dict]:
    """Encode one message batch and return scored carrier records."""
    records: List[Dict] = []
    for i in range(num_messages):
        message = f"{BASE_TOPICS[i % len(BASE_TOPICS)]} [d{draw_index}_msg_{i:03d}]"
        session_key = rng.bytes(32)
        ledger.bump("messages_requested")
        try:
            result = engine.encode(
                message,
                mode="dssc",
                session_key=session_key,
                modalities=MODALITIES,
            )
        except Exception as exc:  # noqa: BLE001
            ledger.bump("messages_failed")
            ledger.note_failure(f"draw {draw_index} message {i}: {type(exc).__name__}: {exc}")
            continue
        carriers = result.dssc_result.encoded_carriers if result.dssc_result else None
        if not carriers:
            ledger.bump("messages_without_carriers")
            continue
        ledger.bump("messages_encoded")
        for chunk_index, carrier in enumerate(carriers):
            ledger.bump("carriers_returned")
            media_id = str(carrier.media_id)
            lookup = engine.index._id_lookup.get(media_id)
            if lookup is None:
                ledger.bump("carriers_missing_embedding")
                ledger.note_failure(f"carrier {media_id} missing from index lookup")
                continue
            modality, offset = lookup
            if modality not in engine.index.indices:
                ledger.bump("carriers_missing_embedding")
                ledger.note_failure(f"carrier {media_id} points at unloaded modality {modality}")
                continue
            vector = (
                engine.index.indices[modality]
                .reconstruct(int(offset))
                .reshape(-1)
                .astype(np.float32)
            )
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                ledger.bump("carriers_zero_norm")
                continue
            vector = vector / norm
            cluster_id = None
            if engine._vcp_mapper is not None:
                cluster_id = engine._vcp_mapper.symbol_for_media_id(media_id)
            if cluster_id is None:
                ledger.bump("carriers_unmapped_cluster")
                ledger.note_failure(f"carrier {media_id} has no Voronoi cluster id")
                continue
            declared = getattr(carrier.media, "modality", None) if carrier.media else None
            if declared is not None and declared != modality:
                ledger.bump("carriers_modality_mismatch")
                ledger.note_failure(
                    f"carrier {media_id} declared {declared} but index stores {modality}"
                )
                continue
            ledger.bump("carriers_scored")
            records.append(
                {
                    "draw": draw_index,
                    "message_index": i,
                    "chunk_index": chunk_index,
                    "modality": modality,
                    "media_id": media_id,
                    "cluster_id": int(cluster_id),
                    "embedding": vector,
                }
            )
    return records


def sample_disjoint_reference(
    engine: SemanticEngine,
    carrier_ids_by_modality: Dict[str, set],
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Uniform reference sample per modality, excluding every carrier media id."""
    embeddings: Dict[str, np.ndarray] = {}
    provenance_rows: Dict[str, Dict] = {}
    for modality in MODALITIES:
        if modality not in engine.index.indices:
            raise RuntimeError(f"modality {modality} missing from FAISS index")
        index = engine.index.indices[modality]
        metadata = engine.index.metadata.get(modality, [])
        if len(metadata) != index.ntotal:
            raise RuntimeError(
                f"metadata/index length mismatch for {modality}: "
                f"{len(metadata)} vs {index.ntotal}"
            )
        media_ids = [
            str(meta.get("id")) if meta.get("id") is not None else None for meta in metadata
        ]
        excluded = carrier_ids_by_modality.get(modality, set())
        chosen, stats = disjoint_offsets(media_ids, excluded, sample_size, rng)
        stats["carrier_media_ids_seen"] = len(excluded)
        vectors = np.vstack(
            [index.reconstruct(int(offset)).reshape(1, -1) for offset in chosen]
        ).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-12
        embeddings[modality] = vectors / norms
        provenance_rows[modality] = stats
        print(
            f"Reference {modality}: sampled {stats['n_sampled']:,} of "
            f"{stats['n_candidates']:,} candidates "
            f"(excluded {stats['n_excluded_carrier_media']:,} carrier media ids)"
        )
    return embeddings, provenance_rows


# ----------------------------------------------------------------------------
# Descriptive summaries: repeats, cluster entropy, modality balance
# ----------------------------------------------------------------------------

def repeat_statistics(records: List[Dict]) -> Dict:
    """Media reuse across slots, messages and draws."""
    ids = [row["media_id"] for row in records]
    counts = Counter(ids)
    unique = len(counts)
    seen_pairs: set = set()
    within_message = 0
    for row in records:
        key = (row["draw"], row["message_index"], row["media_id"])
        if key in seen_pairs:
            within_message += 1
        else:
            seen_pairs.add(key)
    draw_fractions = []
    for draw in sorted({row["draw"] for row in records}):
        draw_ids = [row["media_id"] for row in records if row["draw"] == draw]
        if draw_ids:
            draw_fractions.append(len(set(draw_ids)) / len(draw_ids))
    per_modality = {}
    for modality in MODALITIES:
        mod_ids = [row["media_id"] for row in records if row["modality"] == modality]
        mod_counts = Counter(mod_ids)
        per_modality[modality] = {
            "slots": len(mod_ids),
            "unique_media": len(mod_counts),
            "reuse_factor": (len(mod_ids) / len(mod_counts)) if mod_counts else 0.0,
            "repeated_media": int(sum(1 for count in mod_counts.values() if count > 1)),
            "max_uses": int(max(mod_counts.values())) if mod_counts else 0,
        }
    return {
        "slots_total": len(records),
        "unique_media_total": unique,
        "reuse_factor_total": (len(records) / unique) if unique else 0.0,
        "repeated_media_total": int(sum(1 for count in counts.values() if count > 1)),
        "max_uses_total": int(max(counts.values())) if counts else 0,
        "within_message_repeat_slots": within_message,
        "per_draw_unique_fractions": [round(f, 6) for f in draw_fractions],
        "per_draw_unique_fraction_mean": float(np.mean(draw_fractions)) if draw_fractions else 0.0,
        "per_modality": per_modality,
    }


def cluster_entropy_summary(records: List[Dict]) -> Dict:
    """Cluster entropy on transmitted slots and on unique media ids."""

    def counts_for(rows: List[Dict], unique_only: bool) -> np.ndarray:
        seen: set = set()
        counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)
        for row in rows:
            if unique_only:
                if row["media_id"] in seen:
                    continue
                seen.add(row["media_id"])
            counts[row["cluster_id"]] += 1
        return counts

    max_entropy = float(np.log2(NUM_CLUSTERS))
    slots = counts_for(records, unique_only=False)
    unique = counts_for(records, unique_only=True)
    per_modality = {}
    for modality in MODALITIES:
        rows = [row for row in records if row["modality"] == modality]
        mod_slots = counts_for(rows, unique_only=False)
        mod_unique = counts_for(rows, unique_only=True)
        per_modality[modality] = {
            "slots_entropy_bits": entropy_bits(mod_slots),
            "slots_entropy_ratio": entropy_bits(mod_slots) / max_entropy,
            "unique_entropy_bits": entropy_bits(mod_unique),
            "unique_entropy_ratio": entropy_bits(mod_unique) / max_entropy,
        }
    return {
        "max_entropy_bits": max_entropy,
        "slots_entropy_bits": entropy_bits(slots),
        "slots_entropy_ratio": entropy_bits(slots) / max_entropy,
        "unique_entropy_bits": entropy_bits(unique),
        "unique_entropy_ratio": entropy_bits(unique) / max_entropy,
        "per_modality": per_modality,
        "note": (
            "Slot-level entropy counts every transmitted carrier; unique-media entropy "
            "deduplicates repeated media ids. Finite-sample calibration lives in component 2b."
        ),
    }


def modality_balance_rows(records: List[Dict], engine: SemanticEngine) -> List[Dict]:
    total = len(records)
    corpus_totals = {
        modality: int(engine.index.indices[modality].ntotal)
        if modality in engine.index.indices
        else 0
        for modality in MODALITIES
    }
    corpus_sum = sum(corpus_totals.values())
    rows = []
    for modality in MODALITIES:
        count = sum(1 for row in records if row["modality"] == modality)
        rows.append(
            {
                "modality": modality,
                "num_carriers": count,
                "proportion": (count / total) if total else 0.0,
                "corpus_total": corpus_totals[modality],
                "corpus_proportion": (corpus_totals[modality] / corpus_sum) if corpus_sum else 0.0,
            }
        )
    return rows


def band(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

# ----------------------------------------------------------------------------
# Per-modality statistics
# ----------------------------------------------------------------------------

def compute_multivariate_tests(
    reference_np: np.ndarray,
    carriers_np: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict:
    """Energy distance and MMD^2 with balanced groups and permutation p."""
    n = min(len(reference_np), len(carriers_np))
    if n < 50:
        raise RuntimeError(f"two-sample test needs at least 50 per side, got {n}")
    rng = np.random.default_rng(seed + 17)
    ref_sub = reference_np[rng.choice(len(reference_np), n, replace=False)]
    car_sub = carriers_np[rng.choice(len(carriers_np), n, replace=False)]
    ref_t = torch.from_numpy(np.ascontiguousarray(ref_sub)).to(DEVICE)
    car_t = torch.from_numpy(np.ascontiguousarray(car_sub)).to(DEVICE)
    pooled = torch.cat([car_t, ref_t], dim=0)
    squared = torch.cdist(pooled, pooled, p=2) ** 2
    sigma2 = median_heuristic_sigma2(squared, seed=seed + 19)
    kernel = torch.exp(-squared / (2.0 * sigma2))
    distances = squared.sqrt()
    del squared
    result = two_sample_permutation_test(distances, kernel, n_a=n, n_perm=n_perm, seed=seed + 23)
    result["energy"].update(
        {
            "n_a": n,
            "n_b": n,
            "n_pooled": 2 * n,
            "convention": (
                "unbiased U-statistic, n(n-1) denominators over off-diagonal "
                "within-group pairs; values are on one scale across sample sizes"
            ),
        }
    )
    result["mmd"].update(
        {
            "n_a": n,
            "n_b": n,
            "sigma2": sigma2,
            "kernel": "RBF, median heuristic",
            "convention": "unbiased off-diagonal U-statistic",
        }
    )
    return result


def observed_energy_mmd(
    reference_np: np.ndarray,
    carriers_np: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    """Observed energy distance and MMD^2 without permutation calibration."""
    ref_t = torch.from_numpy(np.ascontiguousarray(reference_np)).to(DEVICE)
    car_t = torch.from_numpy(np.ascontiguousarray(carriers_np)).to(DEVICE)
    pooled = torch.cat([car_t, ref_t], dim=0)
    squared = torch.cdist(pooled, pooled, p=2) ** 2
    sigma2 = median_heuristic_sigma2(squared, seed=seed)
    kernel = torch.exp(-squared / (2.0 * sigma2))
    distances = squared.sqrt()
    del squared
    return energy_mmd_observed(distances, kernel, n_a=int(car_t.shape[0]))


def compute_projection_stats(reference_np: np.ndarray, carriers_np: np.ndarray, seed: int) -> Dict:
    """W1 and KS on the first principal coordinate of a pooled PCA fit."""
    pooled = np.vstack([reference_np, carriers_np]).astype(np.float64)
    pca = PCA(n_components=1, random_state=seed)
    pc1 = pca.fit_transform(pooled).ravel()
    split = len(reference_np)
    ref_pc1, car_pc1 = pc1[:split], pc1[split:]
    stat, p = ks_2samp(ref_pc1, car_pc1)
    return {
        "w1_stat": float(wasserstein_distance(ref_pc1, car_pc1)),
        "pc1_var_ratio": float(pca.explained_variance_ratio_[0]),
        "projection_ks": {"stat": float(stat), "p": float(p)},
    }


def w1_pc1(reference_np: np.ndarray, carriers_np: np.ndarray, seed: int) -> Tuple[float, float]:
    """W1 on PC1 of a pooled PCA fit, with the PC1 explained variance ratio."""
    pooled = np.vstack([reference_np, carriers_np]).astype(np.float64)
    pca = PCA(n_components=1, random_state=seed)
    pc1 = pca.fit_transform(pooled).ravel()
    split = len(reference_np)
    w1 = float(wasserstein_distance(pc1[:split], pc1[split:]))
    return w1, float(pca.explained_variance_ratio_[0])


def compute_ks_family(
    reference_np: np.ndarray,
    carriers_np: np.ndarray,
    n_perm: int,
    seed: int,
    keep_arrays: bool = True,
) -> Dict:
    """Scipy per-dimension KS + BH-FDR + family-wise permutation max test."""
    stats, pvals = ks_family_scipy(reference_np, carriers_np)
    reject, k, _ = bh_fdr(pvals)
    ref_t = torch.from_numpy(np.ascontiguousarray(reference_np)).to(DEVICE)
    car_t = torch.from_numpy(np.ascontiguousarray(carriers_np)).to(DEVICE)
    family = KSFamPermutation(ref_t, car_t).run(n_perm=n_perm, seed=seed + 31)
    drift = float(np.max(np.abs(family["per_dimension"] - stats)))
    if drift > 1e-3:
        raise RuntimeError(
            f"vectorized KS disagreed with scipy by {drift:.2e}; statistic implementation drift"
        )
    result: Dict = {
        "dims": int(len(stats)),
        "bh_rejections": int(k),
        "reject_fraction": float(k / len(stats)),
        "bh_q": BH_Q,
        "familywise_max": float(family["familywise_max"]),
        "familywise_p": float(family["familywise_p"]),
        "familywise_null_p95": float(family["null_p95"]),
        "n_perm": int(family["n_perm"]),
        "n_ref": int(family["n_ref"]),
        "n_carriers": int(family["n_carriers"]),
        "scipy_vectorized_max_abs_diff": drift,
    }
    if keep_arrays:
        result["stat"] = [float(v) for v in stats]
        result["p"] = [float(v) for v in pvals]
        result["bh_reject"] = [bool(v) for v in reject]
    return result


def compute_draw_row(
    reference_np: np.ndarray,
    carriers_np: np.ndarray,
    family_perm: int,
    seed: int,
    draw_index: int,
) -> Dict:
    """One row of the per-draw variance band for a single modality."""
    energy, mmd = observed_energy_mmd(reference_np, carriers_np, seed=seed)
    w1, pc1_var = w1_pc1(reference_np, carriers_np, seed=seed)
    ks = compute_ks_family(reference_np, carriers_np, family_perm, seed, keep_arrays=False)
    return {
        "draw": draw_index,
        "n_carriers": int(len(carriers_np)),
        "energy": energy,
        "mmd": mmd,
        "w1_pc1": w1,
        "pc1_var_ratio": pc1_var,
        "ks_bh_rejections": ks["bh_rejections"],
        "ks_dims": ks["dims"],
        "familywise_p": ks["familywise_p"],
    }

# ----------------------------------------------------------------------------
# Figures and LaTeX table
# ----------------------------------------------------------------------------

def create_manifold_plot(
    corpus_embeddings: Dict[str, np.ndarray],
    carrier_embeddings: Dict[str, np.ndarray],
    output_path: Path,
) -> str:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        "Semantic Manifold: Disjoint Reference Corpus vs DCASS Carriers (t-SNE, PCA init)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    for i, modality in enumerate(MODALITIES):
        ax = fig.add_subplot(1, 3, i + 1)
        c_emb = corpus_embeddings.get(modality)
        d_emb = carrier_embeddings.get(modality)
        if c_emb is None or d_emb is None or c_emb.size == 0 or d_emb.size == 0:
            ax.set_title(f"{modality.capitalize()} - Insufficient Data", fontsize=13)
            ax.axis("off")
            continue

        n_sample = min(1500, len(c_emb), len(d_emb))
        c_idx = rng.choice(len(c_emb), n_sample, replace=False)
        d_idx = rng.choice(len(d_emb), n_sample, replace=False)
        combined = np.vstack([c_emb[c_idx], d_emb[d_idx]]).astype(np.float64)
        labels = np.hstack([np.zeros(n_sample), np.ones(n_sample)])

        pca_init = PCA(n_components=2, random_state=SEED).fit_transform(combined)
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_sample // 4)),
            random_state=SEED,
            max_iter=1000,
            init=pca_init,
            learning_rate="auto",
        )
        coords = tsne.fit_transform(combined)
        c_coords = coords[labels == 0]
        d_coords = coords[labels == 1]

        ax.scatter(c_coords[:, 0], c_coords[:, 1], s=6, alpha=0.35, c="#377eb8",
                   label=f"Reference ({n_sample:,})")
        ax.scatter(d_coords[:, 0], d_coords[:, 1], s=6, alpha=0.35, c="#e41a1c",
                   label=f"Carriers ({n_sample:,})")
        for group, color in ((c_coords, "#377eb8"), (d_coords, "#e41a1c")):
            if len(group) > 10:
                mean = group.mean(axis=0)
                cov = np.cov(group, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
                width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0))
                ax.add_patch(
                    Ellipse(
                        mean,
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=1.6,
                    )
                )
        ax.set_title(f"{modality.capitalize()} (n={n_sample:,} per group)", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def create_distributional_stats_figure(headline: Dict[str, Dict], output_path: Path) -> str:
    from matplotlib import pyplot as plt

    colors = {"image": "#4C72B0", "text": "#DD8452", "audio": "#55A868"}
    panels = [
        ("energy", "Energy distance", True),
        ("mmd", "MMD$^2$ (RBF kernel)", True),
        ("w1_pc1", "W1 on PC1 (secondary)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, label, show_p) in zip(axes, panels):
        xs = np.arange(len(MODALITIES))
        values = [headline[m][key]["stat"] for m in MODALITIES]
        ax.bar(xs, values, color=[colors[m] for m in MODALITIES], alpha=0.75, width=0.6)
        for x, modality in zip(xs, MODALITIES):
            draws = headline[modality][key].get("draw_values", [])
            if draws:
                jitter = np.random.default_rng(SEED).normal(0, 0.04, size=len(draws))
                ax.scatter(x + jitter, draws, s=16, color="black", alpha=0.6, zorder=3)
            if show_p:
                p = headline[modality][key].get("p_perm")
                if p is not None:
                    ax.text(
                        x,
                        values[x],
                        f" p={p:.3f}\n (draws)",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(xs)
        ax.set_xticklabels([m.capitalize() for m in MODALITIES])
        ax.set_title(label, fontsize=12, fontweight="medium")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("statistic (unit-norm embeddings)")
    fig.suptitle(
        "Multivariate two-sample statistics, pooled permutation p, per-draw spread (dots)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def create_ks_heatmap(headline: Dict[str, Dict], output_path: Path) -> str:
    from matplotlib import pyplot as plt

    stat_matrix = np.zeros((len(MODALITIES), 512))
    pval_matrix = np.ones((len(MODALITIES), 512))
    for i, modality in enumerate(MODALITIES):
        stats = np.asarray(headline[modality]["ks_family"]["stat"], dtype=np.float64)
        pvals = np.asarray(headline[modality]["ks_family"]["p"], dtype=np.float64)
        dims = min(512, len(stats))
        stat_matrix[i, :dims] = stats[:dims]
        pval_matrix[i, :dims] = pvals[:dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im1 = ax1.imshow(stat_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=0.5)
    ax1.set_title("KS statistic per dimension", fontsize=13, fontweight="medium")
    ax1.set_xlabel("Embedding dimension (0..511)", fontsize=11)
    ax1.set_ylabel("Modality", fontsize=11)
    ax1.set_yticks(range(len(MODALITIES)))
    ax1.set_yticklabels([m.capitalize() for m in MODALITIES])
    plt.colorbar(im1, ax=ax1, shrink=0.8, label="KS statistic")

    pval_log = -np.log10(np.clip(pval_matrix, 1e-15, 1.0))
    im2 = ax2.imshow(pval_log, cmap="Blues", aspect="auto", vmin=0, vmax=np.ceil(pval_log.max()))
    ax2.set_title("KS -log10(p) per dimension", fontsize=13, fontweight="medium")
    ax2.set_xlabel("Embedding dimension (0..511)", fontsize=11)
    ax2.set_yticks(range(len(MODALITIES)))
    ax2.set_yticklabels([m.capitalize() for m in MODALITIES])
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="-log10(p)")

    annotations = []
    for modality in MODALITIES:
        ks = headline[modality]["ks_family"]
        annotations.append(
            f"{modality}: BH {ks['bh_rejections']}/{ks['dims']}, FW p={ks['familywise_p']:.3f}"
        )
    fig.suptitle(
        "Per-dimension KS with Benjamini-Hochberg and family-wise permutation calibration\n"
        + " | ".join(annotations),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def write_latex_table(
    headline: Dict[str, Dict], output_path: Path, commit: str | None = None
) -> str:
    provenance = caption_provenance(
        n=f"{len(headline)} modalities x 10 draws x 50 messages per draw",
        scope="per-modality distributional drift against a disjoint corpus reference; "
        "not a detector evaluation",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_2_results.json"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component 2 semantic distribution check (schema v2). Energy distance and "
        r"MMD$^2$ are permutation calibrated against a disjoint reference sample; $W_1$ is a "
        r"one-dimensional PC1 projection, not a multivariate transport cost. KS rejects count "
        r"Benjamini--Hochberg discoveries over 512 dimensions; FW $p$ is the family-wise "
        r"permutation p of the maximum ECDF-gap statistic. $H_{\mathrm{uniq}}$ is cluster "
        r"entropy on unique media." + provenance + "}",
        r"\label{tab:component2-distributional-drift}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Modality & Energy $d$ [$p$] & MMD$^2$ [$p$] & $W_1$ PC1 & KS BH & FW $p$ & "
        r"$H_{\mathrm{uniq}}$ & Share \\",
        r"\midrule",
    ]
    for modality in MODALITIES:
        row = headline[modality]
        energy = row["energy"]
        mmd = row["mmd"]
        ks = row["ks_family"]
        lines.append(
            f"{modality.capitalize()} & {energy['stat']:.5f} [{energy['p_perm']:.3f}] & "
            f"{mmd['stat']:.2e} [{mmd['p_perm']:.3f}] & "
            f"{row['w1_pc1']['stat']:.5f} & "
            f"{ks['bh_rejections']}/{ks['dims']} & {ks['familywise_p']:.3f} & "
            f"{row['cluster_entropy_unique_bits']:.3f} & {row['share'] * 100:.2f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    scale_table_to_textwidth(output_path)
    return str(output_path)

# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def run_component2(args) -> Dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else BENCHMARK_DIR
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"

    print("=" * 78)
    print("Component 2: Semantic Distribution and Selection-Bias Benchmark (schema v2)")
    print("=" * 78)
    print(
        f"draws={args.draws} messages/draw={args.messages} reference={args.reference_size} "
        f"permutations={args.permutations} family_permutations={args.family_permutations} "
        f"device={DEVICE}"
    )

    engine = SemanticEngine()
    status = engine.load()
    if isinstance(status, dict):
        failed = [key for key, ok in status.items() if not ok]
        if failed:
            raise RuntimeError(f"engine load failed for modalities: {failed}")
    if not engine.is_loaded():
        raise RuntimeError("engine reported no loaded indices")
    if engine._vcp_mapper is None:
        raise RuntimeError("VCP mapper unavailable; cluster ids cannot be scored")

    ledger = DropLedger()
    records: List[Dict] = []
    for draw_index in range(args.draws):
        draw_records = generate_carriers_for_draw(engine, draw_index, args.messages, rng, ledger)
        ledger.raise_if_dirty(f"draw {draw_index}")
        records.extend(draw_records)
        by_modality = {
            modality: sum(1 for row in draw_records if row["modality"] == modality)
            for modality in MODALITIES
        }
        print(f"draw {draw_index}: {len(draw_records):,} carriers scored {by_modality}")

    if not records:
        raise RuntimeError("no carriers scored; nothing to benchmark")

    carrier_ids_by_modality: Dict[str, set] = {modality: set() for modality in MODALITIES}
    for row in records:
        carrier_ids_by_modality[row["modality"]].add(row["media_id"])

    reference_np, reference_provenance = sample_disjoint_reference(
        engine, carrier_ids_by_modality, args.reference_size, rng
    )

    carriers_slots_np: Dict[str, np.ndarray] = {}
    unique_first: Dict[str, Dict[str, np.ndarray]] = {modality: {} for modality in MODALITIES}
    for row in records:
        unique_first[row["modality"]].setdefault(row["media_id"], row["embedding"])
    for modality in MODALITIES:
        rows = [row["embedding"] for row in records if row["modality"] == modality]
        if not rows:
            raise RuntimeError(f"no carriers for modality {modality}")
        carriers_slots_np[modality] = np.vstack(rows).astype(np.float32)
    carriers_unique_np = {
        modality: np.vstack(list(unique_first[modality].values())).astype(np.float32)
        for modality in MODALITIES
    }

    repeat_rate = repeat_statistics(records)
    cluster_entropy = cluster_entropy_summary(records)
    modality_balance = modality_balance_rows(records, engine)
    print(
        f"repeat rate: {repeat_rate['slots_total']:,} slots, "
        f"{repeat_rate['unique_media_total']:,} unique media "
        f"(factor {repeat_rate['reuse_factor_total']:.4f})"
    )

    draw_rows: Dict[str, List[Dict]] = {modality: [] for modality in MODALITIES}
    for modality in MODALITIES:
        ref_np = reference_np[modality]
        for draw_index in range(args.draws):
            rows = [
                row
                for row in records
                if row["modality"] == modality and row["draw"] == draw_index
            ]
            if len(rows) < 30:
                raise RuntimeError(
                    f"draw {draw_index} modality {modality}: only {len(rows)} carriers, need >= 30"
                )
            car_np = np.vstack([row["embedding"] for row in rows]).astype(np.float32)
            draw_rows[modality].append(
                compute_draw_row(
                    ref_np, car_np, args.family_permutations, args.seed + draw_index, draw_index
                )
            )
        print(f"per-draw band computed for {modality} ({len(draw_rows[modality])} draws)")

    headline: Dict[str, Dict] = {}
    for modality in MODALITIES:
        ref_np = reference_np[modality]
        car_np = carriers_slots_np[modality]
        unique_np = carriers_unique_np[modality]
        print(
            f"[{modality}] permutation tests: {len(car_np):,} carrier slots vs "
            f"{len(ref_np):,} reference embeddings"
        )
        multi = compute_multivariate_tests(ref_np, car_np, args.permutations, args.seed)
        projection = compute_projection_stats(ref_np, car_np, args.seed)
        ks_family = compute_ks_family(ref_np, car_np, args.family_permutations, args.seed)
        if len(unique_np) < len(car_np):
            unique_multi = compute_multivariate_tests(
                ref_np, unique_np, args.permutations, args.seed + 5000
            )
            unique_extra: Dict = {
                "energy": unique_multi["energy"],
                "mmd": unique_multi["mmd"],
                "n_unique": int(len(unique_np)),
            }
        else:
            unique_extra = {"note": "no repeated media observed; identical to slots"}

        rows = draw_rows[modality]
        energy_draws = [row["energy"] for row in rows]
        mmd_draws = [row["mmd"] for row in rows]
        w1_draws = [row["w1_pc1"] for row in rows]
        multi["energy"]["draw_values"] = energy_draws
        multi["mmd"]["draw_values"] = mmd_draws
        share = len(car_np) / repeat_rate["slots_total"] if repeat_rate["slots_total"] else 0.0
        headline[modality] = {
            "n_carriers_slots": int(len(car_np)),
            "n_carriers_unique": int(len(unique_np)),
            "energy": multi["energy"],
            "energy_unique_media": unique_extra,
            "mmd": multi["mmd"],
            "w1_pc1": {
                "stat": projection["w1_stat"],
                "pc1_var_ratio": projection["pc1_var_ratio"],
                "protocol": (
                    "PCA(1) fit on pooled reference+carrier embeddings; W1 on the first "
                    "principal coordinate. One-dimensional projection, not a multivariate "
                    "transport cost."
                ),
                "draw_values": w1_draws,
            },
            "projection_ks": projection["projection_ks"],
            "ks_family": ks_family,
            "draw_band": {
                "energy": band(energy_draws),
                "mmd": band(mmd_draws),
                "w1_pc1": band(w1_draws),
                "bh_rejections": band([row["ks_bh_rejections"] for row in rows]),
                "familywise_p": band([row["familywise_p"] for row in rows]),
                "familywise_reject_fraction": float(
                    np.mean([row["familywise_p"] < 0.05 for row in rows])
                ),
                "n_carriers": [row["n_carriers"] for row in rows],
            },
            "cluster_entropy_unique_bits": cluster_entropy["per_modality"][modality][
                "unique_entropy_bits"
            ],
            "share": share,
        }

    plot_paths = {
        "manifold": create_manifold_plot(
            reference_np, carriers_slots_np, plots_dir / "fig_manifold_tsne.png"
        ),
        "distributional_stats": create_distributional_stats_figure(
            headline, plots_dir / "fig2_distributional_stats.png"
        ),
        "ks_heatmap": create_ks_heatmap(headline, plots_dir / "fig_ks_heatmap.png"),
    }
    artifact_provenance = provenance()
    table_path = write_latex_table(
        headline,
        tables_dir / "table2_distributional_drift.tex",
        commit=artifact_provenance.get("git_commit"),
    )

    artifact = {
        "schema": "component2-summary-v2",
        "provenance": artifact_provenance,
        "config": {
            "seed": args.seed,
            "draws": args.draws,
            "messages_per_draw": args.messages,
            "reference_size": args.reference_size,
            "permutations": args.permutations,
            "family_permutations": args.family_permutations,
            "bh_q": BH_Q,
            "device": str(DEVICE),
        },
        "ledger": ledger.to_dict(),
        "reference": {
            "disjoint_from_carriers": True,
            "seed": args.seed,
            "per_modality": reference_provenance,
        },
        "headline": headline,
        "draws": {modality: draw_rows[modality] for modality in MODALITIES},
        "repeat_rate": repeat_rate,
        "cluster_entropy": cluster_entropy,
        "modality_balance": modality_balance,
        "wasserstein_distances": {modality: headline[modality]["w1_pc1"]["stat"] for modality in MODALITIES},
        "plots": plot_paths,
        "table_path": table_path,
    }

    artifact_path = out_dir / "component_2_results.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {artifact_path}")
    write_manifest(
        build_manifest(
            component="component_2_semantic_distribution",
            outputs=[
                artifact_path,
                out_dir / "tables" / "table2_distributional_drift.tex",
                out_dir / "plots" / "fig2_distributional_stats.png",
                out_dir / "plots" / "fig_ks_heatmap.png",
                out_dir / "plots" / "fig_manifold_tsne.png",
            ],
            inputs=sorted(
                (PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")
            ),
            seeds={"seed": args.seed},
            extra={"config": artifact.get("config", {})},
        ),
        out_dir / "component_2_results_manifest.json",
    )

    print("-" * 78)
    for modality in MODALITIES:
        row = headline[modality]
        print(
            f"{modality:>5}: energy={row['energy']['stat']:.5f} p={row['energy']['p_perm']:.3f} | "
            f"MMD2={row['mmd']['stat']:.3e} p={row['mmd']['p_perm']:.3f} | "
            f"W1(PC1)={row['w1_pc1']['stat']:.5f} | KS BH "
            f"{row['ks_family']['bh_rejections']}/{row['ks_family']['dims']} | FW p="
            f"{row['ks_family']['familywise_p']:.3f}"
        )
    print("=" * 78)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component 2 semantic distribution benchmark (schema v2)"
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES_PER_DRAW,
                        help="messages per draw")
    parser.add_argument("--reference-size", type=int, default=DEFAULT_REFERENCE_SIZE)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--family-permutations", type=int, default=DEFAULT_FAMILY_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true", help="small smoke configuration")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="artifact directory override (default: storage/benchmarks)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Regenerate the LaTeX table from the existing "
        "component_2_results.json without re-running any measurement. "
        "The artifact is left untouched.",
    )
    args = parser.parse_args()

    if args.table_only:
        artifact_path = BENCHMARK_DIR / "component_2_results.json"
        if not artifact_path.exists():
            print(f"ERROR: {artifact_path} not found; run the benchmark first.")
            return
        artifact = json.loads(artifact_path.read_text())
        table_path = write_latex_table(
            artifact["headline"],
            BENCHMARK_DIR / "tables" / "table2_distributional_drift.tex",
            commit=artifact.get("provenance", {}).get("git_commit"),
        )
        print(f"Regenerated LaTeX table: {table_path}")
        return
    if args.quick:
        args.draws = min(args.draws, 2)
        args.messages = min(args.messages, 10)
        args.reference_size = min(args.reference_size, 1500)
        args.permutations = min(args.permutations, 200)
        args.family_permutations = min(args.family_permutations, 100)
    run_component2(args)


if __name__ == "__main__":
    main()

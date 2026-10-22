#!/usr/bin/env python3
"""
Component 2b: Transmitted Cluster Usage, Uniformity, and the Mechanism Null

Component 2 records a plug-in Voronoi cluster entropy for the transmitted
carrier stream that necessarily sits below log2(256) = 8 bits because the
sample is finite. This benchmark tests the stronger question: is the
transmitted cluster distribution distinguishable from what the DSSC
selection mechanism predicts?

The DSSC mechanism is structural:
- Each chunk is assigned a modality by deterministic round robin.
- Each modality owns two semantic families (contiguous cluster ranges).
- The family is chosen by an HMAC digest modulo two.
- The symbol is uniform over the session-permuted candidate pool, so the
  expected cluster usage is proportional to the per-modality corpus density
  inside the active family.

Nulls, calibrated on simulated draws:
  H_uniform:    carriers are uniform over all 256 clusters.
  H_mechanism:  carriers follow the structural model replayed on the observed
                per-message, per-chunk modality schedule (family coin flip,
                then density-weighted cluster inside the family).

Hygiene (schema v2, matches Component 2):
- fail-loud accounting: encode failures, empty messages, unmapped carriers,
  unmapped corpus items, empty families, and unknown modalities abort the run
  before any artifact is written;
- chi-square and max-positive-z are permutation calibrated against the same
  simulated null draws (asymptotic values are still recorded);
- repeat rates are measured, and entropy is recomputed on unique media ids;
- provenance is written into the artifact.

Outputs (default):
    storage/benchmarks/component_2b_cluster_usage_results.json
    storage/benchmarks/plots/fig_2b_cluster_usage.png
    storage/benchmarks/tables/table2b_cluster_uniformity.tex

Usage:
    .venv/bin/python scripts/analysis/benchmark_2b_transmitted_cluster_usage.py
    .venv/bin/python scripts/analysis/benchmark_2b_transmitted_cluster_usage.py --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    recorded_commit,
    scale_table_to_textwidth,
    write_manifest,
)

from src.engine.semantic_engine import SemanticEngine

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLUSTERS = 256
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

MODALITIES = ["image", "text", "audio"]

# Family ranges exactly as SemanticFamilyManager defines them
FAMILY_RANGES: Dict[str, List[int]] = {
    "nature_outdoor": list(range(0, 42)),
    "urban_architecture": list(range(42, 85)),
    "people_interaction": list(range(85, 128)),
    "objects_indoor": list(range(128, 170)),
    "technology_science": list(range(170, 213)),
    "sound_atmosphere": list(range(213, 256)),
}
MODALITY_FAMILIES: Dict[str, List[str]] = {
    "image": ["nature_outdoor", "urban_architecture"],
    "text": ["people_interaction", "objects_indoor"],
    "audio": ["technology_science", "sound_atmosphere"],
}

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
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def entropy_bits_rows(counts_rows: np.ndarray) -> np.ndarray:
    """Row-wise plug-in entropy in bits for a [reps, clusters] count matrix."""
    counts_rows = np.asarray(counts_rows, dtype=np.float64)
    totals = counts_rows.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    p = counts_rows / totals
    logp = np.zeros_like(p)
    nz = p > 0
    logp[nz] = np.log2(p[nz])
    return -(p * logp).sum(axis=1)


def repeat_statistics(records: List[Dict]) -> Dict:
    """Media reuse across transmitted slots and messages."""
    all_counts: Counter = Counter()
    per_modality_counts: Dict[str, Counter] = {mod: Counter() for mod in MODALITIES}
    within_message: set = set()
    within_message_repeats = 0
    for record in records:
        all_counts[record["media_id"]] += 1
        per_modality_counts[record["modality"]][record["media_id"]] += 1
        key = (record["message"], record["media_id"])
        if key in within_message:
            within_message_repeats += 1
        else:
            within_message.add(key)
    slots = len(records)
    unique = len(all_counts)
    return {
        "slots_total": slots,
        "unique_media_total": unique,
        "reuse_factor_total": (slots / unique) if unique else 0.0,
        "within_message_repeat_slots": within_message_repeats,
        "max_uses_total": int(max(all_counts.values())) if all_counts else 0,
        "repeated_media_total": int(sum(1 for count in all_counts.values() if count > 1)),
        "per_modality": {
            mod: {
                "slots": sum(per_modality_counts[mod].values()),
                "unique_media": len(per_modality_counts[mod]),
                "reuse_factor": (
                    sum(per_modality_counts[mod].values()) / len(per_modality_counts[mod])
                    if per_modality_counts[mod]
                    else 0.0
                ),
            }
            for mod in MODALITIES
        },
    }


def simulate_structured_counts(
    schedule: List[str],
    family_probs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replay the DSSC mechanism on the observed modality schedule.

    For every simulated repetition and every chunk in ``schedule``: flip the
    family coin, then draw a cluster from the corpus density inside that
    family. Returns a [reps, 256] count matrix.
    """
    schedule_arr = np.asarray(schedule)
    counts = np.zeros((reps, NUM_CLUSTERS), dtype=np.int64)
    rep_index = np.arange(reps, dtype=np.int64)[:, None]
    for mod in MODALITIES:
        k = int((schedule_arr == mod).sum())
        if k == 0:
            continue
        if mod not in family_probs:
            raise RuntimeError(f"no family distribution provided for modality {mod}")
        cum_family_0, cum_family_1 = family_probs[mod]
        coins = rng.random((reps, k)) < 0.5
        uniforms = rng.random((reps, k))
        idx = np.empty((reps, k), dtype=np.int64)
        idx[coins] = np.searchsorted(cum_family_0, uniforms[coins], side="right")
        idx[~coins] = np.searchsorted(cum_family_1, uniforms[~coins], side="right")
        np.clip(idx, 0, NUM_CLUSTERS - 1, out=idx)
        flat = (idx + rep_index * NUM_CLUSTERS).ravel()
        counts += (
            np.bincount(flat, minlength=reps * NUM_CLUSTERS)
            .reshape(reps, NUM_CLUSTERS)
            .astype(np.int64)
        )
    return counts


def null_statistics(
    counts_reps: np.ndarray,
    expected: np.ndarray,
    chi2_observed: float,
    max_z_observed: float,
) -> Dict[str, float]:
    """Entropy band and permutation-calibrated chi-square / max-z p-values."""
    reps = int(counts_reps.shape[0])
    entropies = entropy_bits_rows(counts_reps)
    valid = expected > 0
    exp_valid = expected[valid]
    diff = counts_reps[:, valid] - exp_valid[None, :]
    chi2_reps = (diff**2 / exp_valid[None, :]).sum(axis=1)
    max_z_reps = (diff / np.sqrt(exp_valid)[None, :]).max(axis=1)
    return {
        "entropy_mean": float(entropies.mean()),
        "entropy_p2_5": float(np.percentile(entropies, 2.5)),
        "entropy_p97_5": float(np.percentile(entropies, 97.5)),
        "chi2_mean": float(chi2_reps.mean()),
        "chi2_perm_pvalue": (1.0 + float((chi2_reps >= chi2_observed).sum())) / (reps + 1.0),
        "max_z_mean": float(max_z_reps.mean()),
        "max_z_p97_5": float(np.percentile(max_z_reps, 97.5)),
        "max_z_perm_pvalue": (1.0 + float((max_z_reps >= max_z_observed).sum())) / (reps + 1.0),
        "reps": reps,
    }

@dataclass
class ClusterUsageSummary:
    num_messages: int
    num_carriers: int
    unmapped_carriers: int
    observed_entropy_bits: float
    observed_entropy_unique_bits: float
    entropy_max_bits: float
    per_modality_entropy_bits: Dict[str, float]
    per_modality_entropy_unique_bits: Dict[str, float]
    modality_weights: Dict[str, float]
    modality_mixing_entropy_bits: float
    decomposition_residual_bits: float
    chi2_vs_uniform: float
    chi2_vs_uniform_pvalue: float
    chi2_vs_uniform_perm_pvalue: float
    cramers_v_vs_uniform: float
    chi2_vs_mechanism: float
    chi2_vs_mechanism_pvalue: float
    chi2_vs_mechanism_perm_pvalue: float
    cramers_v_vs_mechanism: float
    max_positive_z_mechanism: float
    max_positive_z_mechanism_perm_pvalue: float
    max_positive_z_uniform: float
    max_positive_z_uniform_perm_pvalue: float
    observed_h_under_uniform_null_mean: float
    observed_h_under_uniform_null_p2_5: float
    observed_h_under_uniform_null_p97_5: float
    observed_h_under_mechanism_null_mean: float
    observed_h_under_mechanism_null_p2_5: float
    observed_h_under_mechanism_null_p97_5: float
    unique_media_carriers: int
    repeat_factor: float
    within_message_repeat_slots: int
    max_media_uses: int
    mechanism_null_method: str


def build_mechanism_expectation(
    engine: SemanticEngine,
    modality_chunk_counts: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Expected cluster counts under the DSSC mechanism.

    For each modality m and family F, every corpus item of m inside F's
    cluster range is equally likely to be selected per chunk, so usage is
    proportional to per-modality, per-cluster corpus density. Each family
    receives half of the modality's chunks. Also returns cumulative cluster
    distributions per modality and family for the structured null simulation.
    """
    mapper = engine._vcp_mapper
    density: Dict[str, np.ndarray] = {
        mod: np.zeros(NUM_CLUSTERS, dtype=np.float64) for mod in MODALITIES
    }
    unmapped = 0
    for mod in MODALITIES:
        for m in engine.index.metadata.get(mod, []):
            mid = m.get("id")
            if not mid:
                continue
            sym = mapper.symbol_for_media_id(mid)
            if sym is None:
                unmapped += 1
                continue
            density[mod][sym] += 1.0

    expected = np.zeros(NUM_CLUSTERS, dtype=np.float64)
    family_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mod in MODALITIES:
        cum_families = []
        for fam_name in MODALITY_FAMILIES[mod]:
            rng_ids = FAMILY_RANGES[fam_name]
            total_in_family = float(density[mod][rng_ids].sum())
            if total_in_family <= 0:
                raise RuntimeError(
                    f"fail-loud: family {fam_name} has no {mod} corpus items; "
                    "the mechanism expectation is undefined"
                )
            probs = np.zeros(NUM_CLUSTERS, dtype=np.float64)
            probs[rng_ids] = density[mod][rng_ids] / total_in_family
            cum_families.append(np.cumsum(probs))
            n_chunks = modality_chunk_counts.get(mod, 0)
            if n_chunks > 0:
                expected[rng_ids] += (n_chunks / 2.0) * density[mod][rng_ids] / total_in_family
        family_probs[mod] = (cum_families[0], cum_families[1])
    return expected, density, unmapped, family_probs


def run_component2b(
    num_messages: int,
    sim_reps: int,
) -> Tuple[ClusterUsageSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    engine = SemanticEngine()
    status = engine.load()
    if isinstance(status, dict):
        failed = [key for key, ok in status.items() if not ok]
        if failed:
            raise RuntimeError(f"engine load failed for modalities: {failed}")
    if not engine.is_loaded():
        raise RuntimeError("engine reported no loaded indices")
    mapper = engine._vcp_mapper
    if mapper is None:
        raise RuntimeError("VCP mapper unavailable; cluster ids cannot be scored")

    observed_counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)
    per_modality_counts: Dict[str, np.ndarray] = {
        mod: np.zeros(NUM_CLUSTERS, dtype=np.int64) for mod in MODALITIES
    }
    modality_chunk_counts: Dict[str, int] = {mod: 0 for mod in MODALITIES}
    media_records: List[Dict] = []
    schedule: List[str] = []
    per_message_chunk_counts: List[int] = []
    messages_encoded = 0
    unmapped_carriers = 0

    for i in range(num_messages):
        message = f"{BASE_TOPICS[i % len(BASE_TOPICS)]} [msg_{i:03d}]"
        session_key = rng.bytes(32)
        try:
            result = engine.encode(
                message, mode="dssc", session_key=session_key, modalities=MODALITIES
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"fail-loud: message {i} failed to encode: {exc}") from exc
        carriers = result.dssc_result.encoded_carriers if result.dssc_result else None
        if not carriers:
            raise RuntimeError(f"fail-loud: message {i} produced no carriers")
        messages_encoded += 1
        per_message_chunk_counts.append(len(carriers))
        for carrier in carriers:
            mod = carrier.media.modality if carrier.media is not None else None
            if mod not in MODALITIES:
                raise RuntimeError(
                    f"fail-loud: carrier {carrier.media_id} has unknown modality {mod!r}"
                )
            sym = mapper.symbol_for_media_id(carrier.media_id)
            if sym is None:
                unmapped_carriers += 1
                continue
            observed_counts[sym] += 1
            per_modality_counts[mod][sym] += 1
            modality_chunk_counts[mod] += 1
            schedule.append(mod)
            media_records.append(
                {
                    "message": i,
                    "modality": mod,
                    "media_id": str(carrier.media_id),
                    "cluster_id": int(sym),
                }
            )
        if (i + 1) % 25 == 0:
            print(
                f"  encoded {i + 1}/{num_messages} messages, "
                f"carriers so far: {int(observed_counts.sum())}"
            )

    if unmapped_carriers > 0:
        raise RuntimeError(
            f"fail-loud: {unmapped_carriers} carriers had no Voronoi cluster id"
        )
    n_total = int(observed_counts.sum())
    if n_total == 0:
        raise RuntimeError("no carriers observed")
    if len(schedule) != n_total:
        raise RuntimeError(
            f"schedule/count mismatch: {len(schedule)} scheduled chunks vs {n_total} carriers"
        )

    observed_entropy = entropy_bits(observed_counts)
    per_mod_entropy = {mod: entropy_bits(per_modality_counts[mod]) for mod in MODALITIES}
    modality_weights = {
        mod: modality_chunk_counts[mod] / n_total if n_total else 0.0 for mod in MODALITIES
    }
    mixing_entropy = entropy_bits(np.array(list(modality_weights.values())))
    decomposition = (
        sum(modality_weights[mod] * per_mod_entropy[mod] for mod in MODALITIES) + mixing_entropy
    )
    residual = observed_entropy - decomposition

    unique_counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)
    unique_per_modality: Dict[str, np.ndarray] = {
        mod: np.zeros(NUM_CLUSTERS, dtype=np.int64) for mod in MODALITIES
    }
    seen_media: set = set()
    for record in media_records:
        if record["media_id"] in seen_media:
            continue
        seen_media.add(record["media_id"])
        unique_counts[record["cluster_id"]] += 1
        unique_per_modality[record["modality"]][record["cluster_id"]] += 1
    observed_entropy_unique = entropy_bits(unique_counts)
    per_mod_entropy_unique = {
        mod: entropy_bits(unique_per_modality[mod]) for mod in MODALITIES
    }
    repeats = repeat_statistics(media_records)

    expected, density, unmapped_corpus, family_probs = build_mechanism_expectation(
        engine, modality_chunk_counts
    )
    if unmapped_corpus > 0:
        raise RuntimeError(
            f"fail-loud: {unmapped_corpus} corpus items lack a Voronoi cluster id; "
            "the density model is incomplete"
        )
    if expected.sum() <= 0:
        raise RuntimeError("fail-loud: mechanism expectation is empty")
    expected_scaled = expected * (n_total / expected.sum())

    # Chi-square goodness of fit, both nulls (asymptotic)
    chi2_uniform = float(
        ((observed_counts - n_total / NUM_CLUSTERS) ** 2 / (n_total / NUM_CLUSTERS)).sum()
    )
    p_uniform = float(stats.chi2.sf(chi2_uniform, NUM_CLUSTERS - 1))
    v_uniform = math.sqrt(chi2_uniform / (n_total * (NUM_CLUSTERS - 1)))

    valid = expected_scaled > 0
    chi2_mech = float(
        ((observed_counts[valid] - expected_scaled[valid]) ** 2 / expected_scaled[valid]).sum()
    )
    dof_mech = int(valid.sum()) - 1
    p_mech = float(stats.chi2.sf(chi2_mech, dof_mech))
    v_mech = math.sqrt(chi2_mech / (n_total * dof_mech)) if dof_mech > 0 else 0.0

    z_mech = (observed_counts[valid] - expected_scaled[valid]) / np.sqrt(expected_scaled[valid])
    z_uniform = (observed_counts - n_total / NUM_CLUSTERS) / math.sqrt(n_total / NUM_CLUSTERS)
    max_z_mech_observed = float(z_mech.max())
    max_z_uniform_observed = float(z_uniform.max())

    print("\nSimulating null draws (uniform multinomial and structured mechanism)...")
    uniform_probs = np.ones(NUM_CLUSTERS) / NUM_CLUSTERS
    uniform_reps = rng.multinomial(n_total, uniform_probs, size=sim_reps)
    mechanism_reps = simulate_structured_counts(schedule, family_probs, sim_reps, rng)
    uniform_null = null_statistics(
        uniform_reps,
        np.full(NUM_CLUSTERS, n_total / NUM_CLUSTERS),
        chi2_uniform,
        max_z_uniform_observed,
    )
    mechanism_null = null_statistics(
        mechanism_reps, expected_scaled, chi2_mech, max_z_mech_observed
    )

    summary = ClusterUsageSummary(
        num_messages=messages_encoded,
        num_carriers=n_total,
        unmapped_carriers=unmapped_carriers,
        observed_entropy_bits=observed_entropy,
        observed_entropy_unique_bits=observed_entropy_unique,
        entropy_max_bits=math.log2(NUM_CLUSTERS),
        per_modality_entropy_bits={k: float(v) for k, v in per_mod_entropy.items()},
        per_modality_entropy_unique_bits={
            k: float(v) for k, v in per_mod_entropy_unique.items()
        },
        modality_weights={k: float(v) for k, v in modality_weights.items()},
        modality_mixing_entropy_bits=float(mixing_entropy),
        decomposition_residual_bits=float(residual),
        chi2_vs_uniform=chi2_uniform,
        chi2_vs_uniform_pvalue=p_uniform,
        chi2_vs_uniform_perm_pvalue=uniform_null["chi2_perm_pvalue"],
        cramers_v_vs_uniform=float(v_uniform),
        chi2_vs_mechanism=chi2_mech,
        chi2_vs_mechanism_pvalue=p_mech,
        chi2_vs_mechanism_perm_pvalue=mechanism_null["chi2_perm_pvalue"],
        cramers_v_vs_mechanism=float(v_mech),
        max_positive_z_mechanism=max_z_mech_observed,
        max_positive_z_mechanism_perm_pvalue=mechanism_null["max_z_perm_pvalue"],
        max_positive_z_uniform=max_z_uniform_observed,
        max_positive_z_uniform_perm_pvalue=uniform_null["max_z_perm_pvalue"],
        observed_h_under_uniform_null_mean=uniform_null["entropy_mean"],
        observed_h_under_uniform_null_p2_5=uniform_null["entropy_p2_5"],
        observed_h_under_uniform_null_p97_5=uniform_null["entropy_p97_5"],
        observed_h_under_mechanism_null_mean=mechanism_null["entropy_mean"],
        observed_h_under_mechanism_null_p2_5=mechanism_null["entropy_p2_5"],
        observed_h_under_mechanism_null_p97_5=mechanism_null["entropy_p97_5"],
        unique_media_carriers=repeats["unique_media_total"],
        repeat_factor=repeats["reuse_factor_total"],
        within_message_repeat_slots=repeats["within_message_repeat_slots"],
        max_media_uses=repeats["max_uses_total"],
        mechanism_null_method=(
            f"structured schedule replay, {sim_reps} reps "
            "(family coin flip then density-weighted cluster)"
        ),
    )

    diagnostics = {
        "unmapped_corpus_items": unmapped_corpus,
        "dof_mechanism": dof_mech,
        "expected_sum": float(expected_scaled.sum()),
        "messages_encoded": messages_encoded,
        "unmapped_carriers": unmapped_carriers,
        "per_message_chunk_counts": per_message_chunk_counts,
        "schedule_sha256_16": hashlib.sha256("|".join(schedule).encode()).hexdigest()[:16],
        "mechanism_null_reps": sim_reps,
        "repeat_statistics": repeats,
        "uniform_null": uniform_null,
        "mechanism_null": mechanism_null,
    }

    print(f"\nObserved carriers: {n_total}, unmapped: {unmapped_carriers}")
    print(f"Observed H = {observed_entropy:.4f} bits of {math.log2(NUM_CLUSTERS):.4f}")
    print(
        f"Unique media: {repeats['unique_media_total']:,} of {n_total:,} slots "
        f"(factor {repeats['reuse_factor_total']:.4f}), unique-media H = "
        f"{observed_entropy_unique:.4f}"
    )
    print(
        "Decomposition: mean per-modality H "
        f"{sum(modality_weights[m]*per_mod_entropy[m] for m in MODALITIES):.4f} + mixing "
        f"{mixing_entropy:.4f} = {decomposition:.4f} (residual {residual:+.4f})"
    )
    print(
        f"Chi2 vs uniform:   {chi2_uniform:,.1f} (asym p={p_uniform:.3g}, "
        f"perm p={uniform_null['chi2_perm_pvalue']:.3g}, V={v_uniform:.4f})"
    )
    print(
        f"Chi2 vs mechanism: {chi2_mech:,.1f} (dof={dof_mech}, asym p={p_mech:.3g}, "
        f"perm p={mechanism_null['chi2_perm_pvalue']:.3g}, V={v_mech:.4f})"
    )
    print(
        f"Max positive z (mechanism): {max_z_mech_observed:.2f} "
        f"(perm p={mechanism_null['max_z_perm_pvalue']:.3g}, "
        f"null p97.5={mechanism_null['max_z_p97_5']:.2f})"
    )
    print(
        f"Uniform null H: {uniform_null['entropy_mean']:.4f} "
        f"[{uniform_null['entropy_p2_5']:.4f}, {uniform_null['entropy_p97_5']:.4f}]"
    )
    print(
        f"Mechanism null H: {mechanism_null['entropy_mean']:.4f} "
        f"[{mechanism_null['entropy_p2_5']:.4f}, {mechanism_null['entropy_p97_5']:.4f}]"
    )

    return summary, observed_counts, expected_scaled, per_modality_counts, density, diagnostics


def create_plot(
    observed_counts: np.ndarray,
    expected_scaled: np.ndarray,
    summary: ClusterUsageSummary,
    output_path: Path,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    order = np.argsort(expected_scaled)[::-1]
    xs = np.arange(NUM_CLUSTERS)
    ax1.scatter(
        xs, observed_counts[order], s=10, color="#2E86AB", alpha=0.75,
        label="Observed transmitted carriers",
    )
    ax1.plot(
        xs, expected_scaled[order], color="#E85D5D", linewidth=1.6,
        label="DSSC mechanism expectation",
    )
    ax1.axhline(
        summary.num_carriers / NUM_CLUSTERS, color="gray", linestyle=":",
        label="Uniform usage reference",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Clusters sorted by mechanism expectation", fontsize=11)
    ax1.set_ylabel("Carrier count (log scale)", fontsize=11)
    ax1.set_title("Transmitted Usage vs Structural Expectation", fontsize=13, fontweight="medium")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: entropy null distributions with observed marker
    ax2.axvspan(
        summary.observed_h_under_uniform_null_p2_5,
        summary.observed_h_under_uniform_null_p97_5,
        color="gray", alpha=0.18, label="Uniform null 95% band",
    )
    ax2.axvline(
        summary.observed_h_under_uniform_null_mean, color="gray", linestyle="--",
        label=f"Uniform null mean {summary.observed_h_under_uniform_null_mean:.4f}",
    )
    ax2.axvspan(
        summary.observed_h_under_mechanism_null_p2_5,
        summary.observed_h_under_mechanism_null_p97_5,
        color="#2E86AB", alpha=0.22, label="Mechanism null (structured) 95% band",
    )
    ax2.axvline(
        summary.observed_h_under_mechanism_null_mean, color="#2E86AB", linestyle="--",
        label=f"Structured null mean {summary.observed_h_under_mechanism_null_mean:.4f}",
    )
    ax2.axvline(
        summary.observed_entropy_bits, color="#E85D5D", linewidth=2.4,
        label=f"Observed slots {summary.observed_entropy_bits:.4f}",
    )
    ax2.axvline(
        summary.observed_entropy_unique_bits, color="#E85D5D", linewidth=1.4,
        linestyle=":", label=f"Observed unique media {summary.observed_entropy_unique_bits:.4f}",
    )
    ax2.set_xlabel("Plug-in cluster entropy H, bits", fontsize=11)
    ax2.set_yticks([])
    ax2.set_title("Why 7.75 of 8.00: Finite-Sample Nulls", fontsize=13, fontweight="medium")
    ax2.legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved cluster usage plot to {output_path}")


def generate_latex_table(
    summary: ClusterUsageSummary, output_path: Path, commit: str | None = None
):
    provenance = caption_provenance(
        n=f"{summary.num_messages} DSSC messages, {summary.num_carriers:,} carrier slots",
        scope="cluster usage against the pipeline's own mechanism null; the uniform "
        "null is rejected, so this is not uniform-indistinguishability",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_2b_cluster_usage_results.json"),
    )
    with open(output_path, "w") as f:
        f.write("% Component 2b: Transmitted cluster usage and mechanism null\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(
            "\\caption{Transmitted Voronoi cluster usage over "
            f"{summary.num_messages} fresh DSSC messages ({summary.num_carriers:,} slots). "
            "The mechanism null is the structural expectation of the DSSC selection "
            "pipeline replayed on the observed chunk schedule; permutation p-values "
            "come from the same simulated draws." + provenance + "}\n"
        )
        f.write("\\label{tab:cluster_uniformity}\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Statistic & Value \\\\\n\\midrule\n")
        f.write(f"Carriers (slots) & {summary.num_carriers:,} \\\\\n")
        f.write(
            f"Unique media / repeat factor & {summary.unique_media_carriers:,} / "
            f"{summary.repeat_factor:.4f} \\\\\n"
        )
        f.write(f"Observed entropy (slots) & {summary.observed_entropy_bits:.4f} bits \\\\\n")
        f.write(
            f"Observed entropy (unique media) & "
            f"{summary.observed_entropy_unique_bits:.4f} bits \\\\\n"
        )
        f.write(
            f"Per-modality H (image, text, audio) & "
            f"{summary.per_modality_entropy_bits['image']:.4f}, "
            f"{summary.per_modality_entropy_bits['text']:.4f}, "
            f"{summary.per_modality_entropy_bits['audio']:.4f} \\\\\n"
        )
        f.write(f"Modality mixing term & {summary.modality_mixing_entropy_bits:.4f} bits \\\\\n")
        f.write(
            f"Chi-square vs uniform & {summary.chi2_vs_uniform:,.1f} "
            f"(perm p={summary.chi2_vs_uniform_perm_pvalue:.3g}) \\\\\n"
        )
        f.write(
            f"Chi-square vs mechanism & {summary.chi2_vs_mechanism:,.1f} "
            f"(perm p={summary.chi2_vs_mechanism_perm_pvalue:.3g}) \\\\\n"
        )
        f.write(
            f"Max positive z, mechanism null & {summary.max_positive_z_mechanism:.2f} "
            f"(perm p={summary.max_positive_z_mechanism_perm_pvalue:.3g}) \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        f.flush()
        scale_table_to_textwidth(output_path)
    print(f"Saved LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke run")
    parser.add_argument("--messages", type=int, default=200)
    parser.add_argument("--sim-reps", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="artifact directory override (default: storage/benchmarks)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Regenerate the LaTeX table from the existing "
        "component_2b_cluster_usage_results.json without re-running any "
        "measurement. The artifact is left untouched.",
    )
    args = parser.parse_args()

    if args.table_only:
        artifact_path = BENCHMARK_DIR / "component_2b_cluster_usage_results.json"
        if not artifact_path.exists():
            print(f"ERROR: {artifact_path} not found; run the benchmark first.")
            return
        artifact = json.loads(artifact_path.read_text())
        summary = ClusterUsageSummary(**artifact["summary"])
        table_path = BENCHMARK_DIR / "tables" / "table2b_cluster_uniformity.tex"
        generate_latex_table(
            summary,
            table_path,
            commit=artifact.get("provenance", {}).get("git_commit"),
        )
        print(f"Regenerated LaTeX table: {table_path}")
        return

    if args.quick:
        args.messages = 20
        args.sim_reps = 200

    out_dir = Path(args.output_dir) if args.output_dir else BENCHMARK_DIR
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"

    print("=" * 80)
    print(" COMPONENT 2b: TRANSMITTED CLUSTER USAGE AND MECHANISM NULL")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    summary, observed_counts, expected_scaled, per_modality_counts, density, diagnostics = (
        run_component2b(args.messages, args.sim_reps)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / "fig_2b_cluster_usage.png"
    create_plot(observed_counts, expected_scaled, summary, plot_path)

    table_path = tables_dir / "table2b_cluster_uniformity.tex"
    artifact_provenance = provenance()
    generate_latex_table(
        summary, table_path, commit=artifact_provenance.get("git_commit")
    )

    output_json = out_dir / "component_2b_cluster_usage_results.json"
    payload = {
        "schema": "component2b-summary-v2",
        "provenance": artifact_provenance,
        "config": {
            "seed": SEED,
            "device": DEVICE,
            "num_messages": summary.num_messages,
            "sim_reps": args.sim_reps,
        },
        "summary": asdict(summary),
        "observed_counts": observed_counts.tolist(),
        "expected_counts_mechanism": [float(v) for v in expected_scaled],
        "per_modality_counts": {
            mod: per_modality_counts[mod].tolist() for mod in MODALITIES
        },
        "diagnostics": diagnostics,
        "plot_path": str(plot_path),
        "table_path": str(table_path),
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    write_manifest(
        build_manifest(
            component="component_2b_transmitted_cluster_usage",
            outputs=[
                output_json,
                tables_dir / "table2b_cluster_uniformity.tex",
                plots_dir / "fig_2b_cluster_usage.png",
            ],
            inputs=sorted(
                (PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")
            ),
            seeds={"seed": SEED},
            extra={"config": payload.get("config", {})},
        ),
        out_dir / "component_2b_cluster_usage_results_manifest.json",
    )

    print("\n" + "=" * 80)
    print(" COMPONENT 2b COMPLETE")
    print("=" * 80)
    print(f"  JSON Results: {output_json}")
    print(f"  LaTeX Table:  {table_path}")
    print(f"  Plot:         {plot_path}")


if __name__ == "__main__":
    main()

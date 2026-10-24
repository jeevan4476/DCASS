#!/usr/bin/env python3
"""
Component 4c: ECC Failure-Mode Decomposition and Channel Loss Anatomy

Component 4b records an RS-ECC recovery curve that falls from 100 percent to
70 percent at 5 percent carrier loss and collapses to zero beyond 10 percent.
That curve is quoted often but was never decomposed. This benchmark explains
the shape mechanically:

1. Substitution condition: carriers are corrupted in place, exactly as in
   component 4b, but the recovered bitstream is compared against the clean
   bitstream to count the exact number of damaged codeword BYTES per trial.
   The claim under test is that recovery succeeds in exactly those trials
   where damaged bytes <= t = k/2, which turns the recovery curve into a
   count of a simple random variable rather than a codec mystery.

2. Dropout condition: carriers are DELETED instead of substituted. DSSC
   extracts symbols by positional enumeration of the received list, so a
   single deletion shifts every following chunk index. This condition
   measures how destructive real packet loss is compared to in-place
   corruption, which is the honest channel model for a network adversary
   that can drop traffic.

3. Algebraic-layer condition: the same Reed-Solomon configuration used by
   the codec is exercised directly with known byte errors, threshold
   t = k/2, versus known erasures, threshold k. This shows the algebraic
   headroom that the end-to-end path does not currently exploit, because
   the wire format carries no per-carrier sequence markers.

Outputs:
    storage/benchmarks/component_4c_ecc_failure_results.json
    storage/benchmarks/plots/fig_4c_ecc_failure_modes.png
    storage/benchmarks/tables/table4c_ecc_failure_modes.tex

Usage:
    .venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py
    .venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py --quick
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reedsolo

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    recorded_commit,
    scale_table_to_textwidth,
    write_manifest,
)
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.semantic_engine import SemanticEngine
from src.engine.ecc import RSErrorCorrection
from src.engine.payload_framing import frame_payload
from src.engine.dssc_state_space import DSSCStateSpace, derive_session_permutation
from src.engine.dssc_decoder import BitStreamWriter

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"
SCHEMA = "component4c-ecc-failure-v2"
PARITY_BYTES = 8

MODALITIES = ["image", "text", "audio"]

MODALITY_FAMILY_MAP: Dict[str, List[str]] = {
    "image": ["nature_outdoor", "urban_architecture"],
    "text": ["people_interaction", "objects_indoor"],
    "audio": ["technology_science", "sound_atmosphere"],
}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def collect_provenance(engine: SemanticEngine) -> Dict[str, object]:
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = None
    return {
        "retrospective": False,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script": "scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
        "command": " ".join(sys.argv),
        "seed": SEED,
        "index": engine.index.status(),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


# ---------------------------------------------------------------------------
# Instrumentation: mirror the decoder's symbol extraction to compare clean
# versus corrupted bitstreams byte for byte.
# ---------------------------------------------------------------------------
class SymbolReplay:
    """
    Rebuilds the DSSC symbol stream using the same public building blocks the
    decoder uses: canonical candidate pools per modality and family, the
    HMAC-keyed family digest, the session permutation, and the variable width
    symbol read. Verification against the constructed codeword on the clean
    carrier list guards the mirror against implementation drift.
    """

    def __init__(self, engine: SemanticEngine, modalities: List[str]):
        index = engine.index
        self.modalities = modalities
        self.mapper = engine._vcp_mapper
        self.family_manager = engine._dssc_decoder.family_manager

        all_ids_by_modality: Dict[str, List[str]] = {}
        for mod, meta_list in index.metadata.items():
            if modalities and mod not in modalities:
                continue
            ids = sorted(m.get("id") for m in meta_list if m.get("id"))
            all_ids_by_modality[mod] = ids
        self.all_ids_by_modality = all_ids_by_modality

        all_ids: List[str] = []
        for ids in all_ids_by_modality.values():
            all_ids.extend(ids)
        all_ids.sort()
        self.all_ids = all_ids

        self.available_modalities = [
            m for m in ["text", "image", "audio"] if all_ids_by_modality.get(m)
        ] or [m for m in all_ids_by_modality if all_ids_by_modality[m]] or ["image"]

        self._pool_cache: Dict[Tuple[str, str], List[str]] = {}

    def _pool(self, modality: str, family) -> List[str]:
        key = (modality, family.name)
        if key not in self._pool_cache:
            allowed = set(family.cluster_ids)
            cand = [
                cid for cid in self.all_ids_by_modality.get(modality, [])
                if self.mapper.symbol_for_media_id(cid) in allowed
            ]
            if len(cand) < 8:
                cand = [
                    cid for cid in self.all_ids
                    if self.mapper.symbol_for_media_id(cid) in allowed
                ]
            if len(cand) < 8:
                cand = self.all_ids[:256]
            self._pool_cache[key] = cand
        return self._pool_cache[key]

    def chunk_layout(self, carrier_ids: List[str], session_key: bytes):
        """Return list of (pool, eta, state_space) per chunk index."""
        layout = []
        for chunk_idx in range(len(carrier_ids)):
            digest = hmac.new(
                session_key, f"family:{chunk_idx}".encode("utf-8"), hashlib.sha256
            ).digest()
            modality = self.available_modalities[chunk_idx % len(self.available_modalities)]
            native = MODALITY_FAMILY_MAP.get(modality, [])
            matching = [f for f in self.family_manager.families if f.name in native]
            admissible = matching or self.family_manager.families
            family = admissible[int.from_bytes(digest[:4], "big") % len(admissible)]
            cand = self._pool(modality, family)
            eta = max(2, min(16, int(math.floor(math.log2(len(cand))))))
            perm = derive_session_permutation(
                len(cand), session_key, context_salt=f"dssc:{chunk_idx}:{family.name}"
            )
            ss = DSSCStateSpace(
                chunk_index=chunk_idx,
                family_name=family.name,
                candidate_media_ids=cand,
                permuted_indices=perm,
                bits_per_carrier=eta,
            )
            layout.append(ss)
        return layout

    def bitstream_for(self, carrier_ids: List[str], session_key: bytes) -> Tuple[bytes, List[int]]:
        layout = self.chunk_layout(carrier_ids, session_key)
        writer = BitStreamWriter()
        widths = []
        for cid, ss in zip(carrier_ids, layout):
            symbol = ss.media_id_to_symbol(cid)
            if symbol is None:
                symbol = 0
            writer.append_bits(symbol, ss.bits_per_carrier)
            widths.append(ss.bits_per_carrier)
        return writer.to_bytes(), widths


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    total = 0.0
    for i in range(0, min(k, n) + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


# ---------------------------------------------------------------------------
# End-to-end trials
# ---------------------------------------------------------------------------
@dataclass
class E2EResult:
    condition: str
    carrier_loss_rate: float
    num_trials: int
    num_recovered: int
    recovery_rate_pct: float
    recovery_ci_low_pct: float
    recovery_ci_high_pct: float
    mean_carriers: float
    mean_received_carriers: float
    mean_corrupted: float
    mean_damaged_bytes: float
    mean_damaged_per_loss: float
    mean_fixed_errors: float
    mean_bitstream_length_delta_bytes: float
    num_decoder_successes: int
    num_false_acceptances: int
    num_symbol_changes: int
    symbol_change_pct: float
    trials_damaged_le_t_recovered: int
    trials_damaged_le_t_failed: int
    trials_damaged_le_t_false_acceptances: int
    trials_damaged_gt_t_recovered: int
    trials_damaged_gt_t_accepted: int
    instrumentation_failures: int
    parity_bytes: int
    t_correctable: int
    mean_latency_ms: float
    damaged_samples: List[int] = field(default_factory=list)


def run_e2e_condition(
    engine: SemanticEngine,
    replay: SymbolReplay,
    rng: np.random.Generator,
    all_corpus_ids: List[str],
    loss_rates: List[float],
    num_trials: int,
    condition: str,
) -> List[E2EResult]:
    results: List[E2EResult] = []
    for rate in loss_rates:
        recovered = 0
        decoder_successes = 0
        false_acceptances = 0
        damaged_counts: List[int] = []
        corrupted_counts: List[int] = []
        carrier_counts: List[int] = []
        received_counts: List[int] = []
        length_deltas: List[int] = []
        fixed_counts: List[int] = []
        latencies: List[float] = []
        leak_recovered = 0           # damaged <= t and exact recovery
        leak_failed = 0              # damaged <= t but no exact recovery
        leak_false_accepts = 0       # damaged <= t, CRC passed, wrong plaintext
        over_recovered = 0           # damaged > t and exact recovery
        over_accepted = 0            # damaged > t and any CRC-passing decode
        symbol_changes = 0
        total_corrupted = 0
        instr_failures = 0
        parity_seen: List[int] = []

        for trial in range(num_trials):
            session_key = rng.bytes(32)
            payload = (
                f"DCASS channel stress payload #{trial:03d} - RS fault tolerance verification!"
            )[:50].ljust(50, " ")

            t0 = time.perf_counter()
            enc = engine.encode(
                payload,
                mode="dssc",
                session_key=session_key,
                modalities=MODALITIES,
                use_ecc=True,
                ecc_parity_bytes=PARITY_BYTES,
            )
            carriers = list(enc.media_ids)
            k = int(enc.ecc_parity_bytes)
            parity_seen.append(k)
            if k != PARITY_BYTES:
                raise RuntimeError(f"unexpected parity {k}; expected {PARITY_BYTES}")
            n = len(carriers)
            n_corrupt = min(int(round(n * rate)), n)

            positions = (
                sorted(rng.choice(n, size=n_corrupt, replace=False).tolist())
                if n_corrupt > 0
                else []
            )
            position_set = set(positions)
            corrupted_ids: List[Optional[str]] = []
            for i in range(n):
                if i not in position_set:
                    corrupted_ids.append(carriers[i])
                elif condition == "substitute":
                    new_id = str(rng.choice(all_corpus_ids))
                    while new_id == carriers[i]:
                        new_id = str(rng.choice(all_corpus_ids))
                    corrupted_ids.append(new_id)
                    total_corrupted += 1
                    if (
                        replay.mapper.symbol_for_media_id(carriers[i])
                        != replay.mapper.symbol_for_media_id(new_id)
                    ):
                        symbol_changes += 1
                # dropout: position omitted entirely

            received = [cid for cid in corrupted_ids if cid is not None]

            dec = engine.decode(
                received,
                mode="dssc",
                session_key=session_key,
                modalities=MODALITIES,
                ecc_parity_bytes=k,
                use_ecc=True,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            decoder_success = bool(dec.success)
            is_recovered = bool(decoder_success and dec.reconstructed_message == payload)
            false_accept = bool(decoder_success and not is_recovered)
            if decoder_success:
                decoder_successes += 1
            if false_accept:
                false_acceptances += 1

            # Instrumentation: clean versus received bitstreams
            clean_bits, _ = replay.bitstream_for(carriers, session_key)
            recv_bits, _ = replay.bitstream_for(received, session_key)
            expected = RSErrorCorrection(parity_bytes=k).encode(frame_payload(payload))
            if clean_bits != expected:
                instr_failures += 1

            span = max(len(clean_bits), len(recv_bits))
            damaged = 0
            for bi in range(span):
                a = clean_bits[bi] if bi < len(clean_bits) else -1
                b = recv_bits[bi] if bi < len(recv_bits) else -1
                if a != b:
                    damaged += 1

            t_correct = k // 2
            if decoder_success and damaged > t_correct:
                over_accepted += 1
            if is_recovered:
                recovered += 1
                if damaged <= t_correct:
                    leak_recovered += 1
                else:
                    over_recovered += 1
            elif damaged <= t_correct:
                leak_failed += 1
                if false_accept:
                    leak_false_accepts += 1

            damaged_counts.append(damaged)
            corrupted_counts.append(n_corrupt)
            carrier_counts.append(n)
            received_counts.append(len(received))
            length_deltas.append(len(clean_bits) - len(recv_bits))
            fixed_counts.append(len(dec.ecc_fixed_errors))
            latencies.append(latency_ms)

        if len(set(parity_seen)) != 1:
            raise RuntimeError(f"parity changed across trials: {sorted(set(parity_seen))}")
        if instr_failures:
            raise RuntimeError(
                f"instrumentation mirror disagreed with the clean codeword in "
                f"{instr_failures}/{num_trials} trials"
            )

        k_value = parity_seen[0]
        ci_low, ci_high = wilson_interval(recovered, num_trials)
        mean_damaged = float(np.mean(damaged_counts))
        mean_corrupt = float(np.mean(corrupted_counts))
        results.append(
            E2EResult(
                condition=condition,
                carrier_loss_rate=rate,
                num_trials=num_trials,
                num_recovered=recovered,
                recovery_rate_pct=100.0 * recovered / num_trials,
                recovery_ci_low_pct=ci_low,
                recovery_ci_high_pct=ci_high,
                mean_carriers=float(np.mean(carrier_counts)),
                mean_received_carriers=float(np.mean(received_counts)),
                mean_corrupted=mean_corrupt,
                mean_damaged_bytes=mean_damaged,
                mean_damaged_per_loss=(
                    mean_damaged / mean_corrupt if mean_corrupt > 0 else 0.0
                ),
                mean_fixed_errors=float(np.mean(fixed_counts)),
                mean_bitstream_length_delta_bytes=float(np.mean(length_deltas)),
                num_decoder_successes=decoder_successes,
                num_false_acceptances=false_acceptances,
                num_symbol_changes=symbol_changes,
                symbol_change_pct=(
                    100.0 * symbol_changes / total_corrupted if total_corrupted else 0.0
                ),
                trials_damaged_le_t_recovered=leak_recovered,
                trials_damaged_le_t_failed=leak_failed,
                trials_damaged_le_t_false_acceptances=leak_false_accepts,
                trials_damaged_gt_t_recovered=over_recovered,
                trials_damaged_gt_t_accepted=over_accepted,
                instrumentation_failures=instr_failures,
                parity_bytes=k_value,
                t_correctable=k_value // 2,
                mean_latency_ms=float(np.mean(latencies)),
                damaged_samples=damaged_counts,
            )
        )
        print(
            f"  [{condition:10s}] loss {rate*100:5.1f}% | recovered {recovered:3d}/{num_trials}"
            f" ({100.0*recovered/num_trials:5.1f}%) | accepted {decoder_successes:3d}"
            f" / wrong {false_acceptances:2d} | carriers {np.mean(carrier_counts):5.1f}"
            f" | received {np.mean(received_counts):5.1f} | corrupted {mean_corrupt:4.1f}"
            f" | damaged bytes {mean_damaged:5.2f}"
            f" ({results[-1].mean_damaged_per_loss:4.2f}/loss) | t={k_value//2}"
            f" | symbol changes {symbol_changes}/{total_corrupted}"
            f" | leak(<=t): {leak_recovered} rec / {leak_failed} fail / "
            f"{leak_false_accepts} wrong"
            f" | over(>t): {over_recovered} exact / {over_accepted} accepted"
            f" | instr bad: {instr_failures}"
        )
    return results


# ---------------------------------------------------------------------------
# Algebraic-layer experiment
# ---------------------------------------------------------------------------
@dataclass
class AlgebraicPoint:
    injected: int
    error_successes: int
    error_repeats: int
    error_success_pct: float
    erasure_successes: int
    erasure_repeats: int
    erasure_success_pct: float


def run_algebraic_benchmark(rng: np.random.Generator, reps: int = 40) -> Tuple[List[AlgebraicPoint], Dict[str, int]]:
    payload = "Algebraic-layer codeword test payload for RS threshold measurement!"[:50].ljust(50, " ")
    framed = frame_payload(payload)
    k = 8
    rs = RSErrorCorrection(parity_bytes=k)
    codeword = rs.encode(framed)
    n = len(codeword)

    points: List[AlgebraicPoint] = []
    for s in range(0, 13):
        err_ok = 0
        for _ in range(reps):
            c = bytearray(codeword)
            positions = rng.choice(n, size=min(s, n), replace=False)
            for p in positions:
                new = int(rng.integers(0, 256))
                while new == c[p]:
                    new = int(rng.integers(0, 256))
                c[p] = new
            data, ok, _ = RSErrorCorrection(parity_bytes=k).decode_bytes(bytes(c))
            if ok and data == framed:
                err_ok += 1

        era_ok = 0
        for _ in range(reps):
            c = bytearray(codeword)
            positions = rng.choice(n, size=min(s, n), replace=False)
            for p in positions:
                c[p] = 0
            try:
                data, _, _ = rs._codec.decode(
                    bytearray(c), erase_pos=[int(p) for p in positions]
                )
                ok = bytes(data) == framed
            except reedsolo.ReedSolomonError:
                ok = False
            if ok:
                era_ok += 1

        points.append(
            AlgebraicPoint(
                injected=s,
                error_successes=err_ok,
                error_repeats=reps,
                error_success_pct=100.0 * err_ok / reps,
                erasure_successes=era_ok,
                erasure_repeats=reps,
                erasure_success_pct=100.0 * era_ok / reps,
            )
        )
        print(
            f"  [algebraic] injected {s:2d} | errors {err_ok:3d}/{reps} ok"
            f" | erasures {era_ok:3d}/{reps} ok"
        )

    meta = {
        "codeword_bytes": n,
        "framed_bytes": len(framed),
        "parity_bytes": k,
        "t_errors": k // 2,
        "t_erasures": k,
    }
    return points, meta


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def create_plots(
    sub_results: List[E2EResult],
    drop_results: List[E2EResult],
    algebraic_points: List[AlgebraicPoint],
    algebraic_meta: Dict[str, int],
    output_path: Path,
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.4))

    # Panel 1: recovery vs carrier loss
    for results, color, label, marker in [
        (sub_results, "#27AE60", "Substitution (corrupted in place)", "o"),
        (drop_results, "#E85D5D", "Dropout (carriers deleted)", "s"),
    ]:
        rates = [r.carrier_loss_rate * 100 for r in results]
        rec = [r.recovery_rate_pct for r in results]
        lo = [r.recovery_rate_pct - r.recovery_ci_low_pct for r in results]
        hi = [r.recovery_ci_high_pct - r.recovery_rate_pct for r in results]
        ax1.errorbar(
            rates, rec, yerr=[lo, hi], marker=marker, color=color,
            linewidth=2.2, markersize=7, capsize=3, label=label,
        )
    # Erasure-bound reference: with known positions the same measured damage is
    # correctable up to k byte erasures instead of t = k/2 byte errors.
    k_ref = sub_results[0].parity_bytes
    rates_pts = [r.carrier_loss_rate * 100 for r in sub_results]
    bound = [
        100.0
        * sum(1 for d in r.damaged_samples if d <= k_ref)
        / max(1, len(r.damaged_samples))
        for r in sub_results
    ]
    ax1.plot(
        rates_pts, bound, "--^", color="#2E86AB", linewidth=2.0, markersize=6,
        label=f"Known-position erasure bound from measured damage (k={k_ref})",
    )
    ax1.set_xlabel("Carrier loss rate (%)", fontsize=12)
    ax1.set_ylabel("Message recovery rate (%)", fontsize=12)
    ax1.set_title("Recovery vs Carrier Loss", fontsize=13, fontweight="medium")
    ax1.set_ylim(-5, 108)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left")

    # Panel 2: damaged byte count distributions under substitution
    rates = sorted(set(r.carrier_loss_rate for r in sub_results))
    grouped = {r.carrier_loss_rate: r for r in sub_results}
    box_data = [grouped[r].damaged_samples for r in rates]
    bp = ax2.boxplot(
        box_data, positions=[r * 100 for r in rates], widths=1.1,
        patch_artist=True, medianprops=dict(color="#2C3E50", linewidth=2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#AED6F1")
        patch.set_alpha(0.85)
    ax2.axhline(
        sub_results[0].t_correctable, color="#E85D5D", linestyle="--",
        label=f"RS correction limit t={sub_results[0].t_correctable} damaged bytes",
    )
    # Annotate mean damage and mean damage per corrupted carrier above boxes
    for r in rates:
        row = grouped[r]
        ax2.annotate(
            f"{row.mean_damaged_bytes:.1f} B\n{row.mean_damaged_per_loss:.2f}/loss",
            (r * 100, max(1.0, max(row.damaged_samples) if row.damaged_samples else 1.0)),
            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5,
        )
    ax2.set_xlabel("Carrier loss rate (%)", fontsize=12)
    ax2.set_ylabel("Damaged codeword bytes per trial", fontsize=12)
    ax2.set_title("Why the Cliff: Damage Distribution Crossing t", fontsize=13, fontweight="medium")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    # Panel 3: algebraic error vs erasure thresholds
    xs = [p.injected for p in algebraic_points]
    err = [p.error_success_pct for p in algebraic_points]
    era = [p.erasure_success_pct for p in algebraic_points]
    ax3.step(xs, err, where="post", color="#E85D5D", linewidth=2.2, label="Byte errors injected")
    ax3.step(xs, era, where="post", color="#2E86AB", linewidth=2.2, label="Known-position erasures")
    ax3.axvline(
        algebraic_meta["t_errors"], color="#C0392B", linestyle=":",
        label=f"Error limit t={algebraic_meta['t_errors']}",
    )
    ax3.axvline(
        algebraic_meta["t_erasures"], color="#1F618D", linestyle=":",
        label=f"Erasure limit k={algebraic_meta['t_erasures']}",
    )
    ax3.set_xlabel("Injected byte damage", fontsize=12)
    ax3.set_ylabel("Correct decode rate (%)", fontsize=12)
    ax3.set_title("Algebraic Limits of the Same RS(63,55)", fontsize=13, fontweight="medium")
    ax3.set_ylim(-5, 108)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved failure-mode plot to {output_path}")


def generate_latex_table(
    sub_results: List[E2EResult],
    drop_results: List[E2EResult],
    algebraic_points: List[AlgebraicPoint],
    algebraic_meta: Dict[str, int],
    output_path: Path,
    commit: str | None = None,
):
    false_accepts = sum(
        result.num_false_acceptances for result in [*sub_results, *drop_results]
    )
    if false_accepts == 0:
        acceptance_note = (
            "No CRC-passing wrong plaintext occurred in the recorded trials after "
            "the framed-payload hardening."
        )
    else:
        acceptance_note = (
            f"{false_accepts} CRC-passing wrong plaintext decodes occurred; "
            "exact recovery and acceptance therefore differ."
        )
    provenance = caption_provenance(
        n=f"{len(sub_results)} substitution and {len(drop_results)} dropout conditions, "
        f"{algebraic_points[0].error_repeats if algebraic_points else 0} repeats per algebraic point",
        scope="post-hardening artifact; framed payloads required when ECC is enabled",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_4c_ecc_failure_results.json"),
    )
    with open(output_path, "w") as f:
        f.write("% Component 4c: ECC Failure-Mode Decomposition and Channel Loss Anatomy\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(
            "\\caption{ECC failure anatomy: substitution versus dropout versus algebraic limits. "
            f"Codeword is RS({algebraic_meta['codeword_bytes']},{algebraic_meta['framed_bytes']}) "
            f"correcting {algebraic_meta['t_errors']} byte errors or "
            f"{algebraic_meta['t_erasures']} known erasures. Exact is plaintext recovery; "
            f"accepted is any CRC-passing decode. {acceptance_note}{provenance}}}\n"
        )
        f.write("\\label{tab:ecc_failure_modes}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write(
            "Condition & Loss & Exact & Accepted & 95\\% CI & Damaged bytes & "
            "Per affected carrier \\\\\n"
        )
        f.write("\\midrule\n")
        for rows, name in [(sub_results, "Substitution"), (drop_results, "Dropout")]:
            for r in rows:
                accepted_pct = 100.0 * r.num_decoder_successes / r.num_trials
                f.write(
                    f"{name} & {r.carrier_loss_rate*100:.0f}\\% & "
                    f"{r.recovery_rate_pct:.1f}\\% & {accepted_pct:.1f}\\% & "
                    f"[{r.recovery_ci_low_pct:.1f}, {r.recovery_ci_high_pct:.1f}] & "
                    f"{r.mean_damaged_bytes:.1f} & {r.mean_damaged_per_loss:.2f} \\\\\n"
                )
        f.write("\\midrule\n")
        f.write("\\multicolumn{7}{c}{\\textbf{Algebraic layer, known positions}} \\\\\n")
        f.write("Injected bytes & Error success & Erasure success & & & & \\\\\n")
        for p in algebraic_points:
            f.write(
                f"{p.injected} & {p.error_success_pct:.0f}\\% & "
                f"{p.erasure_success_pct:.0f}\\% & & & & \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        f.flush()
        scale_table_to_textwidth(output_path)
    print(f"Saved LaTeX table to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke test with few trials")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate plot and table from the saved JSON without rerunning trials",
    )
    args = parser.parse_args()

    if args.replot:
        output_json = BENCHMARK_DIR / "component_4c_ecc_failure_results.json"
        payload = json.loads(output_json.read_text())
        sub_results = [E2EResult(**row) for row in payload["substitution_results"]]
        drop_results = [E2EResult(**row) for row in payload["dropout_results"]]
        algebraic_points = [AlgebraicPoint(**row) for row in payload["algebraic_points"]]
        algebraic_meta = payload["algebraic_meta"]
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = PLOTS_DIR / "fig_4c_ecc_failure_modes.png"
        create_plots(sub_results, drop_results, algebraic_points, algebraic_meta, plot_path)
        table_path = TABLES_DIR / "table4c_ecc_failure_modes.tex"
        generate_latex_table(
            sub_results,
            drop_results,
            algebraic_points,
            algebraic_meta,
            table_path,
        )
        payload["plot_path"] = _relative(plot_path)
        payload["table_path"] = _relative(table_path)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Regenerated plot from {output_json}")
        return

    if args.quick:
        sub_rates = [0.0, 0.05]
        drop_rates = [0.0, 0.02]
        trials = 3
        algebraic_reps = 3
    else:
        sub_rates = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        drop_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        trials = args.trials
        algebraic_reps = 40

    print("=" * 80)
    print(" COMPONENT 4c: ECC FAILURE-MODE DECOMPOSITION AND CHANNEL LOSS ANATOMY")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("\nLoading SemanticEngine and FAISS indices...")
    engine = SemanticEngine()
    engine.load()

    replay = SymbolReplay(engine, MODALITIES)
    all_corpus_ids = []
    for mod in MODALITIES:
        for m in engine.index.metadata.get(mod, []):
            if m.get("id"):
                all_corpus_ids.append(m["id"])
    print(f"Corpus IDs available for substitution: {len(all_corpus_ids):,}")

    print("\n--- Part 1: End-to-end substitution condition ---")
    sub_results = run_e2e_condition(
        engine, replay, rng, all_corpus_ids, sub_rates, trials, "substitute"
    )

    print("\n--- Part 2: End-to-end dropout condition ---")
    drop_results = run_e2e_condition(
        engine, replay, rng, all_corpus_ids, drop_rates, trials, "dropout"
    )

    print("\n--- Part 3: Algebraic layer, errors versus known erasures ---")
    algebraic_points, algebraic_meta = run_algebraic_benchmark(rng, reps=algebraic_reps)
    print(
        f"  Codeword RS({algebraic_meta['codeword_bytes']},{algebraic_meta['framed_bytes']}),"
        f" parity {algebraic_meta['parity_bytes']}, error limit t={algebraic_meta['t_errors']},"
        f" erasure limit k={algebraic_meta['t_erasures']}"
    )

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    plot_path = PLOTS_DIR / "fig_4c_ecc_failure_modes.png"
    create_plots(sub_results, drop_results, algebraic_points, algebraic_meta, plot_path)

    table_path = TABLES_DIR / "table4c_ecc_failure_modes.tex"
    output_json = BENCHMARK_DIR / "component_4c_ecc_failure_results.json"
    payload = {
        "schema": SCHEMA,
        "provenance": collect_provenance(engine),
        "config": {
            "seed": SEED,
            "device": str(DEVICE),
            "payload_chars": 50,
            "trials_per_point": trials,
            "substitution_rates": sub_rates,
            "dropout_rates": drop_rates,
            "modalities": MODALITIES,
            "parity_bytes": PARITY_BYTES,
            "t_correctable": PARITY_BYTES // 2,
            "channel_model": (
                "Substitution replaces a carrier with a different random corpus ID, "
                "including replacements that map to the same VCP symbol. Dropout removes "
                "the carrier from the received list."
            ),
            "framing_policy": (
                "Framed payloads are required whenever ECC is enabled. Legacy raw "
                "payloads remain decodable only when ECC is off. This prevents a "
                "deletion-shifted codeword from being accepted as an unframed message."
            ),
        },
        "decoder_hardening": {
            "applied": True,
            "policy": "require_framed=True when ECC is enabled",
            "pre_hardening_artifact": (
                "storage/benchmarks/archive/"
                "component_4c_ecc_failure_results_pre_hardening.json"
            ),
            "pre_hardening_dropout_false_acceptances": 41,
            "pre_hardening_dropout_trials_total": 480,
            "pre_hardening_dropout_trials_with_false_accepts": 320,
        },
        "metric_definitions": {
            "exact_recovery": (
                "Decoder success plus exact plaintext equality. This is the recovery "
                "metric in the substitution and dropout tables."
            ),
            "accepted": (
                "Any decode for which the CRC gate returns success, including a wrong "
                "plaintext. The table reports accepted separately from exact."
            ),
            "false_acceptance": (
                "A decode with success true whose reconstructed plaintext differs from "
                "the payload. The decoder must return zero such decodes in both "
                "conditions."
            ),
            "damaged_bytes": (
                "Count of differing bytes between the clean and received bitstreams. "
                "For dropout, missing trailing bytes are counted as differences; the "
                "metric is not a sufficient statistic for alignment failures."
            ),
            "symbol_changes": (
                "Substitution replacements whose decoded VCP symbol differs from the "
                "original carrier. Same-symbol replacements are no-ops in the bitstream."
            ),
            "intervals": "Wilson 95 percent intervals.",
        },
        "substitution_results": [asdict(r) for r in sub_results],
        "dropout_results": [asdict(r) for r in drop_results],
        "algebraic_points": [asdict(p) for p in algebraic_points],
        "algebraic_meta": algebraic_meta,
        "plot_path": _relative(plot_path),
        "table_path": _relative(table_path),
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    # The table is generated after the artifact is written, so the commit its
    # caption carries is the one that produced the measurement in the
    # artifact rather than the previous run's.
    generate_latex_table(
        sub_results,
        drop_results,
        algebraic_points,
        algebraic_meta,
        table_path,
    )
    write_manifest(
        build_manifest(
            component="component_4c_ecc_failure_analysis",
            outputs=[
                output_json,
                table_path,
                plot_path,
            ],
            inputs=sorted((PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")),
            seeds={"seed": SEED},
            extra={"config": payload.get("config", {})},
        ),
        BENCHMARK_DIR / "component_4c_ecc_failure_results_manifest.json",
    )

    print("\n" + "=" * 80)
    print(" COMPONENT 4c COMPLETE")
    print("=" * 80)
    print(f"  JSON Results: {output_json}")
    print(f"  LaTeX Table:  {table_path}")
    print(f"  Plot:         {plot_path}")


if __name__ == "__main__":
    main()

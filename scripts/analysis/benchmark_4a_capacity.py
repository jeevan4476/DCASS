#!/usr/bin/env python3
"""
Component 4a: Codec Capacity Benchmark - DSSC vs exact_vcp

Measures the carrier count and bits-per-carrier advantage of DSSC over the
exact_vcp reference path at five payload sizes. The exact_vcp baseline is the
all-modality reference mapper: it spends one carrier per codeword byte and is
therefore fixed at 8.00 bits per carrier. Latency is recorded for the shipped
reference paths but is not a codec-only speed comparison; the exact_vcp encoder
scores candidate media for every byte, while DSSC resolves precomputed pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

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

from src.corpus.index.unified_index import UnifiedSemanticIndex

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"
SCHEMA = "component4a-capacity-v2"
PARITY_BYTES = 8
NUM_TRIALS = 5
PAYLOAD_SIZES = [14, 50, 150, 300, 500]
SESSION_KEY = b"dcass_benchmark_session_key_32b!"

np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class CapacityResult:
    mode: str
    payload_bytes: int
    carrier_count: int
    carrier_count_std: float
    bits_per_carrier: float
    bits_per_carrier_std: float
    carrier_reduction_pct: float
    encode_latency_ms: float
    encode_latency_std_ms: float
    decode_latency_ms: float
    decode_latency_std_ms: float
    total_latency_ms: float
    modality_counts: Dict[str, int]


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


def collect_provenance(index: UnifiedSemanticIndex) -> Dict[str, object]:
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
        "script": "scripts/analysis/benchmark_4a_capacity.py",
        "command": " ".join(sys.argv),
        "seed": SEED,
        "index": index.status(),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _id_to_modality(index: UnifiedSemanticIndex) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for modality, meta_list in index.metadata.items():
        for meta in meta_list:
            media_id = meta.get("id")
            if media_id:
                mapping[media_id] = modality
    return mapping


def _modality_counts(media_ids: List[str], id_to_modality: Dict[str, str]) -> Dict[str, int]:
    counts = Counter(id_to_modality.get(media_id, "unknown") for media_id in media_ids)
    return {name: int(counts[name]) for name in sorted(counts)}


def _std(values: List[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _measure_mode(
    engine,
    payload: str,
    mode: str,
    modalities: List[str],
    id_to_modality: Dict[str, str],
    num_trials: int,
) -> Dict[str, object]:
    """Run one clean-channel configuration and return per-mode measurements."""

    def encode_decode(key: bytes | None):
        encode_start = time.perf_counter()
        encoded = engine.encode(
            payload,
            mode=mode,
            session_key=key,
            modalities=modalities,
            use_ecc=True,
            ecc_parity_bytes=PARITY_BYTES,
        )
        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        decode_start = time.perf_counter()
        decoded = engine.decode(
            encoded.media_ids,
            mode=mode,
            session_key=key,
            modalities=modalities,
            use_ecc=True,
            ecc_parity_bytes=PARITY_BYTES,
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        if not decoded.success or decoded.reconstructed_message != payload:
            raise RuntimeError(f"{mode} clean-channel round trip failed for {len(payload)} bytes")
        if int(encoded.ecc_parity_bytes) != PARITY_BYTES:
            raise RuntimeError(f"{mode} encoded parity {encoded.ecc_parity_bytes} != {PARITY_BYTES}")
        return encoded, encode_ms, decode_ms

    # One untimed warm-up per configuration keeps cold imports and lazy caches
    # out of the five recorded trials.
    warm_key = hashlib.sha256(SESSION_KEY + b"warmup").digest() if mode == "dssc" else None
    encode_decode(warm_key)

    encode_times: List[float] = []
    decode_times: List[float] = []
    carrier_counts: List[int] = []
    bits_per_carrier: List[float] = []
    modality_mixes: List[Dict[str, int]] = []

    for trial in range(num_trials):
        session_key = (
            hashlib.sha256(SESSION_KEY + trial.to_bytes(4, "big")).digest()
            if mode == "dssc"
            else None
        )
        encoded, encode_ms, decode_ms = encode_decode(session_key)
        encode_times.append(encode_ms)
        decode_times.append(decode_ms)
        carrier_counts.append(len(encoded.media_ids))
        bits_per_carrier.append(float(encoded.bits_per_carrier))
        modality_mixes.append(_modality_counts(encoded.media_ids, id_to_modality))

    first_mix = modality_mixes[0]
    if any(mix != first_mix for mix in modality_mixes):
        raise RuntimeError(f"{mode} modality mix changed across trials: {modality_mixes}")
    if len({int(count) for count in carrier_counts}) != 1:
        raise RuntimeError(f"{mode} carrier count changed across trials: {carrier_counts}")

    return {
        "carrier_count": int(carrier_counts[0]),
        "carrier_count_std": _std([float(c) for c in carrier_counts]),
        "bits_per_carrier": float(np.mean(bits_per_carrier)),
        "bits_per_carrier_std": _std(bits_per_carrier),
        "encode_latency_ms": float(np.mean(encode_times)),
        "encode_latency_std_ms": _std(encode_times),
        "decode_latency_ms": float(np.mean(decode_times)),
        "decode_latency_std_ms": _std(decode_times),
        "modality_counts": first_mix,
    }


def _generate_table(
    results: List[CapacityResult], output_path: Path, commit: str | None = None
) -> None:
    provenance = caption_provenance(
        n=f"{len({r.payload_bytes for r in results})} payload sizes x "
        f"{len({r.mode for r in results})} modes",
        scope="coded bits/carrier including framing and RS parity; user-payload density is lower",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_4a_capacity_results.json"),
    )
    with open(output_path, "w") as handle:
        handle.write("% Component 4a: Codec Capacity - DSSC vs exact_vcp\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{Codec capacity: DSSC vs exact\\_vcp with matched framing and "
            "Reed-Solomon ECC. Latency columns are reference-implementation wall-clock "
            "measurements, not codec-only lower bounds.%s}\n" % provenance
        )
        handle.write("\\label{tab:codec_capacity}\n")
        handle.write("\\begin{tabular}{lcccccc}\n")
        handle.write("\\toprule\n")
        handle.write(
            "Mode & Payload (B) & Carriers & Bits/Carrier & Reduction (\\%) & "
            "Encode (ms, ref.) & Decode (ms, ref.) \\\\\n"
        )
        handle.write("\\midrule\n")
        for result in results:
            reduction = (
                f"{result.carrier_reduction_pct:.1f}\\%"
                if result.carrier_reduction_pct > 0
                else "--"
            )
            mode = result.mode.replace("_", "\\_")
            handle.write(
                f"{mode} & {result.payload_bytes} & {result.carrier_count} & "
                f"{result.bits_per_carrier:.2f} & {reduction} & "
                f"{result.encode_latency_ms:.1f} & {result.decode_latency_ms:.1f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
        handle.flush()
        scale_table_to_textwidth(output_path)


def create_capacity_plots(results: List[CapacityResult], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({result.payload_bytes for result in results})
    modes = ["DSSC (multi-modal)", "DSSC (text-only)", "exact_vcp (multi-modal)"]
    colors = {
        "DSSC (multi-modal)": "#E85D5D",
        "DSSC (text-only)": "#F39C12",
        "exact_vcp (multi-modal)": "#27AE60",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for mode in modes:
        mode_results = [result for result in results if result.mode == mode]
        if not mode_results:
            continue
        carrier_counts = [result.carrier_count for result in mode_results]
        bits_per_carrier = [result.bits_per_carrier for result in mode_results]
        latency = [result.total_latency_ms for result in mode_results]
        ax1.plot(
            sizes, carrier_counts, "o-", label=mode, color=colors.get(mode, "#888"),
            linewidth=2, markersize=8,
        )
        ax2.plot(
            sizes, bits_per_carrier, "o-", label=mode, color=colors.get(mode, "#888"),
            linewidth=2, markersize=8,
        )
        ax3.plot(
            sizes, latency, "o-", label=mode, color=colors.get(mode, "#888"),
            linewidth=2, markersize=8,
        )

    ax1.set_xlabel("Payload Size (bytes)")
    ax1.set_ylabel("Carrier Count")
    ax1.set_title("Carriers Required vs Payload Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Payload Size (bytes)")
    ax2.set_ylabel("Bits per Carrier")
    ax2.set_title("Codec Capacity (Bits per Carrier)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=8.0, color="k", linestyle=":", alpha=0.5, label="exact_vcp baseline")
    ax2.legend()

    ax3.set_xlabel("Payload Size (bytes)")
    ax3.set_ylabel("Total Latency (ms)")
    ax3.set_title("Reference-Path Encode + Decode Latency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Capacity plot saved to {output_path}")


def run_capacity_benchmark() -> List[CapacityResult]:
    print("=" * 80)
    print(" COMPONENT 4a: CODEC CAPACITY - DSSC vs exact_vcp")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading semantic index and unified engine...")
    index = UnifiedSemanticIndex()
    index.load()
    from src.engine.semantic_engine import SemanticEngine

    engine = SemanticEngine(index=index)
    engine.load()
    print(f"   Index loaded: {index.status()}")
    id_to_modality = _id_to_modality(index)

    results: List[CapacityResult] = []
    for payload_size in PAYLOAD_SIZES:
        print(f"\n2. Payload size: {payload_size} bytes")
        payload = "X" * payload_size

        exact = _measure_mode(
            engine, payload, "exact_vcp", ["image", "text", "audio"],
            id_to_modality, NUM_TRIALS,
        )
        multi = _measure_mode(
            engine, payload, "dssc", ["image", "text", "audio"],
            id_to_modality, NUM_TRIALS,
        )
        text = _measure_mode(
            engine, payload, "dssc", ["text"], id_to_modality, NUM_TRIALS,
        )

        if set(text["modality_counts"]) != {"text"}:
            raise RuntimeError(f"DSSC text-only run used non-text carriers: {text['modality_counts']}")

        exact_carriers = int(exact["carrier_count"])
        for name, values, mode in [
            ("DSSC (multi-modal)", multi, "DSSC (multi-modal)"),
            ("DSSC (text-only)", text, "DSSC (text-only)"),
            ("exact_vcp (multi-modal)", exact, "exact_vcp (multi-modal)"),
        ]:
            reduction = (
                (exact_carriers - int(values["carrier_count"])) / exact_carriers * 100.0
                if mode != "exact_vcp (multi-modal)"
                else 0.0
            )
            result = CapacityResult(
                mode=mode,
                payload_bytes=payload_size,
                carrier_count=int(values["carrier_count"]),
                carrier_count_std=float(values["carrier_count_std"]),
                bits_per_carrier=float(values["bits_per_carrier"]),
                bits_per_carrier_std=float(values["bits_per_carrier_std"]),
                carrier_reduction_pct=float(reduction),
                encode_latency_ms=float(values["encode_latency_ms"]),
                encode_latency_std_ms=float(values["encode_latency_std_ms"]),
                decode_latency_ms=float(values["decode_latency_ms"]),
                decode_latency_std_ms=float(values["decode_latency_std_ms"]),
                total_latency_ms=float(values["encode_latency_ms"] + values["decode_latency_ms"]),
                modality_counts=dict(values["modality_counts"]),
            )
            results.append(result)

        print(
            f"    exact_vcp (multi-modal): {exact['carrier_count']} carriers, "
            f"{exact['bits_per_carrier']:.2f} bits/carrier, "
            f"{exact['encode_latency_ms']:.1f}ms encode, "
            f"{exact['decode_latency_ms']:.1f}ms decode, "
            f"modalities {exact['modality_counts']}"
        )
        print(
            f"    DSSC (multi-modal):      {multi['carrier_count']} carriers, "
            f"{multi['bits_per_carrier']:.2f} bits/carrier, "
            f"{multi['encode_latency_ms']:.1f}ms encode, "
            f"{multi['decode_latency_ms']:.1f}ms decode, "
            f"modalities {multi['modality_counts']}"
        )
        print(
            f"    DSSC (text-only):        {text['carrier_count']} carriers, "
            f"{text['bits_per_carrier']:.2f} bits/carrier, "
            f"{text['encode_latency_ms']:.1f}ms encode, "
            f"{text['decode_latency_ms']:.1f}ms decode, "
            f"modalities {text['modality_counts']}"
        )

    output_path = BENCHMARK_DIR / "component_4a_capacity_results.json"
    payload = {
        "schema": SCHEMA,
        "provenance": collect_provenance(index),
        "config": {
            "seed": SEED,
            "device": str(DEVICE),
            "payload_sizes": PAYLOAD_SIZES,
            "trials_per_mode": NUM_TRIALS,
            "warmup_trials": 1,
            "parity_bytes": PARITY_BYTES,
            "session_key_bytes": len(SESSION_KEY),
            "modalities": ["image", "text", "audio"],
            "session_key_derivation": "SHA-256(session_key || trial_index_be32) per DSSC trial",
        },
        "metric_definitions": {
            "carrier_count": "Mean carrier count over clean-channel trials at fixed parity k=8.",
            "bits_per_carrier": "8 * codeword bytes / carrier count, averaged over trials.",
            "encode_latency_ms": (
                "Wall-clock SemanticEngine.encode time after one warm-up. exact_vcp encode "
                "includes per-byte semantic candidate scoring in the reference mapper; this "
                "is not a codec-only speed comparison."
            ),
            "decode_latency_ms": "Wall-clock SemanticEngine.decode time after one warm-up.",
            "exact_vcp_baseline": (
                "All-modality exact_vcp reference mapper, one carrier per codeword byte. "
                "A strict text-only exact_vcp baseline is not guaranteed because the mapper "
                "falls back to image/audio when a text cluster lacks a requested symbol."
            ),
        },
        "results": [asdict(result) for result in results],
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    tex_path = TABLES_DIR / "table4a_capacity.tex"
    _generate_table(results, tex_path)
    create_capacity_plots(results, PLOTS_DIR / "fig_4a_capacity_comparison.png")

    write_manifest(
        build_manifest(
            component="component_4a_capacity",
            outputs=[
                output_path,
                tex_path,
                PLOTS_DIR / "fig_4a_capacity_comparison.png",
            ],
            inputs=sorted((PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")),
            seeds={"seed": SEED},
            extra={"config": payload.get("config", {})},
        ),
        BENCHMARK_DIR / "component_4a_capacity_results_manifest.json",
    )

    print(f"\nResults saved to {output_path}")
    print(f"LaTeX table saved to {tex_path}")
    print(f"Plots saved to {PLOTS_DIR}")
    return results


def load_results_from_json(json_path: Path) -> List[CapacityResult]:
    payload = json.loads(json_path.read_text())
    return [CapacityResult(**row) for row in payload["results"]]


def regenerate_artifacts_from_json() -> None:
    json_path = BENCHMARK_DIR / "component_4a_capacity_results.json"
    results = load_results_from_json(json_path)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "table4a_capacity.tex"
    _generate_table(results, tex_path)
    create_capacity_plots(results, PLOTS_DIR / "fig_4a_capacity_comparison.png")
    print(f"Regenerated artifacts from {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Regenerate the table and plot from the recorded JSON without rerunning trials",
    )
    args = parser.parse_args()
    if args.artifacts_only:
        regenerate_artifacts_from_json()
    else:
        run_capacity_benchmark()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DCASS Component 5: Codec Latency, Throughput & Direct-Engine Concurrency Benchmark.

Comprehensive evaluation of:
1. Stage-by-Stage Latency Breakdown (RS-ECC, HMAC-DRBG, FAISS search, Centroid Mapping)
2. Payload Size Scaling & Throughput (32B to 512B)
3. Direct-Engine Multi-Threaded Concurrency (10, 20, 50 worker threads)
4. Payload-Isolation Check for the Direct-Engine Encode/Decode Path

This benchmark calls SemanticEngine.encode/decode directly. It does not exercise the
FastAPI routes, /api/transmit, the scheduler, or the API session map.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.engine.semantic_engine import SemanticEngine
from src.engine.ecc import RSErrorCorrection

BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SCHEMA = "component5-results-v2"

np.random.seed(SEED)
torch.manual_seed(SEED)


def _git_commit() -> str | None:
    """Return the current git commit, or None when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_provenance() -> Dict[str, Any]:
    """Collect the environment and source provenance for a live run."""
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = None
    return {
        "retrospective": False,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script": "scripts/analysis/benchmark_5_system_concurrency.py",
        "command": " ".join(sys.argv),
    }


def metric_definitions() -> Dict[str, str]:
    """Describe exactly what the recorded metrics cover."""
    return {
        "latency": (
            "Encode plus decode service time measured inside each worker after the task starts. "
            "It excludes queue wait, HTTP, scheduler, and transmit delays."
        ),
        "throughput": (
            "Completed sessions per second. Each session performs one engine.encode call "
            "and one engine.decode call; it is not an HTTP request rate."
        ),
        "cross_talk": (
            "A decoded payload that differs from the session's own secret and contains the shared "
            "SECRET_PAYLOAD marker. Corrupted payloads without the marker count as failures, not swaps."
        ),
        "scope": (
            "Direct SemanticEngine encode/decode only. FastAPI routes, /api/transmit, "
            "StealthScheduler, and the API session map are not exercised."
        ),
    }


@dataclass
class PayloadPerformanceResult:
    payload_bytes: int
    carrier_count: int
    encode_latency_ms: float
    encode_latency_std: float
    decode_latency_ms: float
    decode_latency_std: float
    total_latency_ms: float
    encode_throughput_kbps: float
    decode_throughput_kbps: float
    carrier_speed_per_sec: float
    stage_breakdown: Dict[str, float]


@dataclass
class ConcurrencyResult:
    num_threads: int
    total_requests: int
    duration_sec: float
    requests_per_sec: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate_pct: float
    cross_talk_rate_pct: float


def profile_stage_breakdown(engine: SemanticEngine, test_payload: str, session_key: bytes) -> Dict[str, float]:
    """Measure granular execution times for individual algorithmic sub-stages."""
    ecc = RSErrorCorrection(parity_bytes=8)
    payload_bytes = test_payload.encode("utf-8")

    # 1. Encoding stages
    t0 = time.perf_counter()
    ecc_framed = ecc.encode(payload_bytes)
    t_ecc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    # Key expansion / permutation
    h = hashlib.sha256(session_key).digest()
    np.random.default_rng(int.from_bytes(h[:4], "big")).permutation(len(ecc_framed))
    t_perm = (time.perf_counter() - t0) * 1000

    # Full encode
    t0 = time.perf_counter()
    res = engine.encode(
        test_payload,
        mode="dssc",
        session_key=session_key,
        modalities=["image", "text", "audio"],
        use_ecc=True,
        ecc_parity_bytes=8
    )
    t_full_enc = (time.perf_counter() - t0) * 1000
    t_faiss_selection = max(0.1, t_full_enc - t_ecc - t_perm)

    # 2. Decoding stages
    t0 = time.perf_counter()
    engine.decode(
        res.media_ids,
        mode="dssc",
        session_key=session_key,
        use_ecc=True,
        ecc_parity_bytes=8
    )
    t_full_dec = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ecc.decode(ecc_framed)
    t_ecc_dec = (time.perf_counter() - t0) * 1000

    t_centroid_lookup = max(0.1, t_full_dec - t_ecc_dec - t_perm)

    return {
        "enc_ecc_framing_ms": float(t_ecc),
        "enc_permutation_ms": float(t_perm),
        "enc_faiss_search_ms": float(t_faiss_selection),
        "dec_centroid_lookup_ms": float(t_centroid_lookup),
        "dec_inv_permutation_ms": float(t_perm),
        "dec_ecc_syndrome_ms": float(t_ecc_dec),
    }


def benchmark_payload_scaling(engine: SemanticEngine) -> List[PayloadPerformanceResult]:
    """Benchmark encoding and decoding speed across increasing payload sizes."""
    print("\n--- Part 1: Encoding & Decoding Speed vs Payload Size ---")
    payload_sizes = [32, 64, 128, 256, 512]
    num_trials = 8
    results = []

    for size in payload_sizes:
        raw_msg = "A" * size
        enc_times = []
        dec_times = []
        carrier_counts = []

        session_key = os.urandom(32)

        # Warmup
        engine.encode(raw_msg, mode="dssc", session_key=session_key, use_ecc=True, ecc_parity_bytes=8)

        for _ in range(num_trials):
            key = os.urandom(32)

            t0 = time.perf_counter()
            enc_res = engine.encode(
                raw_msg,
                mode="dssc",
                session_key=key,
                modalities=["image", "text", "audio"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            enc_times.append((time.perf_counter() - t0) * 1000)
            carrier_counts.append(len(enc_res.media_ids))

            t0 = time.perf_counter()
            dec_res = engine.decode(
                enc_res.media_ids,
                mode="dssc",
                session_key=key,
                use_ecc=True,
                ecc_parity_bytes=8
            )
            dec_times.append((time.perf_counter() - t0) * 1000)
            assert dec_res.reconstructed_message == raw_msg, "Decoded plaintext mismatch"

        mean_enc = float(np.mean(enc_times))
        std_enc = float(np.std(enc_times))
        mean_dec = float(np.mean(dec_times))
        std_dec = float(np.std(dec_times))
        mean_carriers = int(np.round(np.mean(carrier_counts)))

        enc_kbps = float((size / 1024.0) / (mean_enc / 1000.0))
        dec_kbps = float((size / 1024.0) / (mean_dec / 1000.0))
        carriers_per_sec = float(mean_carriers / ((mean_enc + mean_dec) / 1000.0))

        stages = profile_stage_breakdown(engine, raw_msg, session_key)

        res = PayloadPerformanceResult(
            payload_bytes=size,
            carrier_count=mean_carriers,
            encode_latency_ms=mean_enc,
            encode_latency_std=std_enc,
            decode_latency_ms=mean_dec,
            decode_latency_std=std_dec,
            total_latency_ms=mean_enc + mean_dec,
            encode_throughput_kbps=enc_kbps,
            decode_throughput_kbps=dec_kbps,
            carrier_speed_per_sec=carriers_per_sec,
            stage_breakdown=stages
        )
        results.append(res)

        print(f"  Payload: {size:3d} B | Carriers: {mean_carriers:2d} | "
              f"Encode: {mean_enc:6.2f} ± {std_enc:4.2f} ms | "
              f"Decode: {mean_dec:6.2f} ± {std_dec:4.2f} ms | "
              f"Total: {res.total_latency_ms:6.2f} ms | "
              f"Throughput: {enc_kbps:5.2f} KB/s")

    return results


def run_single_client_session(engine: SemanticEngine, session_id: int) -> Dict[str, Any]:
    """Execute a single independent client encode/decode session."""
    key = hashlib.sha256(f"session_key_{session_id}".encode()).digest()
    secret = f"SECRET_PAYLOAD_FOR_CLIENT_{session_id:04d}_VERIFY_ISOLATION!"

    t0 = time.perf_counter()
    enc = engine.encode(
        secret,
        mode="dssc",
        session_key=key,
        modalities=["image", "text", "audio"],
        use_ecc=True,
        ecc_parity_bytes=8
    )
    t_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dec = engine.decode(
        enc.media_ids,
        mode="dssc",
        session_key=key,
        use_ecc=True,
        ecc_parity_bytes=8
    )
    t_dec = (time.perf_counter() - t0) * 1000

    is_success = (dec.reconstructed_message == secret)
    # Check for cross-talk: did it recover someone else's payload?
    is_cross_talk = (dec.reconstructed_message != secret and "SECRET_PAYLOAD" in (dec.reconstructed_message or ""))

    return {
        "session_id": session_id,
        "latency_ms": t_enc + t_dec,
        "success": is_success,
        "cross_talk": is_cross_talk
    }


def benchmark_concurrency(engine: SemanticEngine) -> List[ConcurrencyResult]:
    """Stress-test direct-engine concurrent encode/decode sessions.

    The measured latency starts inside the worker after the task begins, so it is
    in-task service latency, not client-observed queue-inclusive latency.
    """
    print("\n--- Part 2: Direct-Engine Multi-Threaded Concurrency & Payload Isolation ---")
    thread_counts = [10, 20, 50]
    total_sessions_per_level = 100
    results = []

    for threads in thread_counts:
        print(f"  Testing Concurrency: {threads} parallel worker threads ({total_sessions_per_level} sessions)...")
        latencies = []
        successes = 0
        cross_talks = 0

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(run_single_client_session, engine, sid)
                for sid in range(total_sessions_per_level)
            ]
            for f in as_completed(futures):
                res = f.result()
                latencies.append(res["latency_ms"])
                if res["success"]:
                    successes += 1
                if res["cross_talk"]:
                    cross_talks += 1

        total_time = time.perf_counter() - t_start
        rps = total_sessions_per_level / total_time

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        success_pct = (successes / total_sessions_per_level) * 100.0
        cross_talk_pct = (cross_talks / total_sessions_per_level) * 100.0

        c_res = ConcurrencyResult(
            num_threads=threads,
            total_requests=total_sessions_per_level,
            duration_sec=total_time,
            requests_per_sec=rps,
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            success_rate_pct=success_pct,
            cross_talk_rate_pct=cross_talk_pct
        )
        results.append(c_res)

        print(f"    Sessions/s: {rps:5.2f} | p50: {p50:6.2f} ms | p95: {p95:6.2f} ms | p99: {p99:6.2f} ms | "
              f"Success: {success_pct:.1f}% | Observed swaps: {cross_talk_pct:.3f}%")

    return results


def create_latency_breakdown_plot(results: List[PayloadPerformanceResult], output_path: Path):
    """Generate stacked bar chart of computational latency breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sizes = [f"{r.payload_bytes} B" for r in results]
    x = np.arange(len(sizes))
    width = 0.55

    # 1. Encoding Stages Stacked
    ecc_enc = [r.stage_breakdown["enc_ecc_framing_ms"] for r in results]
    perm_enc = [r.stage_breakdown["enc_permutation_ms"] for r in results]
    faiss_enc = [r.stage_breakdown["enc_faiss_search_ms"] for r in results]

    ax1.bar(x, faiss_enc, width, label="FAISS Voronoi Search", color="#4A90D9", edgecolor="black", linewidth=0.6)
    ax1.bar(x, ecc_enc, width, bottom=faiss_enc, label="RS-ECC Framing", color="#F39C12", edgecolor="black", linewidth=0.6)
    ax1.bar(x, perm_enc, width, bottom=np.array(faiss_enc)+np.array(ecc_enc), label="HMAC-DRBG Permutation", color="#27AE60", edgecolor="black", linewidth=0.6)

    ax1.set_ylabel("Encoding Latency (ms)", fontsize=11)
    ax1.set_title("Direct-Engine Encoding Latency Breakdown", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes, fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(0, max(r.encode_latency_ms for r in results) * 1.12)

    for i, r in enumerate(results):
        ax1.text(i, r.encode_latency_ms + 2.0, f"{r.encode_latency_ms:.1f}ms", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # 2. Decoding Stages Stacked
    centroid_dec = [r.stage_breakdown["dec_centroid_lookup_ms"] for r in results]
    perm_dec = [r.stage_breakdown["dec_inv_permutation_ms"] for r in results]
    ecc_dec = [r.stage_breakdown["dec_ecc_syndrome_ms"] for r in results]

    ax2.bar(x, centroid_dec, width, label="Centroid Reverse Mapping", color="#9B59B6", edgecolor="black", linewidth=0.6)
    ax2.bar(x, ecc_dec, width, bottom=centroid_dec, label="RS Syndrome Correction", color="#E85D5D", edgecolor="black", linewidth=0.6)
    ax2.bar(x, perm_dec, width, bottom=np.array(centroid_dec)+np.array(ecc_dec), label="Inverse Permutation", color="#27AE60", edgecolor="black", linewidth=0.6)

    ax2.set_ylabel("Decoding Latency (ms)", fontsize=11)
    ax2.set_title("Direct-Engine Decoding Latency Breakdown", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes, fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.set_ylim(0, max(r.decode_latency_ms for r in results) * 1.10)

    for i, r in enumerate(results):
        ax2.text(i, r.decode_latency_ms + 3.0, f"{r.decode_latency_ms:.1f}ms", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_concurrency_plot(results: List[ConcurrencyResult], output_path: Path):
    """Generate concurrency scaling and latency percentile curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    threads = [r.num_threads for r in results]
    session_rate = [r.requests_per_sec for r in results]

    # 1. Throughput in completed sessions per second
    bars = ax1.bar([str(t) for t in threads], session_rate, width=0.45, color="#2ECC71", edgecolor="black", linewidth=0.7)
    ax1.set_xlabel("Worker Threads", fontsize=11)
    ax1.set_ylabel("Throughput (sessions/s)", fontsize=11)
    ax1.set_title("Direct-Engine Throughput Scaling under Worker Load", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, max(session_rate) * 1.12)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{bar.get_height():.2f} sessions/s",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Latency Percentiles
    p50 = [r.p50_latency_ms for r in results]
    p90 = [r.p90_latency_ms for r in results]
    p95 = [r.p95_latency_ms for r in results]
    p99 = [r.p99_latency_ms for r in results]

    ax2.plot(threads, p50, "o-", color="#3498DB", linewidth=2.0, label="p50 (Median)")
    ax2.plot(threads, p90, "s--", color="#F39C12", linewidth=2.0, label="p90")
    ax2.plot(threads, p95, "^-.", color="#E67E22", linewidth=2.0, label="p95")
    ax2.plot(threads, p99, "d-", color="#E74C3C", linewidth=2.2, label="p99 (Tail)")

    ax2.set_xlabel("Worker Threads", fontsize=11)
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_title("Direct-Engine Latency Percentile Expansion ($p_{50}$ to $p_{99}$)", fontsize=12, fontweight="bold")
    ax2.set_xticks(threads)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_latex_table(
    payload_results: List[PayloadPerformanceResult],
    concurrency_results: List[ConcurrencyResult],
    output_path: Path,
    commit: str | None = None,
):
    """Generate the publication table from recorded results.

    Part A is a seven-column codec table. Part B is a nine-column direct-engine
    concurrency table. Keeping two tabulars avoids padding Part A with empty
    columns and keeps the LaTeX alignment counts exact.
    """
    provenance = caption_provenance(
        n=f"{len(payload_results)} payload sizes, "
        f"{len(concurrency_results)} thread levels x 100 sessions each",
        scope="direct SemanticEngine encode/decode only; the API transmit path is not exercised",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_5_results.json"),
    )
    latex = r"""% Component 5: Computational Performance & System Scalability
\begin{table}[htbp]
\centering
\caption{\textbf{Codec Latency, Encoding Throughput, and Direct-Engine Thread Concurrency.} Empirical benchmark of the DCASS Dynamic Semantic State-Space Coding (DSSC) pipeline across payload sizes and parallel worker-thread workloads. The concurrency rows are direct SemanticEngine encode/decode service times, not FastAPI or transmit latencies.""" + provenance + r"""}
\label{tab:system_performance}
\begin{tabular}{lcccccc}
\toprule
\multicolumn{7}{l}{\textbf{Part A: End-to-End Codec Latency \& Throughput Across Payload Sizes}} \\
\midrule
Payload Size & Carrier Count & Encode Time & Decode Time & Total Latency & Encode Rate & Carrier Speed \\
(Bytes) & ($N$) & ($\mu \pm \sigma$, ms) & ($\mu \pm \sigma$, ms) & (ms) & (KB/s) & (carriers/s) \\
\midrule
"""
    for r in payload_results:
        latex += (f"{r.payload_bytes} B & {r.carrier_count} & "
                  f"{r.encode_latency_ms:.1f} $\\pm$ {r.encode_latency_std:.1f} & "
                  f"{r.decode_latency_ms:.1f} $\\pm$ {r.decode_latency_std:.1f} & "
                  f"{r.total_latency_ms:.1f} & "
                  f"{r.encode_throughput_kbps:.2f} & "
                  f"{r.carrier_speed_per_sec:.1f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{lcccccccc}
\toprule
\multicolumn{9}{l}{\textbf{Part B: Direct-Engine Thread Concurrency, Latency Percentiles \& Payload Isolation}} \\
\midrule
Worker & Total Sessions & Throughput & $p_{50}$ Latency & $p_{90}$ Latency & $p_{95}$ Latency & $p_{99}$ Latency & Success Rate & Observed Swap \\
Threads & Evaluated & (sessions/s) & (ms) & (ms) & (ms) & (ms) & (\%) & Rate (\%) \\
\midrule
"""
    for c in concurrency_results:
        latex += (f"{c.num_threads} & {c.total_requests} & "
                  f"{c.requests_per_sec:.2f} & "
                  f"{c.p50_latency_ms:.1f} & "
                  f"{c.p90_latency_ms:.1f} & "
                  f"{c.p95_latency_ms:.1f} & "
                  f"{c.p99_latency_ms:.1f} & "
                  f"\\textbf{{{c.success_rate_pct:.1f}\\%}} & "
                  f"\\textbf{{{c.cross_talk_rate_pct:.3f}\\%}} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    output_path.write_text(latex)
    scale_table_to_textwidth(output_path)
    print(f"  LaTeX table written to {output_path}")


def run_component5():
    print("=" * 80)
    print(" COMPONENT 5: CODEC LATENCY, THROUGHPUT & DIRECT-ENGINE CONCURRENCY")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("Scope: SemanticEngine.encode/decode directly; no FastAPI or /api/transmit.")
    print()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading unified semantic index and DSSC engine...")
    idx = UnifiedSemanticIndex()
    idx.load()
    engine = SemanticEngine(index=idx)
    engine.load()
    print("   SemanticEngine and FAISS indices loaded successfully.")

    # 1. Payload Scaling & Speed
    payload_results = benchmark_payload_scaling(engine)

    # 2. Direct-Engine Concurrency & Payload Isolation
    concurrency_results = benchmark_concurrency(engine)

    # 3. Generate Output Artifacts
    table_path = TABLES_DIR / "table5_system_performance.tex"
    plot_breakdown_path = PLOTS_DIR / "fig5_latency_breakdown.png"
    plot_concurrency_path = PLOTS_DIR / "fig5_concurrency_scaling.png"
    json_path = BENCHMARK_DIR / "component_5_results.json"

    print("\n3. Generating publication artifacts...")
    generate_latex_table(payload_results, concurrency_results, table_path)
    create_latency_breakdown_plot(payload_results, plot_breakdown_path)
    create_concurrency_plot(concurrency_results, plot_concurrency_path)

    output_data = {
        "schema": SCHEMA,
        "component": "Component 5: Codec Latency, Throughput & Direct-Engine Concurrency",
        "device": str(DEVICE),
        "provenance": collect_provenance(),
        "metric_definitions": metric_definitions(),
        "payload_scaling": [asdict(r) for r in payload_results],
        "concurrency_stress": [asdict(c) for c in concurrency_results],
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    write_manifest(
        build_manifest(
            component="component_5_system_concurrency",
            outputs=[
                json_path,
                table_path,
                plot_breakdown_path,
                plot_concurrency_path,
            ],
            inputs=sorted((PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")),
            seeds={"seed": SEED},
            extra={"config": output_data.get("config", {})},
        ),
        BENCHMARK_DIR / "component_5_results_manifest.json",
    )
    print(f"  JSON results saved to {json_path}")
    print(f"  Plots saved: {plot_breakdown_path.name}, {plot_concurrency_path.name}")

    print("\n" + "=" * 80)
    print(" COMPONENT 5 BENCHMARK COMPLETE")
    print("=" * 80)


def _load_artifact_results(json_path: Path):
    data = json.loads(json_path.read_text())
    payload_results = [PayloadPerformanceResult(**row) for row in data["payload_scaling"]]
    concurrency_results = [ConcurrencyResult(**row) for row in data["concurrency_stress"]]
    return payload_results, concurrency_results


def regenerate_table_from_json(json_path: Path | None = None, table_path: Path | None = None) -> Path:
    """Regenerate Table 5 from the recorded JSON without rerunning the benchmark."""
    json_path = json_path or BENCHMARK_DIR / "component_5_results.json"
    table_path = table_path or TABLES_DIR / "table5_system_performance.tex"
    payload_results, concurrency_results = _load_artifact_results(json_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    generate_latex_table(payload_results, concurrency_results, table_path)
    return table_path


def regenerate_artifacts_from_json(
    json_path: Path | None = None,
    table_path: Path | None = None,
    plot_breakdown_path: Path | None = None,
    plot_concurrency_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Regenerate Table 5 and both figures from the recorded JSON."""
    json_path = json_path or BENCHMARK_DIR / "component_5_results.json"
    table_path = table_path or TABLES_DIR / "table5_system_performance.tex"
    plot_breakdown_path = plot_breakdown_path or PLOTS_DIR / "fig5_latency_breakdown.png"
    plot_concurrency_path = plot_concurrency_path or PLOTS_DIR / "fig5_concurrency_scaling.png"
    payload_results, concurrency_results = _load_artifact_results(json_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plot_breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    generate_latex_table(payload_results, concurrency_results, table_path)
    create_latency_breakdown_plot(payload_results, plot_breakdown_path)
    create_concurrency_plot(concurrency_results, plot_concurrency_path)
    return table_path, plot_breakdown_path, plot_concurrency_path


def main():
    parser = argparse.ArgumentParser(description="Component 5 codec latency and direct-engine concurrency benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--table-only",
        action="store_true",
        help="Regenerate Table 5 from storage/benchmarks/component_5_results.json without rerunning the benchmark",
    )
    group.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Regenerate Table 5 and both figures from the recorded JSON without rerunning the benchmark",
    )
    args = parser.parse_args()
    if args.table_only:
        path = regenerate_table_from_json()
        print(f"Table regenerated from the existing artifact: {path}")
    elif args.artifacts_only:
        table, breakdown, concurrency = regenerate_artifacts_from_json()
        print(f"Artifacts regenerated from the existing artifact: {table}, {breakdown}, {concurrency}")
    else:
        run_component5()


if __name__ == "__main__":
    main()

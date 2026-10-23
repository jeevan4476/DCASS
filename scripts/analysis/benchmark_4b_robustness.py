#!/usr/bin/env python3
"""
Component 4b: Codec Robustness, Cryptographic Permutation & Channel Fault Tolerance

Tests:
1. HMAC-DRBG permutation scaling, determinism, and key sensitivity.
2. RS-ECC recovery under random carrier substitution.
3. Cryptographic key separation and unforgeability (random keys and single-bit
   mutated keys), counting any CRC-passing decode as a false acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from src.engine.dssc_state_space import derive_session_permutation
from src.engine.semantic_engine import SemanticEngine

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"
SCHEMA = "component4b-robustness-v2"
PARITY_BYTES = 8
MODALITIES = ["image", "text", "audio"]

np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class PermutationBenchmarkResult:
    candidate_size: int
    num_trials: int
    total_time_ms: float
    time_per_permutation_ms: float
    time_per_permutation_p50_ms: float
    time_per_permutation_std_ms: float
    determinism_passed: bool
    key_sensitivity_rho: float


@dataclass
class ECCChannelRecoveryResult:
    carrier_error_rate: float
    num_trials: int
    num_recovered: int
    recovery_rate_pct: float
    recovery_ci_low_pct: float
    recovery_ci_high_pct: float
    mean_ber_pct: float
    mean_fixed_errors: float
    mean_latency_ms: float
    parity_bytes: int
    num_decoder_successes: int
    num_false_acceptances: int
    num_corrupted_carriers: int
    num_symbol_changes: int
    symbol_change_pct: float


@dataclass
class KeySeparationResult:
    test_type: str
    num_trials: int
    num_false_acceptances: int
    num_exact_message_matches: int
    false_acceptance_rate: float
    false_acceptance_upper_95_pct: float
    rejection_rate_pct: float


@dataclass
class Component4bResult:
    schema: str
    provenance: Dict[str, object]
    config: Dict[str, object]
    metric_definitions: Dict[str, str]
    permutation_benchmarks: List[PermutationBenchmarkResult]
    ecc_recovery_benchmarks: List[ECCChannelRecoveryResult]
    key_separation_benchmarks: List[KeySeparationResult]
    plot_paths: Dict[str, str]


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
        "script": "scripts/analysis/benchmark_4b_robustness.py",
        "command": " ".join(sys.argv),
        "seed": SEED,
        "index": engine.index.status(),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval as percentages."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def run_permutation_benchmark(
    candidate_sizes: List[int] = [1000, 5000, 10000, 20000, 35000],
    num_trials: int = 100,
) -> List[PermutationBenchmarkResult]:
    """Benchmark HMAC-DRBG permutation throughput and key sensitivity."""
    print("\n--- Part 1: Permutation Acceleration & Scaling ---")
    results = []
    base_key = b"dcass_benchmark_session_key_32b!"
    salt = "dssc:benchmark:salt"

    for size in candidate_sizes:
        # Warm the generator once so lazy initialization is not charged to the
        # first recorded permutation.
        derive_session_permutation(size, base_key, context_salt=f"{salt}:warmup")

        trial_times_ms: List[float] = []
        for i in range(num_trials):
            start = time.perf_counter()
            derive_session_permutation(size, base_key, context_salt=f"{salt}:{i}")
            trial_times_ms.append((time.perf_counter() - start) * 1000.0)
        elapsed_ms = float(sum(trial_times_ms))
        time_per_perm_ms = elapsed_ms / num_trials
        time_per_perm_p50_ms = float(np.median(trial_times_ms))
        time_per_perm_std_ms = (
            float(np.std(trial_times_ms, ddof=1)) if num_trials > 1 else 0.0
        )

        p1 = derive_session_permutation(size, base_key, context_salt=salt)
        p2 = derive_session_permutation(size, base_key, context_salt=salt)
        determinism = bool(np.array_equal(p1, p2))

        flipped_key = bytearray(base_key)
        flipped_key[15] ^= 0x01
        p_flipped = derive_session_permutation(size, bytes(flipped_key), context_salt=salt)
        rho = float(np.corrcoef(p1.astype(float), p_flipped.astype(float))[0, 1])
        if np.isnan(rho):
            rho = 0.0

        results.append(
            PermutationBenchmarkResult(
                candidate_size=size,
                num_trials=num_trials,
                total_time_ms=elapsed_ms,
                time_per_permutation_ms=time_per_perm_ms,
                time_per_permutation_p50_ms=time_per_perm_p50_ms,
                time_per_permutation_std_ms=time_per_perm_std_ms,
                determinism_passed=determinism,
                key_sensitivity_rho=rho,
            )
        )

        print(
            f"  N = {size:5d} candidates | Mean: {time_per_perm_ms:.4f} ms/perm | "
            f"p50: {time_per_perm_p50_ms:.4f} ms | Determinism: {determinism} | "
            f"Key Sensitivity rho: {rho:.6f}"
        )

    return results


def run_ecc_recovery_benchmark(
    engine: SemanticEngine,
    error_rates: List[float] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    num_trials: int = 20,
    payload_size: int = 50,
) -> List[ECCChannelRecoveryResult]:
    """Stress-test Reed-Solomon decoding under random carrier substitution."""
    print("\n--- Part 2: Channel Degradation & RS-ECC Fault Tolerance ---")
    results = []

    all_corpus_ids: List[str] = []
    for modality in MODALITIES:
        if modality in engine.index.metadata:
            all_corpus_ids.extend(
                str(item["id"]) for item in engine.index.metadata[modality] if item.get("id")
            )
    mapper = engine._vcp_mapper

    rng = np.random.default_rng(SEED)

    for err_rate in error_rates:
        recovered_count = 0
        decoder_successes = 0
        false_acceptances = 0
        total_corrupted = 0
        symbol_changes = 0
        latencies: List[float] = []
        fixed_errors_list: List[float] = []
        ber_list: List[float] = []
        parity_values: List[int] = []

        for trial in range(num_trials):
            session_key = rng.bytes(32)
            test_payload = (
                f"DCASS channel stress payload #{trial:03d} - RS fault tolerance verification!"
            )[:payload_size]

            t0 = time.perf_counter()
            enc_res = engine.encode(
                test_payload,
                mode="dssc",
                session_key=session_key,
                modalities=MODALITIES,
                use_ecc=True,
                ecc_parity_bytes=PARITY_BYTES,
            )
            encode_time = (time.perf_counter() - t0) * 1000.0
            parity_values.append(int(enc_res.ecc_parity_bytes))
            if int(enc_res.ecc_parity_bytes) != PARITY_BYTES:
                raise RuntimeError(
                    f"unexpected parity {enc_res.ecc_parity_bytes}; expected {PARITY_BYTES}"
                )

            carrier_ids = list(enc_res.media_ids)
            n_carriers = len(carrier_ids)
            n_corrupt = int(round(n_carriers * err_rate))

            corrupted_carriers = carrier_ids.copy()
            if n_corrupt > 0:
                corrupt_indices = rng.choice(
                    n_carriers, size=min(n_corrupt, n_carriers), replace=False
                )
                for idx in corrupt_indices:
                    rand_id = str(rng.choice(all_corpus_ids))
                    while rand_id == carrier_ids[idx]:
                        rand_id = str(rng.choice(all_corpus_ids))
                    corrupted_carriers[idx] = rand_id
                    total_corrupted += 1
                    if mapper.symbol_for_media_id(carrier_ids[idx]) != mapper.symbol_for_media_id(rand_id):
                        symbol_changes += 1

            t1 = time.perf_counter()
            dec_res = engine.decode(
                corrupted_carriers,
                mode="dssc",
                session_key=session_key,
                modalities=MODALITIES,
                ecc_parity_bytes=enc_res.ecc_parity_bytes,
                use_ecc=True,
            )
            decode_time = (time.perf_counter() - t1) * 1000.0
            latencies.append(encode_time + decode_time)

            decoder_success = bool(dec_res.success)
            exact_recovery = (
                decoder_success and dec_res.reconstructed_message == test_payload
            )
            if decoder_success:
                decoder_successes += 1
                if not exact_recovery:
                    false_acceptances += 1

            if exact_recovery:
                recovered_count += 1
                ber_list.append(0.0)
            else:
                if dec_res.reconstructed_message:
                    original = test_payload.encode("utf-8")
                    reconstructed = dec_res.reconstructed_message.encode("utf-8")
                    min_len = min(len(original), len(reconstructed))
                    mismatches = sum(
                        bin(original[i] ^ reconstructed[i]).count("1")
                        for i in range(min_len)
                    )
                    mismatches += abs(len(original) - len(reconstructed)) * 8
                    ber = (mismatches / (len(original) * 8)) * 100.0
                else:
                    ber = 100.0
                ber_list.append(ber)

            fixed_errors_list.append(len(dec_res.ecc_fixed_errors))

        if len(set(parity_values)) != 1:
            raise RuntimeError(f"parity changed across trials: {sorted(set(parity_values))}")

        recovery_rate = (recovered_count / num_trials) * 100.0
        ci_low, ci_high = wilson_interval(recovered_count, num_trials)
        mean_ber = float(np.mean(ber_list))
        mean_fixed = float(np.mean(fixed_errors_list))
        mean_lat = float(np.mean(latencies))

        results.append(
            ECCChannelRecoveryResult(
                carrier_error_rate=err_rate,
                num_trials=num_trials,
                num_recovered=recovered_count,
                recovery_rate_pct=recovery_rate,
                recovery_ci_low_pct=ci_low,
                recovery_ci_high_pct=ci_high,
                mean_ber_pct=mean_ber,
                mean_fixed_errors=mean_fixed,
                mean_latency_ms=mean_lat,
                parity_bytes=parity_values[0],
                num_decoder_successes=decoder_successes,
                num_false_acceptances=false_acceptances,
                num_corrupted_carriers=total_corrupted,
                num_symbol_changes=symbol_changes,
                symbol_change_pct=(
                    100.0 * symbol_changes / total_corrupted if total_corrupted else 0.0
                ),
            )
        )

        print(
            f"  Substitution: {err_rate * 100:4.1f}% | Recovered: "
            f"{recovered_count:2d}/{num_trials} ({recovery_rate:5.1f}%) | "
            f"Accepted: {decoder_successes:2d} | Wrong accepts: {false_acceptances} | "
            f"Mean BER: {mean_ber:5.2f}% | Fixed errors: {mean_fixed:.1f} bytes | "
            f"Symbol changes: {symbol_changes}/{total_corrupted} | Latency: {mean_lat:.1f} ms"
        )

    return results


def run_key_separation_benchmark(
    engine: SemanticEngine,
    num_random_trials: int = 100,
    payload_size: int = 50,
) -> List[KeySeparationResult]:
    """Test key separation, counting any CRC-passing decode as acceptance."""
    print("\n--- Part 3: Cryptographic Key Separation & Zero FAR ---")
    results = []
    rng = np.random.default_rng(SEED)

    correct_key = rng.bytes(32)
    test_payload = "Cryptographic session key authorization test payload!"[:payload_size]

    enc_res = engine.encode(
        test_payload,
        mode="dssc",
        session_key=correct_key,
        modalities=MODALITIES,
        use_ecc=True,
        ecc_parity_bytes=PARITY_BYTES,
    )
    carrier_ids = enc_res.media_ids
    parity_bytes = int(enc_res.ecc_parity_bytes)
    if parity_bytes != PARITY_BYTES:
        raise RuntimeError(f"unexpected parity {parity_bytes}; expected {PARITY_BYTES}")

    print(f"  Testing {num_random_trials} independent random wrong keys...")
    false_accept_random = 0
    exact_random = 0
    for _ in range(num_random_trials):
        wrong_key = rng.bytes(32)
        dec_res = engine.decode(
            carrier_ids,
            mode="dssc",
            session_key=wrong_key,
            modalities=MODALITIES,
            ecc_parity_bytes=parity_bytes,
            use_ecc=True,
        )
        # Acceptance means the CRC gate returned a plaintext, regardless of
        # whether that plaintext equals the original message.
        if dec_res.success:
            false_accept_random += 1
            if dec_res.reconstructed_message == test_payload:
                exact_random += 1

    far_random = false_accept_random / num_random_trials
    upper_random = wilson_interval(false_accept_random, num_random_trials)[1]
    results.append(
        KeySeparationResult(
            test_type="random_wrong_keys",
            num_trials=num_random_trials,
            num_false_acceptances=false_accept_random,
            num_exact_message_matches=exact_random,
            false_acceptance_rate=far_random,
            false_acceptance_upper_95_pct=upper_random,
            rejection_rate_pct=(1.0 - far_random) * 100.0,
        )
    )
    print(
        f"    Random Keys: {false_accept_random}/{num_random_trials} accepted | "
        f"FAR = {far_random:.6f} | 95% upper = {upper_random / 100.0:.6f} | "
        f"Exact matches = {exact_random}"
    )

    num_bit_trials = 256
    print(f"  Testing all {num_bit_trials} single-bit flipped keys (exhaustive 256-bit check)...")
    false_accept_bitflip = 0
    exact_bitflip = 0
    for bit_idx in range(num_bit_trials):
        byte_pos = bit_idx // 8
        bit_pos = bit_idx % 8
        mutated_key = bytearray(correct_key)
        mutated_key[byte_pos] ^= 1 << bit_pos

        dec_res = engine.decode(
            carrier_ids,
            mode="dssc",
            session_key=bytes(mutated_key),
            modalities=MODALITIES,
            ecc_parity_bytes=parity_bytes,
            use_ecc=True,
        )
        if dec_res.success:
            false_accept_bitflip += 1
            if dec_res.reconstructed_message == test_payload:
                exact_bitflip += 1

    far_bitflip = false_accept_bitflip / num_bit_trials
    upper_bitflip = wilson_interval(false_accept_bitflip, num_bit_trials)[1]
    results.append(
        KeySeparationResult(
            test_type="single_bit_flipped_keys",
            num_trials=num_bit_trials,
            num_false_acceptances=false_accept_bitflip,
            num_exact_message_matches=exact_bitflip,
            false_acceptance_rate=far_bitflip,
            false_acceptance_upper_95_pct=upper_bitflip,
            rejection_rate_pct=(1.0 - far_bitflip) * 100.0,
        )
    )
    print(
        f"    1-Bit Mutated Keys: {false_accept_bitflip}/{num_bit_trials} accepted | "
        f"FAR = {far_bitflip:.6f} | 95% upper = {upper_bitflip / 100.0:.6f} | "
        f"Exact matches = {exact_bitflip}"
    )

    return results


def create_permutation_plots(
    results: List[PermutationBenchmarkResult], output_path: Path
) -> None:
    sizes = [result.candidate_size for result in results]
    latencies = [result.time_per_permutation_ms for result in results]
    rhos = [result.key_sensitivity_rho for result in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sizes, latencies, "o-", color="#2E86AB", linewidth=2.5, markersize=8)
    ax1.axhline(0.5, color="#E85D5D", linestyle="--", label="Sub-ms Ceiling (0.5 ms)")
    ax1.set_xlabel("Candidate Pool Size ($N$)", fontsize=12)
    ax1.set_ylabel("Execution Time per Permutation (ms)", fontsize=12)
    ax1.set_title("HMAC-DRBG Permutation Throughput", fontsize=13, fontweight="medium")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.plot(sizes, rhos, "s-", color="#A23B72", linewidth=2.5, markersize=8)
    ax2.axhline(0.0, color="gray", linestyle=":", label=r"Ideal Orthogonality ($\rho = 0$)")
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_xlabel("Candidate Pool Size ($N$)", fontsize=12)
    ax2.set_ylabel(r"Pearson Correlation ($\rho$)", fontsize=12)
    ax2.set_title("Avalanche Effect: 1-Bit Mutated Key Sensitivity", fontsize=13, fontweight="medium")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved Permutation Plot to {output_path}")


def create_ecc_recovery_plot(
    results: List[ECCChannelRecoveryResult], output_path: Path
) -> None:
    rates = [result.carrier_error_rate * 100 for result in results]
    recovery = [result.recovery_rate_pct for result in results]
    bers = [result.mean_ber_pct for result in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(rates, recovery, "o-", color="#27AE60", linewidth=2.5, markersize=8, label="Empirical Recovery")
    ax1.axvline(
        5.0, color="#E85D5D", linestyle="--",
        label=r"Algebraic Threshold ($t = 4$ byte errors, $\approx$5%)",
    )
    ax1.set_xlabel("Carrier Substitution Rate (%)", fontsize=12)
    ax1.set_ylabel("Message Recovery Rate (%)", fontsize=12)
    ax1.set_title("RS-ECC Recovery Under Carrier Substitution", fontsize=13, fontweight="medium")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.plot(rates, bers, "^-", color="#E67E22", linewidth=2.5, markersize=8, label="Residual BER")
    ax2.set_xlabel("Carrier Substitution Rate (%)", fontsize=12)
    ax2.set_ylabel("Bit Error Rate (%)", fontsize=12)
    ax2.set_title("Residual Bit Error Rate After RS Decoding", fontsize=13, fontweight="medium")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved ECC Recovery Plot to {output_path}")


def create_key_separation_plot(
    results: List[KeySeparationResult], output_path: Path
) -> None:
    categories = [
        f"Random Wrong Keys\n({results[0].num_trials} trials)",
        f"1-Bit Mutated Keys\n({results[1].num_trials} bit-flips)",
    ]
    rejections = [result.rejection_rate_pct for result in results]
    uppers = [result.false_acceptance_upper_95_pct for result in results]
    accepted = [result.num_false_acceptances for result in results]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(categories, rejections, color=["#2E86AB", "#A23B72"], width=0.45, alpha=0.85)

    for bar, accepted_count, upper in zip(bars, accepted, uppers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 8,
            f"{accepted_count} accepted\nFAR 95% upper: {upper:.2f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11,
        )

    ax.set_ylabel("Integrity Rejection Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Key Separation: Observed Rejection and FAR Bound", fontsize=13, fontweight="medium")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved Key Separation Plot to {output_path}")


def generate_latex_table(
    perm_results: List[PermutationBenchmarkResult],
    ecc_results: List[ECCChannelRecoveryResult],
    key_results: List[KeySeparationResult],
    output_path: Path,
    commit: str | None = None,
) -> None:
    provenance = caption_provenance(
        n=f"{len(perm_results)} permutation pool sizes, "
        f"{len(ecc_results)} substitution rates, "
        f"{sum(r.num_trials for r in key_results)} wrong-key trials",
        scope="clean-channel codec; substitution recovery and key separation; dropout is not covered here",
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_4b_robustness_results.json"),
    )
    with open(output_path, "w") as handle:
        handle.write("% Component 4b: Codec Robustness, Permutation Scaling & Key Separation\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{DSSC codec robustness: permutation scaling, RS-ECC resilience, "
            "and key separation. Intervals are Wilson 95 percent.%s}\n" % provenance
        )
        handle.write("\\label{tab:codec_robustness}\n")
        handle.write("\\begin{tabular}{lcccccc}\n")
        handle.write("\\toprule\n")
        handle.write("\\multicolumn{6}{c}{\\textbf{Sub-Millisecond HMAC-DRBG Permutation Scaling}} \\\\\n")
        handle.write(
            "Candidate Pool ($N$) & Mean (ms) & p50 (ms) & Determinism & "
            "Key Sensitivity ($\\rho$) & Throughput (perm/s) \\\\\n"
        )
        handle.write("\\midrule\n")
        for result in perm_results:
            throughput = (
                1000.0 / result.time_per_permutation_ms
                if result.time_per_permutation_ms > 0
                else 0.0
            )
            determinism = "\\checkmark" if result.determinism_passed else "\\times"
            handle.write(
                f"{result.candidate_size:,} & {result.time_per_permutation_ms:.4f} & "
                f"{result.time_per_permutation_p50_ms:.4f} & {determinism} & "
                f"{result.key_sensitivity_rho:+.6f} & {throughput:,.0f} \\\\\n"
            )
        handle.write("\\midrule\n")
        handle.write(
            "\\multicolumn{6}{c}{\\textbf{Algebraic RS-ECC Resilience Under Carrier Substitution}} \\\\\n"
        )
        handle.write(
            "Substitution Rate (\\%) & Recovery (\\%) & 95\\% CI & Mean BER (\\%) & "
            "Mean Fixed Errors & Latency (ms) \\\\\n"
        )
        handle.write("\\midrule\n")
        for result in ecc_results:
            handle.write(
                f"{result.carrier_error_rate * 100:.1f}\\% & {result.recovery_rate_pct:.1f}\\% & "
                f"[{result.recovery_ci_low_pct:.1f}, {result.recovery_ci_high_pct:.1f}] & "
                f"{result.mean_ber_pct:.2f}\\% & {result.mean_fixed_errors:.1f} B & "
                f"{result.mean_latency_ms:.1f} \\\\\n"
            )
        handle.write("\\midrule\n")
        handle.write(
            "\\multicolumn{6}{c}{\\textbf{Cryptographic Key Separation \\& Unforgeability}} \\\\\n"
        )
        handle.write(
            "Test Type & Trials & Accepted & Exact & FAR (\\%) & FAR 95\\% Upper (\\%) \\\\\n"
        )
        handle.write("\\midrule\n")
        for result in key_results:
            name = "Random Keys" if "random" in result.test_type else "Single-Bit Flipped Keys"
            handle.write(
                f"{name} & {result.num_trials} & {result.num_false_acceptances} & "
                f"{result.num_exact_message_matches} & {result.false_acceptance_rate * 100.0:.2f}\\% & "
                f"{result.false_acceptance_upper_95_pct:.2f}\\% \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
        handle.flush()
        scale_table_to_textwidth(output_path)


def load_results_from_json(json_path: Path) -> Component4bResult:
    with open(json_path) as handle:
        data = json.load(handle)
    return Component4bResult(
        schema=data["schema"],
        provenance=data["provenance"],
        config=data["config"],
        metric_definitions=data["metric_definitions"],
        permutation_benchmarks=[
            PermutationBenchmarkResult(**row) for row in data["permutation_benchmarks"]
        ],
        ecc_recovery_benchmarks=[
            ECCChannelRecoveryResult(**row) for row in data["ecc_recovery_benchmarks"]
        ],
        key_separation_benchmarks=[
            KeySeparationResult(**row) for row in data["key_separation_benchmarks"]
        ],
        plot_paths=data["plot_paths"],
    )


def regenerate_artifacts_from_json() -> None:
    json_path = BENCHMARK_DIR / "component_4b_robustness_results.json"
    result_obj = load_results_from_json(json_path)
    print(f"Loaded recorded results from {json_path}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    create_permutation_plots(
        result_obj.permutation_benchmarks, PLOTS_DIR / "fig_4b_permutation_scaling.png"
    )
    create_ecc_recovery_plot(
        result_obj.ecc_recovery_benchmarks, PLOTS_DIR / "fig_4b_ecc_recovery_curve.png"
    )
    create_key_separation_plot(
        result_obj.key_separation_benchmarks, PLOTS_DIR / "fig_4b_key_separation.png"
    )

    output_tex = TABLES_DIR / "table4b_codec_robustness.tex"
    generate_latex_table(
        result_obj.permutation_benchmarks,
        result_obj.ecc_recovery_benchmarks,
        result_obj.key_separation_benchmarks,
        output_tex,
    )
    print(f"Regenerated LaTeX table: {output_tex}")


def run_component4b() -> None:
    print("=" * 80)
    print(" COMPONENT 4b: CODEC ROBUSTNESS, PERMUTATION & CHANNEL FAULT TOLERANCE")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    perm_results = run_permutation_benchmark()

    print("\nLoading SemanticEngine and FAISS indices...")
    engine = SemanticEngine()
    engine.load()

    ecc_results = run_ecc_recovery_benchmark(engine)
    key_results = run_key_separation_benchmark(engine)

    print("\nGenerating benchmark plots...")
    p_perm = PLOTS_DIR / "fig_4b_permutation_scaling.png"
    p_ecc = PLOTS_DIR / "fig_4b_ecc_recovery_curve.png"
    p_key = PLOTS_DIR / "fig_4b_key_separation.png"
    create_permutation_plots(perm_results, p_perm)
    create_ecc_recovery_plot(ecc_results, p_ecc)
    create_key_separation_plot(key_results, p_key)

    output_json = BENCHMARK_DIR / "component_4b_robustness_results.json"
    payload = Component4bResult(
        schema=SCHEMA,
        provenance=collect_provenance(engine),
        config={
            "seed": SEED,
            "device": str(DEVICE),
            "candidate_sizes": [1000, 5000, 10000, 20000, 35000],
            "permutation_trials": 100,
            "permutation_warmup_trials": 1,
            "ecc_error_rates": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
            "ecc_trials_per_rate": 20,
            "payload_bytes": 50,
            "parity_bytes": PARITY_BYTES,
            "modalities": MODALITIES,
            "framing_policy": (
                "Framed payloads are required whenever ECC is enabled; legacy raw "
                "payloads remain decodable only when ECC is off."
            ),
            "key_separation_random_trials": 100,
            "key_separation_bit_flip_trials": 256,
        },
        metric_definitions={
            "key_separation_acceptance": (
                "A false acceptance is any decode for which the CRC gate returns success "
                "under an unauthorized key, whether or not the plaintext equals the original. "
                "Exact message matches are recorded separately."
            ),
            "false_acceptance_upper_95_pct": (
                "Wilson 95 percent upper bound on the per-attempt acceptance rate. A zero "
                "observed count does not by itself demonstrate the 2^-16 CRC bound."
            ),
            "carrier_substitution": (
                "The selected carrier is replaced with a different random corpus ID. Some "
                "replacements map to the same VCP symbol; symbol_change_pct records the share "
                "that actually changed the decoded symbol."
            ),
        },
        permutation_benchmarks=perm_results,
        ecc_recovery_benchmarks=ecc_results,
        key_separation_benchmarks=key_results,
        plot_paths={
            "permutation_scaling": _relative(p_perm),
            "ecc_recovery_curve": _relative(p_ecc),
            "key_separation": _relative(p_key),
        },
    )
    with open(output_json, "w") as handle:
        json.dump(asdict(payload), handle, indent=2)

    output_tex = TABLES_DIR / "table4b_codec_robustness.tex"
    generate_latex_table(
        perm_results,
        ecc_results,
        key_results,
        output_tex,
    )
    # The manifest is written last so the table it lists already exists and
    # its hash is of the final file.
    write_manifest(
        build_manifest(
            component="component_4b_robustness",
            outputs=[
                output_json,
                output_tex,
                PLOTS_DIR / "fig_4b_permutation_scaling.png",
                PLOTS_DIR / "fig_4b_ecc_recovery_curve.png",
                PLOTS_DIR / "fig_4b_key_separation.png",
            ],
            inputs=sorted((PROJECT_ROOT / "storage" / "data" / "indices").glob("*.index")),
            seeds={"seed": SEED},
            extra={"config": asdict(payload).get("config", {})},
        ),
        BENCHMARK_DIR / "component_4b_robustness_results_manifest.json",
    )

    print("\n" + "=" * 80)
    print(" COMPONENT 4b BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"  JSON Results: {output_json}")
    print(f"  LaTeX Table:  {output_tex}")
    print(f"  Plots:        {p_perm}")
    print(f"                {p_ecc}")
    print(f"                {p_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-from-json",
        action="store_true",
        help="Rebuild the table and plots from the recorded JSON without rerunning measurements",
    )
    args = parser.parse_args()
    if args.regenerate_from_json:
        regenerate_artifacts_from_json()
    else:
        run_component4b()


if __name__ == "__main__":
    main()

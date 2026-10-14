#!/usr/bin/env python3
"""
Component 4a: Codec Capacity Benchmark - DSSC vs exact_vcp
Tests: Carrier count, bits per carrier, latency comparison
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
import hashlib

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.dssc_encoder import DSSCEncoder
from src.engine.dssc_decoder import DSSCDecoder
from src.engine.vcp_payload import VCPPayloadMapper
from src.corpus.index.unified_index import UnifiedSemanticIndex

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class CapacityResult:
    mode: str
    payload_bytes: int
    carrier_count: int
    bits_per_carrier: float
    carrier_reduction_pct: float
    encode_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float


def run_capacity_benchmark():
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
    
    payload_sizes = [14, 50, 150, 300, 500]
    num_trials = 5
    session_key = b"test_session_key_32bytes_long!!"
    
    results = []
    
    for payload_size in payload_sizes:
        print(f"\n2. Payload size: {payload_size} bytes")
        test_payload = "X" * payload_size
        
        # --- exact_vcp Mode (text-only, production framed + 8B ECC) ---
        vcp_carrier_counts = []
        vcp_encode_times = []
        vcp_decode_times = []
        
        for trial in range(num_trials):
            start = time.perf_counter()
            vcp_result = engine.encode(
                test_payload,
                mode="exact_vcp",
                modalities=["text"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            encode_time = (time.perf_counter() - start) * 1000
            
            start = time.perf_counter()
            dec_vcp = engine.decode(
                vcp_result.media_ids,
                mode="exact_vcp",
                modalities=["text"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            decode_time = (time.perf_counter() - start) * 1000
            assert dec_vcp.reconstructed_message == test_payload, f"exact_vcp decode mismatch on trial {trial}"
            
            vcp_carrier_counts.append(len(vcp_result.media_ids))
            vcp_encode_times.append(encode_time)
            vcp_decode_times.append(decode_time)
        
        vcp_avg_carriers = float(np.mean(vcp_carrier_counts))
        vcp_bits_per_carrier = 8.00  # 1 byte per carrier channel rate
        
        # --- DSSC Mode (full multi-modal, production framed + 8B ECC) ---
        dssc_carrier_counts = []
        dssc_encode_times = []
        dssc_decode_times = []
        dssc_bpc_list = []
        
        for trial in range(num_trials):
            session_key_trial = hashlib.sha256(session_key + trial.to_bytes(4, 'big')).digest()
            
            start = time.perf_counter()
            dssc_result = engine.encode(
                test_payload,
                mode="dssc",
                session_key=session_key_trial,
                modalities=["image", "text", "audio"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            encode_time = (time.perf_counter() - start) * 1000
            
            start = time.perf_counter()
            dec_dssc = engine.decode(
                dssc_result.media_ids,
                mode="dssc",
                session_key=session_key_trial,
                use_ecc=True,
                ecc_parity_bytes=8
            )
            decode_time = (time.perf_counter() - start) * 1000
            assert dec_dssc.reconstructed_message == test_payload, f"DSSC multi-modal decode mismatch on trial {trial}"
            
            dssc_carrier_counts.append(len(dssc_result.media_ids))
            dssc_encode_times.append(encode_time)
            dssc_decode_times.append(decode_time)
            dssc_bpc_list.append(dssc_result.bits_per_carrier)
        
        dssc_avg_carriers = float(np.mean(dssc_carrier_counts))
        dssc_bits_per_carrier = float(np.mean(dssc_bpc_list))
        carrier_reduction_multi = ((vcp_avg_carriers - dssc_avg_carriers) / vcp_avg_carriers) * 100.0
        
        # --- DSSC Mode (text-only for fair comparison, production framed + 8B ECC) ---
        dssc_carrier_counts_text = []
        dssc_encode_times_text = []
        dssc_decode_times_text = []
        dssc_bpc_text_list = []
        
        for trial in range(num_trials):
            session_key_trial = hashlib.sha256(session_key + trial.to_bytes(4, 'big')).digest()
            
            start = time.perf_counter()
            dssc_result = engine.encode(
                test_payload,
                mode="dssc",
                session_key=session_key_trial,
                modalities=["text"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            encode_time = (time.perf_counter() - start) * 1000
            
            start = time.perf_counter()
            dec_dssc = engine.decode(
                dssc_result.media_ids,
                mode="dssc",
                session_key=session_key_trial,
                modalities=["text"],
                use_ecc=True,
                ecc_parity_bytes=8
            )
            decode_time = (time.perf_counter() - start) * 1000
            assert dec_dssc.reconstructed_message == test_payload, f"DSSC text-only decode mismatch on trial {trial}"
            
            dssc_carrier_counts_text.append(len(dssc_result.media_ids))
            dssc_encode_times_text.append(encode_time)
            dssc_decode_times_text.append(decode_time)
            dssc_bpc_text_list.append(dssc_result.bits_per_carrier)
        
        dssc_avg_carriers_text = float(np.mean(dssc_carrier_counts_text))
        dssc_bits_per_carrier_text = float(np.mean(dssc_bpc_text_list))
        carrier_reduction_text = ((vcp_avg_carriers - dssc_avg_carriers_text) / vcp_avg_carriers) * 100.0
        
        print(f"    exact_vcp (text):   {vcp_avg_carriers:.0f} carriers, {vcp_bits_per_carrier:.2f} bits/carrier, {np.mean(vcp_encode_times):.1f}ms encode, {np.mean(vcp_decode_times):.1f}ms decode")
        print(f"    DSSC (multi-modal): {dssc_avg_carriers:.0f} carriers, {dssc_bits_per_carrier:.2f} bits/carrier, {np.mean(dssc_encode_times):.1f}ms encode, {np.mean(dssc_decode_times):.1f}ms decode (Reduction: {carrier_reduction_multi:.1f}%)")
        print(f"    DSSC (text-only):   {dssc_avg_carriers_text:.0f} carriers, {dssc_bits_per_carrier_text:.2f} bits/carrier, {np.mean(dssc_encode_times_text):.1f}ms encode, {np.mean(dssc_decode_times_text):.1f}ms decode (Reduction: {carrier_reduction_text:.1f}%)")
        
        results.append(CapacityResult(
            mode="DSSC (multi-modal)",
            payload_bytes=payload_size,
            carrier_count=int(dssc_avg_carriers),
            bits_per_carrier=float(dssc_bits_per_carrier),
            carrier_reduction_pct=float(carrier_reduction_multi),
            encode_latency_ms=float(np.mean(dssc_encode_times)),
            decode_latency_ms=float(np.mean(dssc_decode_times)),
            total_latency_ms=float(np.mean(dssc_encode_times) + np.mean(dssc_decode_times))
        ))
        
        results.append(CapacityResult(
            mode="DSSC (text-only)",
            payload_bytes=payload_size,
            carrier_count=int(dssc_avg_carriers_text),
            bits_per_carrier=float(dssc_bits_per_carrier_text),
            carrier_reduction_pct=float(carrier_reduction_text),
            encode_latency_ms=float(np.mean(dssc_encode_times_text)),
            decode_latency_ms=float(np.mean(dssc_decode_times_text)),
            total_latency_ms=float(np.mean(dssc_encode_times_text) + np.mean(dssc_decode_times_text))
        ))
        
        results.append(CapacityResult(
            mode="exact_vcp (text)",
            payload_bytes=payload_size,
            carrier_count=int(vcp_avg_carriers),
            bits_per_carrier=float(vcp_bits_per_carrier),
            carrier_reduction_pct=0.0,
            encode_latency_ms=float(np.mean(vcp_encode_times)),
            decode_latency_ms=float(np.mean(vcp_decode_times)),
            total_latency_ms=float(np.mean(vcp_encode_times) + np.mean(vcp_decode_times))
        ))
    
    # Save results
    output_path = BENCHMARK_DIR / "component_4a_capacity_results.json"
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    # Generate LaTeX table
    tex_path = TABLES_DIR / "table4a_capacity.tex"
    with open(tex_path, 'w') as f:
        f.write("% Component 4a: Codec Capacity - DSSC vs exact_vcp\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Codec Capacity: DSSC vs exact_vcp (matched framing and Reed-Solomon ECC)}\n")
        f.write("\\label{tab:codec_capacity}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Mode & Payload (B) & Carriers & Bits/Carrier & Reduction (\\%) & Encode (ms) & Decode (ms) \\\\\n")
        f.write("\\midrule\n")
        for r in results:
            red_str = f"{r.carrier_reduction_pct:.1f}\\%" if r.carrier_reduction_pct > 0 else "--"
            f.write(f"{r.mode} & {r.payload_bytes} & {r.carrier_count} & ")
            f.write(f"{r.bits_per_carrier:.2f} & {red_str} & {r.encode_latency_ms:.1f} & {r.decode_latency_ms:.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Create plots
    create_capacity_plots(results, PLOTS_DIR / "fig_4a_capacity_comparison.png")
    
    print(f"\nResults saved to {output_path}")
    print(f"LaTeX table saved to {tex_path}")
    print(f"Plots saved to {PLOTS_DIR}")
    
    return results


def create_capacity_plots(results: List[CapacityResult], output_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    sizes = sorted(set(r.payload_bytes for r in results))
    modes = ["DSSC (multi-modal)", "DSSC (text-only)", "exact_vcp (text)"]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'DSSC (multi-modal)': '#E85D5D', 'DSSC (text-only)': '#F39C12', 'exact_vcp (text)': '#27AE60'}
    
    for mode in modes:
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        carrier_counts = [r.carrier_count for r in mode_results]
        bpc = [r.bits_per_carrier for r in mode_results]
        lat = [r.total_latency_ms for r in mode_results]
        ax1.plot(sizes, carrier_counts, 'o-', label=mode, color=colors.get(mode, '#888'), linewidth=2, markersize=8)
        ax2.plot(sizes, bpc, 'o-', label=mode, color=colors.get(mode, '#888'), linewidth=2, markersize=8)
        ax3.plot(sizes, lat, 'o-', label=mode, color=colors.get(mode, '#888'), linewidth=2, markersize=8)
    
    ax1.set_xlabel('Payload Size (bytes)')
    ax1.set_ylabel('Carrier Count')
    ax1.set_title('Carriers Required vs Payload Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Payload Size (bytes)')
    ax2.set_ylabel('Bits per Carrier')
    ax2.set_title('Codec Capacity (Bits per Carrier)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=8.0, color='k', linestyle=':', alpha=0.5, label='exact_vcp baseline')
    
    ax3.set_xlabel('Payload Size (bytes)')
    ax3.set_ylabel('Total Latency (ms)')
    ax3.set_title('End-to-End Encode + Decode Latency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Capacity plot saved to {output_path}")


if __name__ == "__main__":
    import json
    run_capacity_benchmark()
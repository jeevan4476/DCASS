#!/usr/bin/env python3
"""
Compare the two candidate human references for the C3 KS term.

The benchmark currently measures KS against `generate_human_baseline_delays`,
a synthetic surrogate built from two mixture components. The trained
detector, by contrast, was trained and calibrated against the corpus
traffic in `storage/data/traffic/`. If the two disagree about what human
traffic looks like, the KS term has a large null and the detection result
depends on which reference is used.

This measures, for each scheduler and for real human traffic:
  * the KS statistic against each reference;
  * the trained detector's anomaly score;
  * how a 50-item burst's KS behaves when drawn from the reference itself,
    which is the null the calibration has to beat.

The output decides which reference the KS term should use.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from src.utils.checkpoint_io import load_trained_checkpoint
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402
from train_traffic_detector import load_human_sessions  # noqa: E402


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "b3", PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["b3"] = module
    spec.loader.exec_module(module)
    return module


def score(model, delays, channels):
    d_t = torch.tensor(np.asarray(delays, dtype=np.float32)).unsqueeze(0).to("cpu")
    c_t = torch.tensor(np.asarray(channels, dtype=np.int64)).unsqueeze(0).to("cpu")
    with torch.no_grad():
        verdict = model(d_t, c_t)
    return float(verdict.anomaly_scores.mean())


def main() -> int:
    bench = load_benchmark()
    ckpt = load_trained_checkpoint(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt",
        map_location="cpu",)
    model = DeepPacketInspectionWarden(num_channels=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    surrogate = bench.generate_human_baseline_delays(5000)
    human = load_human_sessions()
    corpus_pool = np.concatenate(
        [np.asarray(x["delays"], dtype=np.float64) for x in human]
    )

    print("Reference distributions:")
    for name, ref in (("surrogate", surrogate), ("corpus", corpus_pool)):
        print(
            f"  {name:10s} n={len(ref):7d}  median {np.median(ref):8.2f}  "
            f"p95 {np.percentile(ref,95):9.2f}  max {ref.max():9.2f}"
        )
    print()

    # The null: a 50-item burst drawn from the reference itself.
    print("Null behaviour (50-item burst drawn from the reference itself):")
    rng = np.random.default_rng(0)
    for name, ref in (("surrogate", surrogate), ("corpus", corpus_pool)):
        ks_null = [
            ks_2samp(
                ref[rng.choice(len(ref), 50, replace=False)], ref
            ).statistic
            for _ in range(300)
        ]
        print(
            f"  {name:10s} KS mean {np.mean(ks_null):.4f}  p95 {np.percentile(ks_null,95):.4f}  "
            f"max {np.max(ks_null):.4f}"
        )
    print()

    # Real human sessions from the corpus, against each reference.
    print("Real corpus human sessions (the population the detector was trained on):")
    ks_s, ks_c, an = [], [], []
    for session in human[-300:]:
        d = np.asarray(session["delays"], dtype=np.float64)
        c = np.asarray(session["channels"], dtype=np.int64)
        ks_s.append(ks_2samp(d, surrogate).statistic)
        ks_c.append(ks_2samp(d, corpus_pool).statistic)
        an.append(score(model, d, c))
    print(
        f"  KS vs surrogate mean {np.mean(ks_s):.4f} p95 {np.percentile(ks_s,95):.4f}"
    )
    print(
        f"  KS vs corpus    mean {np.mean(ks_c):.4f} p95 {np.percentile(ks_c,95):.4f}"
    )
    print(f"  anomaly         mean {np.mean(an):.4f} p95 {np.percentile(an,95):.4f}")
    print()

    # The schedulers, recomputed from the recorded run's parameters is not
    # possible, so the recorded KS values are used for the comparison.
    results = json.loads(
        (PROJECT_ROOT / "storage" / "benchmarks" / "component_3_results.json").read_text()
    )
    print("Recorded scheduler KS values (against the surrogate):")
    for s in results["schedulers"]:
        print(f"  {s['scheduler']:24s} {s['ks_statistic']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

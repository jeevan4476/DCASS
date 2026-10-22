#!/usr/bin/env python3
"""
Measure the distributional term under a null referenced the same way the
test statistic is computed.

The C3 KS term compares a 50-item burst against a 5,000-item pool. The
null of that comparison is not zero: a 50-item burst drawn from the pool
itself already scores about 0.12. The recorded `human_reference` in the
benchmark is a synthetic burst-level surrogate, 80 percent exponential with mean 5
seconds and 20 percent log-normal, whose delay distribution does not match the corpus

This script measures, for each candidate reference:
  * the null: 50-item bursts drawn from the reference itself;
  * the session null: a whole human *session* scored against the pooled
    reference, which is the level a real human session reaches;
  * the recorded scheduler KS values against that reference.

The distributional term should be the excess over the session null, so
that "looks like a real human session" is the zero point.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
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


def main() -> int:
    bench = load_benchmark()
    surrogate = bench.generate_human_baseline_delays(5000)
    human = load_human_sessions()
    corpus_pool = np.concatenate(
        [np.asarray(x["delays"], dtype=np.float64) for x in human]
    )

    rng = np.random.default_rng(0)
    print("Null behaviour by reference:")
    for name, ref, sessions in (
        ("surrogate", surrogate, None),
        ("corpus", corpus_pool, human),
    ):
        burst_null = [
            ks_2samp(
                ref[rng.choice(len(ref), 50, replace=False)], ref
            ).statistic
            for _ in range(300)
        ]
        print(
            f"  {name:10s} burst-from-pool null: mean {np.mean(burst_null):.4f} "
            f"p95 {np.percentile(burst_null,95):.4f} max {np.max(burst_null):.4f}"
        )
        if sessions is not None:
            sess = [
                ks_2samp(
                    np.asarray(s["delays"], dtype=np.float64), ref
                ).statistic
                for s in sessions[-300:]
            ]
            print(
                f"             real human SESSION null: mean {np.mean(sess):.4f} "
                f"p95 {np.percentile(sess,95):.4f} max {np.max(sess):.4f}"
            )
    print()

    # The DCASS schedulers, recomputed against the corpus reference.
    print("DCASS schedulers scored against the corpus reference:")
    ckpt = torch.load(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector_eval.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = DeepPacketInspectionWarden(num_channels=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scheduler = bench.StealthScheduler(
        num_channels=bench.NUM_CHANNELS, device="cpu", profile="casual", seed=bench.SEED
    )
    scheduler._load_generator(PROJECT_ROOT / "storage" / "models" / "gan_generator.pt")
    media_ids = [f"carrier_{i:04d}" for i in range(bench.CARRIERS_PER_MESSAGE)]

    for label, mode in (("DCASS Temporal WGAN", "gan"), ("DCASS Hybrid Engine", "hybrid")):
        ks_list, an_list = [], []
        for _ in range(20):
            sched = scheduler.schedule(
                media_ids=media_ids, mode=mode, base_delay=bench.BASE_DELAY, time_of_day=14.0
            )
            d = np.asarray(sched["delays"], dtype=np.float64)
            c = np.asarray(sched["channels"], dtype=np.int64)
            ks_list.append(ks_2samp(d, corpus_pool).statistic)
            with torch.no_grad():
                v = model(
                    torch.tensor(d, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(c, dtype=torch.long).unsqueeze(0),
                )
            an_list.append(float(v.anomaly_scores.mean()))
        print(
            f"  {label:24s} KS vs corpus mean {np.mean(ks_list):.4f}  "
            f"anomaly mean {np.mean(an_list):.4f}"
        )

    # The same for the benchmark's surrogate reference, for comparison.
    print("\n  (same schedulers against the surrogate reference)")
    for label, mode in (("DCASS Temporal WGAN", "gan"), ("DCASS Hybrid Engine", "hybrid")):
        ks_list, an_list = [], []
        for _ in range(20):
            sched = scheduler.schedule(
                media_ids=media_ids, mode=mode, base_delay=bench.BASE_DELAY, time_of_day=14.0
            )
            d = np.asarray(sched["delays"], dtype=np.float64)
            c = np.asarray(sched["channels"], dtype=np.int64)
            ks_list.append(ks_2samp(d, surrogate).statistic)
            with torch.no_grad():
                v = model(
                    torch.tensor(d, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(c, dtype=torch.long).unsqueeze(0),
                )
            an_list.append(float(v.anomaly_scores.mean()))
        print(
            f"  {label:24s} KS vs surrogate mean {np.mean(ks_list):.4f}  "
            f"anomaly mean {np.mean(an_list):.4f}"
        )

    # Record the null band, so the documentation can cite an artifact
    # rather than a number that was written down once.
    results_path = (
        PROJECT_ROOT / "storage" / "benchmarks" / "traffic_ks_null_band.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "schema": "traffic-ks-null-band-v1",
                "detector": "storage/models/traffic_detector_eval.pt",
                "reference": "the burst-level synthetic human reference",
                "note": (
                    "The per-trial KS distances a 50-item burst drawn from the "
                    "reference itself produces. A sampler that matches the "
                    "reference lands inside this band; a burst that is far from "
                    "it does not."
                ),
                "burst_from_pool": {
                    "mean": float(np.mean(burst_null)),
                    "p95": float(np.percentile(burst_null, 95)),
                    "max": float(np.max(burst_null)),
                    "n": int(len(burst_null)),
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved KS null band to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

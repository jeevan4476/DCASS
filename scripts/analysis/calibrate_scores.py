#!/usr/bin/env python3
# scripts/analysis/calibrate_scores.py
"""
Re-measure per-modality similarity-score calibration (audit M-17).

Runs a set of probe queries against the live FAISS indices, records the raw
score distribution per modality, and writes:

    storage/data/indices/score_calibration.json

ScoreNormalizer picks this file up automatically at import time, so the
cross-modal z-score+sigmoid normalization is measured on the ACTUAL corpus
instead of relying on the hardcoded 2026-02-17 constants.

Usage:
    python scripts/analysis/calibrate_scores.py [--queries 30] [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

PROBE_QUERIES = [
    # image-friendly
    "a dog running on the beach",
    "sunset over mountains",
    "children playing soccer",
    "a red sports car",
    "snowy forest landscape",
    "person cooking in kitchen",
    "city street at night with neon lights",
    "bird flying over the ocean",
    "birthday cake with candles",
    "crowd at a concert",
    # text/knowledge friendly
    "the theory of general relativity",
    "history of the roman empire",
    "photosynthesis in plants",
    "how vaccines work",
    "climate change evidence",
    "quantum computing basics",
    "the french revolution",
    "machine learning algorithms",
    "renewable energy sources",
    "human digestive system",
    # audio friendly
    "gentle piano melody",
    "thunder and heavy rain",
    "birds chirping in the morning",
    "engine starting up",
    "ocean waves crashing",
    "acoustic guitar chords",
    "busy restaurant ambience",
    "footsteps on gravel",
    "wind blowing through trees",
    "church bells ringing",
]


def measure(queries: list[str], top_k: int) -> dict:
    from src.corpus.index.unified_index import UnifiedSemanticIndex

    index = UnifiedSemanticIndex()
    status = index.load()
    failed = [m for m, ok in status.items() if not ok]
    if failed:
        print(f"WARNING: indices failed to load: {failed}")

    scores: dict[str, list[float]] = {"image": [], "text": [], "audio": []}
    t0 = time.perf_counter()

    for i, query in enumerate(queries, 1):
        results = index.search(query=query, k=top_k * len(scores), min_score=-1.0)
        by_modality: dict[str, list[float]] = {m: [] for m in scores}
        for r in results:
            if r.modality in by_modality and len(by_modality[r.modality]) < top_k:
                by_modality[r.modality].append(float(r.score))
        for m, vals in by_modality.items():
            if vals:
                scores[m].extend(vals)
        print(
            f"  [{i}/{len(queries)}] {query!r} -> "
            + ", ".join(
                f"{m}: n={len(v)} mean={np.mean(v):.3f}" if v else f"{m}: none"
                for m, v in by_modality.items()
            )
        )

    elapsed = time.perf_counter() - t0
    modalities = {}
    for m, vals in scores.items():
        if len(vals) < 10:
            print(f"  WARNING: only {len(vals)} samples for '{m}' - statistics unreliable")
            continue
        modalities[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "probe_queries": len(queries),
        "top_k_per_modality": top_k,
        "elapsed_seconds": round(elapsed, 1),
        "modalities": modalities,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=int, default=None, help="Number of probe queries (default: all built-ins)"
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    queries = PROBE_QUERIES[: args.queries] if args.queries else PROBE_QUERIES
    print(f"Calibrating score normalizer with {len(queries)} queries (k={args.top_k})...")

    report = measure(queries, args.top_k)

    from src.corpus.index.unified_index import ScoreNormalizer

    out_path = ScoreNormalizer._calibration_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration written -> {out_path}\n")
    print(json.dumps(report["modalities"], indent=2))
    print("\nRestart API/CLI processes to pick up the new calibration.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check that the trained detector discriminates rather than memorizes.

The concern: the detector was trained on the corpus traffic, so it could be
scoring "not exactly my training corpus" rather than "bot-like". This
script measures its behaviour on traffic it never saw:

  * held-out corpus human sessions (same family as training, unseen rows);
  * the benchmark's synthetic surrogate reference;
  * human traffic from an independent generator, so a distribution the
    detector has never seen in any form;
  * the two DCASS generator schedules, the WGAN and the hybrid engine;
  * synthetic bot traffic.

The question the numbers answer: does the detector separate the DCASS
schedules from human traffic in general, or only from its own training
rows?
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402
from train_traffic_detector import load_human_sessions, make_bot_session  # noqa: E402

# The detector's input length, matching how it was trained.
DETECTOR_SEQ_LEN = 64


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "b3", PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["b3"] = module
    spec.loader.exec_module(module)
    return module


def independent_human_sessions(n: int, rng: np.random.Generator) -> list[dict]:
    """Human traffic from a generator the detector has never seen.

    A Weibull-mixture reading model with a session-level heavy tail: short
    gaps while actively posting, occasional long pauses, and a burst of
    rapid successive posts. Deliberately different in shape from both the
    corpus and the benchmark's surrogate.
    """
    sessions = []
    for _ in range(n):
        length = int(rng.integers(20, 60))
        scale = rng.uniform(2.0, 9.0)
        shape = rng.uniform(0.6, 1.8)
        delays = scale * rng.weibull(shape, length)
        # Occasionally the user steps away.
        pauses = rng.random(length) < 0.08
        delays[pauses] = rng.uniform(45.0, 900.0, pauses.sum())
        # And occasionally posts in a quick burst.
        bursts = rng.random(length) < 0.10
        delays[bursts] = rng.uniform(0.4, 1.2, bursts.sum())
        sessions.append(
            {
                "delays": np.clip(delays, 0.4, None).tolist(),
                "channels": rng.integers(0, 3, length).tolist(),
                "time_of_day": int(rng.integers(0, 24)),
            }
        )
    return sessions


def score(model, delays, channels) -> float:
    """The detector's bot probability, the quantity the soft branch uses."""
    """Score a session the way the benchmark does.

    The detector was trained on sequences padded to DETECTOR_SEQ_LEN with a
    validity mask, and its transformer excludes the padded steps from
    attention, so the padding and the mask are part of how the detector is
    meant to be fed.
    """
    n = min(len(delays), DETECTOR_SEQ_LEN)
    pad_d = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_c = np.zeros(DETECTOR_SEQ_LEN, dtype=np.int64)
    mask = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_d[:n] = np.asarray(delays[:n], dtype=np.float32)
    pad_c[:n] = np.asarray(channels[:n], dtype=np.int64)
    mask[:n] = 1.0
    with torch.no_grad():
        verdict = model(
            torch.tensor(pad_d).unsqueeze(0),
            torch.tensor(pad_c).unsqueeze(0),
            mask=torch.tensor(mask).unsqueeze(0),
        )
    return float(verdict.bot_probability.numpy()[0])


def _stats(values: np.ndarray) -> dict:
    """A distributional summary of the detector's score on one population."""
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def report(name, scores):
    a = np.asarray(scores)
    print(
        f"  {name:34s} n={len(a):4d}  mean {a.mean():7.4f}  "
        f"p05 {np.percentile(a,5):7.4f}  p50 {np.percentile(a,50):7.4f}  "
        f"p95 {np.percentile(a,95):7.4f}"
    )
    return a


def main() -> int:
    bench = load_benchmark()
    ckpt = torch.load(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = DeepPacketInspectionWarden(num_channels=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rng = np.random.default_rng(7)
    human = load_human_sessions()
    print()

    # 1. Held-out corpus human sessions.
    heldout = human[-300:]
    a_corpus = report(
        "corpus human (held-out rows)",
        [
            score(model, s["delays"], s["channels"])
            for s in heldout
        ],
    )

    # 2. The benchmark's synthetic surrogate.
    surrogate = bench.generate_human_baseline_delays(5000)
    a_sur = report(
        "benchmark synthetic surrogate",
        [
            score(
                model,
                surrogate[
                    rng.choice(len(surrogate), 50, replace=False)
                ],
                rng.integers(0, 3, 50),
            )
            for _ in range(300)
        ],
    )

    # 3. Independent human generator.
    indep = independent_human_sessions(300, rng)
    a_indep = report(
        "independent human generator",
        [score(model, s["delays"], s["channels"]) for s in indep],
    )

    # 4. Synthetic bots.
    bots = [make_bot_session(s, rng) for s in human[:300]]
    a_bot = report(
        "synthetic bot traffic",
        [score(model, s["delays"], s["channels"]) for s in bots],
    )

    # 5. DCASS schedulers.
    scheduler = bench.StealthScheduler(
        num_channels=bench.NUM_CHANNELS, device="cpu", profile="casual", seed=bench.SEED
    )
    scheduler._load_generator(PROJECT_ROOT / "storage" / "models" / "gan_generator.pt")
    media_ids = [f"carrier_{i:04d}" for i in range(bench.CARRIERS_PER_MESSAGE)]
    scheduler_outputs = {}
    for label, mode in (("DCASS Temporal WGAN", "gan"), ("DCASS Hybrid Engine", "hybrid")):
        vals = []
        for _ in range(20):
            sched = scheduler.schedule(
                media_ids=media_ids,
                mode=mode,
                base_delay=bench.BASE_DELAY,
                time_of_day=14.0,
            )
            vals.append(score(model, sched["delays"], sched["channels"]))
        scheduler_outputs[label] = np.asarray(vals)
        report(label, vals)
    wgan = scheduler_outputs["DCASS Temporal WGAN"]
    hybrid = scheduler_outputs["DCASS Hybrid Engine"]

    print()
    print("Separation of the DCASS schedules from each human population:")
    for label, arr in (
        ("corpus held-out", a_corpus),
        ("synthetic surrogate", a_sur),
        ("independent human", a_indep),
    ):
        print(
            f"  DCASS score must exceed the {label} p95 to separate cleanly: "
            f"{label} p95 = {np.percentile(arr,95):.4f}"
        )
    print()
    print(f"  synthetic bot p05 = {np.percentile(a_bot,5):.4f}")

    # Record the measurement, so the numbers quoted in the documentation
    # have an artifact behind them.
    results_path = (
        PROJECT_ROOT / "storage" / "benchmarks" / "traffic_detector_generalization.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "schema": "traffic-detector-generalization-v1",
                "detector": "storage/models/traffic_detector.pt",
                "note": (
                    "The detector's score on six traffic populations. The "
                    "independent human generator is a Weibull-mixture model the "
                    "detector has never seen in any form, so it measures "
                    "generalization rather than memorization of the training "
                    "corpus. The human corpus is the repository's own traffic "
                    "data, whose time-of-day distribution is uniform and whose "
                    "delay values are bounded, so it is a synthetic surrogate "
                    "rather than recorded human browsing."
                ),
                "populations": {
                    "corpus_human_heldout": _stats(a_corpus),
                    "benchmark_synthetic_surrogate": _stats(a_sur),
                    "independent_human_generator": _stats(a_indep),
                    "synthetic_bot": _stats(a_bot),
                    "dcass_temporal_wgan": _stats(wgan),
                    "dcass_hybrid_engine": _stats(hybrid),
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved generalization results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

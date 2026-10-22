#!/usr/bin/env python3
"""
Compare candidate soft-branch designs on the same evidence.

The soft branch has to answer one question: is this traffic machine-like?
It has two available signals, and they are not commensurable:

  * the trained detector's calibrated bot probability, in [0, 1], whose
    human null is about 0.002;
  * the KS statistic against the human reference, whose human null is
    about 0.12 for a burst and about 0.28 for a whole session, because a
    50-item sample cannot match a 5000-item pool.

Adding a signal with a large null to a well-calibrated one dilutes it. This
script measures each design's separation and its false-positive rate on
held-out human traffic, so the choice is made from data.

Designs:
  A  detector only, threshold calibrated on human traffic
  B  detector + KS excess over the burst null (0.12)
  C  detector + KS excess over the session null (0.28)
  D  detector + raw KS
Only A to D are implemented. E and F, the union of two independently
calibrated rules, were considered and dropped: the union's false-positive
rate is the sum of the two rates, so it cannot hold a 5 percent target
without each rule being calibrated below it.
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
from train_traffic_detector import load_human_sessions, make_bot_session  # noqa: E402

W_A = 0.35
W_K = 0.45
BURST_NULL = 0.124
SESSION_NULL = 0.284


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "b3", PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["b3"] = module
    spec.loader.exec_module(module)
    return module


def detector_score(model, delays, channels) -> float:
    n = len(delays)
    d = np.zeros(64, dtype=np.float32)
    c = np.zeros(64, dtype=np.int64)
    d[:n] = np.asarray(delays[:n], dtype=np.float32)
    c[:n] = np.asarray(channels[:n], dtype=np.int64)
    with torch.no_grad():
        v = model(
            torch.tensor(d).unsqueeze(0),
            torch.tensor(c).unsqueeze(0),
            mask=torch.ones(1, 64),
        )
    return float(v.anomaly_scores.numpy()[0][:n].mean())


def design_scores(anomaly: float, ks: float) -> dict[str, float]:
    ks_burst = max(0.0, (ks - BURST_NULL) / (1 - BURST_NULL))
    ks_sess = max(0.0, (ks - SESSION_NULL) / (1 - SESSION_NULL))
    return {
        "A_detector_only": anomaly,
        "B_burst_null": W_A * anomaly + W_K * ks_burst,
        "C_session_null": W_A * anomaly + W_K * ks_sess,
        "D_raw_ks": W_A * anomaly + W_K * ks,
    }


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

    rng = np.random.default_rng(11)
    human = load_human_sessions()
    surrogate = bench.generate_human_baseline_delays(5000)

    # The detector's own holdout split, so the calibration population is
    # genuinely unseen.
    hold_idx = ckpt.get("holdout_human_indices") or []
    corpus = [human[i] for i in hold_idx if 0 <= i < len(human)][:300]
    if not corpus:
        corpus = human[-300:]

    pop: dict[str, list[tuple[float, float]]] = {"corpus held-out": [], "surrogate": []}
    for s in corpus:
        d = np.asarray(s["delays"], dtype=np.float64)[:50]
        c = np.asarray(s["channels"], dtype=np.int64)[:50]
        pop["corpus held-out"].append(
            (detector_score(model, d, c), ks_2samp(d, surrogate).statistic)
        )
    for _ in range(300):
        d = surrogate[rng.choice(len(surrogate), 50, replace=False)]
        c = rng.integers(0, 3, 50)
        pop["surrogate"].append(
            (detector_score(model, d, c), ks_2samp(d, surrogate).statistic)
        )

    bots = []
    for s in human[:300]:
        b = make_bot_session(s, rng)
        d = np.asarray(b["delays"], dtype=np.float64)[:50]
        c = np.asarray(b["channels"], dtype=np.int64)[:50]
        bots.append((detector_score(model, d, c), ks_2samp(d, surrogate).statistic))

    scheduler = bench.StealthScheduler(
        num_channels=bench.NUM_CHANNELS, device="cpu", profile="casual", seed=bench.SEED
    )
    scheduler._load_generator(PROJECT_ROOT / "storage" / "models" / "gan_generator.pt")
    media_ids = [f"carrier_{i:04d}" for i in range(bench.CARRIERS_PER_MESSAGE)]
    dcass: dict[str, list[tuple[float, float]]] = {}
    for label, mode in (
        ("DCASS Temporal WGAN", "gan"),
        ("DCASS Hybrid Engine", "hybrid"),
    ):
        rows = []
        for _ in range(30):
            s = scheduler.schedule(
                media_ids=media_ids, mode=mode, base_delay=bench.BASE_DELAY, time_of_day=14.0
            )
            d = np.asarray(s["delays"], dtype=np.float64)
            c = np.asarray(s["channels"], dtype=np.int64)
            rows.append((detector_score(model, d, c), ks_2samp(d, surrogate).statistic))
        dcass[label] = rows

    human_rows = pop["corpus held-out"] + pop["surrogate"]
    h_anom = np.array([a for a, _ in human_rows])
    h_ks = np.array([k for _, k in human_rows])
    b_anom = np.array([a for a, _ in bots])
    b_ks = np.array([k for _, k in bots])
    w_anom = np.array([a for a, _ in dcass["DCASS Temporal WGAN"]])
    w_ks = np.array([k for _, k in dcass["DCASS Temporal WGAN"]])
    y_anom = np.array([a for a, _ in dcass["DCASS Hybrid Engine"]])
    y_ks = np.array([k for _, k in dcass["DCASS Hybrid Engine"]])

    def calibrate(values: np.ndarray, target: float = 0.05) -> float:
        thr = float(np.quantile(values, 1.0 - target))
        if (values >= thr).mean() > target + 1e-9:
            thr = float(values.max()) + max(float(np.ptp(values)), 1e-3) * 1e-3
        return thr

    def pct(x) -> float:
        return 100.0 * float(np.asarray(x, dtype=float).mean())

    designs = ["A_detector_only", "B_burst_null", "C_session_null", "D_raw_ks"]
    print(f"{'design':18s} {'thr':>8s} {'human FPR':>10s} {'bot flag':>9s} "
          f"{'WGAN':>7s} {'Hybrid':>7s}")
    print("-" * 66)
    for design in designs:
        h = np.array([design_scores(a, k)[design] for a, k in human_rows])
        b = np.array([design_scores(a, k)[design] for a, k in bots])
        w = np.array([design_scores(a, k)[design] for a, k in dcass["DCASS Temporal WGAN"]])
        y = np.array([design_scores(a, k)[design] for a, k in dcass["DCASS Hybrid Engine"]])
        thr = calibrate(h)
        print(
            f"{design:18s} {thr:8.4f} {pct(h >= thr):9.1f}% {pct(b >= thr):8.1f}% "
            f"{pct(w >= thr):6.1f}% {pct(y >= thr):6.1f}%"
        )

    # Design E: the two as independently calibrated rules, unioned, each
    # calibrated at 5%.
    a_thr = calibrate(h_anom)
    k_thr = calibrate(h_ks)
    print()
    print(f"detector threshold {a_thr:.6f}   KS threshold {k_thr:.4f}")
    print()
    print(f"{'design':22s} {'human FPR':>10s} {'bot flag':>9s} {'WGAN':>7s} {'Hybrid':>7s}")
    print("-" * 62)
    print(
        f"{'E detector OR ks':22s} {pct((h_anom >= a_thr) | (h_ks >= k_thr)):9.1f}% "
        f"{pct((b_anom >= a_thr) | (b_ks >= k_thr)):8.1f}% "
        f"{pct((w_anom >= a_thr) | (w_ks >= k_thr)):6.1f}% "
        f"{pct((y_anom >= a_thr) | (y_ks >= k_thr)):6.1f}%"
    )

    # Design F: each threshold chosen so the union holds 5% on human
    # traffic. Scale the KS threshold until the union rate is on target.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = float(((h_anom >= a_thr) | (h_ks >= mid)).mean())
        if rate > 0.05:
            hi = mid
        else:
            lo = mid
    k_thr_f = hi
    print(
        f"{'F detector OR ks (5%)':22s} {pct((h_anom >= a_thr) | (h_ks >= k_thr_f)):9.1f}% "
        f"{pct((b_anom >= a_thr) | (b_ks >= k_thr_f)):8.1f}% "
        f"{pct((w_anom >= a_thr) | (w_ks >= k_thr_f)):6.1f}% "
        f"{pct((y_anom >= a_thr) | (y_ks >= k_thr_f)):6.1f}%"
    )
    print(f"   (KS threshold {k_thr_f:.4f})")

    # Record the comparison, so the claim about why the KS term is not part
    # of the soft branch has an artifact behind it.
    results_path = (
        PROJECT_ROOT / "storage" / "benchmarks" / "soft_branch_design_comparison.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    designs = {}
    for design in ("A_detector_only", "B_burst_null", "C_session_null", "D_raw_ks"):
        h = np.array([design_scores(a, k)[design] for a, k in human_rows])
        b = np.array([design_scores(a, k)[design] for a, k in bots])
        w = np.array([design_scores(a, k)[design] for a, k in dcass["DCASS Temporal WGAN"]])
        y = np.array([design_scores(a, k)[design] for a, k in dcass["DCASS Hybrid Engine"]])
        thr = calibrate(h)
        designs[design] = {
            "threshold": thr,
            "human_fpr_pct": pct(h >= thr),
            "synthetic_bot_flagged_pct": pct(b >= thr),
            "dcass_wgan_flagged_pct": pct(w >= thr),
            "dcass_hybrid_flagged_pct": pct(y >= thr),
        }
    results_path.write_text(
        json.dumps(
            {
                "schema": "soft-branch-design-comparison-v1",
                "note": (
                    "Each candidate soft-branch score is thresholded at the "
                    "smallest value whose false-positive rate on held-out human "
                    "traffic is at or below 5 percent, and then evaluated on "
                    "synthetic bot traffic and on the two DCASS schedules. "
                    "Design A, the detector alone, is the one the benchmark "
                    "uses: the KS term's null against the human reference is "
                    "about 0.12 for a burst and 0.28 for a whole session, so "
                    "adding it to a calibrated detector dilutes the detector."
                ),
                "human_populations": {
                    "corpus_held_out_sessions": len(pop["corpus held-out"]),
                    "surrogate_bursts": len(pop["surrogate"]),
                },
                "designs": designs,
            },
            indent=2,
        )
    )
    print(f"\nSaved design comparison to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

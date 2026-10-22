#!/usr/bin/env python3
"""
Measurement for the C3 soft-branch redesign (remediation item B1-b).

Measures the candidate soft-branch terms on the real scheduler outputs and
on held-out human traffic, so the weighting is chosen from data rather than
guessed.

Terms:
  anomaly   the trained detector's mean per-timestep anomaly score
  ks_sur    KS statistic against the benchmark's synthetic surrogate
  ks_corpus KS statistic against the pooled corpus human traffic

and the null behaviour of each term on human traffic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from train_traffic_detector import (  # noqa: E402
    DEVICE,
    SEQ_LEN,
    load_human_sessions,
    make_bot_session,
)
from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "b3", PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["b3"] = module
    spec.loader.exec_module(module)
    return module


def load_detector():
    ckpt = torch.load(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    model = DeepPacketInspectionWarden(num_channels=3).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("config", {})


def score_session(model, delays: np.ndarray, channels: np.ndarray) -> float:
    n = min(len(delays), SEQ_LEN)
    pad_d = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_c = np.zeros(SEQ_LEN, dtype=np.int64)
    mask = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_d[:n] = np.asarray(delays[:n], dtype=np.float32)
    pad_c[:n] = np.asarray(channels[:n], dtype=np.int64)
    mask[:n] = 1.0
    with torch.no_grad():
        verdict = model(
            torch.tensor(pad_d).unsqueeze(0).to(DEVICE),
            torch.tensor(pad_c).unsqueeze(0).to(DEVICE),
            mask=torch.tensor(mask).unsqueeze(0).to(DEVICE),
        )
    return float(verdict.anomaly_scores.cpu().numpy()[0][:n].mean())


def main() -> int:
    bench = load_benchmark()
    model, config = load_detector()

    print("Loading corpus and detector...")
    human = load_human_sessions()
    rng = np.random.default_rng(0)

    corpus_pool = np.concatenate(
        [np.asarray(x["delays"], dtype=np.float64) for x in human]
    )
    surrogate = bench.generate_human_baseline_delays(5000)

    # A size-matched corpus reference: the benchmark compares a 50-item
    # burst against a 5000-item pool, so the null KS is not zero.
    print(f"  corpus pool: {len(corpus_pool)} delays; surrogate pool: {len(surrogate)}")

    # Human sessions, held out from anything the detector saw.
    human_probe = human[-400:]

    bots = [make_bot_session(s, rng) for s in human[:400]]

    # The benchmark's own human self-test protocol: 50-item bursts drawn
    # from the surrogate reference, which is what `run_human_reference_self_test`
    # does. This is the natural null for the C3 KS term.
    print("\nScoring with the benchmark's self-test protocol...")
    self_test_rows = []
    for _ in range(400):
        pick = rng.choice(len(surrogate), 50, replace=False)
        d = surrogate[pick].astype(np.float64)
        c = rng.integers(0, 3, 50)
        self_test_rows.append(
            {
                "anomaly": score_session(model, d, c),
                "ks_sur": ks_2samp(d, surrogate).statistic,
                "ks_corpus": ks_2samp(d, corpus_pool).statistic,
            }
        )
    bot_rows = []
    for session in bots:
        d = np.asarray(session["delays"], dtype=np.float64)[:50]
        c = np.asarray(session["channels"], dtype=np.int64)[:50]
        bot_rows.append(
            {
                "anomaly": score_session(model, d, c),
                "ks_sur": ks_2samp(d, surrogate).statistic,
                "ks_corpus": ks_2samp(d, corpus_pool).statistic,
            }
        )
    print(f"  synthesized bot bursts: n={len(bot_rows)}")

    h = self_test_rows
    b = bot_rows

    def summarize(rows, label):
        a = np.array([r["anomaly"] for r in rows])
        ks_s = np.array([r["ks_sur"] for r in rows])
        ks_c = np.array([r["ks_corpus"] for r in rows])
        print(f"\n{label}:")
        print(f"  anomaly    mean {a.mean():.4f}  p95 {np.percentile(a,95):.4f}  max {a.max():.4f}")
        print(f"  ks_sur     mean {ks_s.mean():.4f}  p95 {np.percentile(ks_s,95):.4f}")
        print(f"  ks_corpus  mean {ks_c.mean():.4f}  p95 {np.percentile(ks_c,95):.4f}")
        return a, ks_s, ks_c

    h_a, h_ks, h_kc = summarize(h, "HUMAN (benchmark self-test protocol)")
    b_a, b_ks, b_kc = summarize(b, "BOT")

    print("\nSeparation of each candidate term (human vs bot):")
    for name, hv, bv in (
        ("anomaly", h_a, b_a),
        ("ks_sur", h_ks, b_ks),
        ("ks_corpus", h_kc, b_kc),
    ):
        # Threshold-free: AUC via rank.
        from sklearn.metrics import roc_auc_score

        y = np.concatenate([np.zeros(len(hv)), np.ones(len(bv))])
        x = np.concatenate([hv, bv])
        print(f"  {name:10s} AUC {roc_auc_score(y, x):.4f}   human p95 {np.percentile(hv,95):.4f}   bot p05 {np.percentile(bv,5):.4f}")

    print("\nCandidate composites, flagged rate on human vs bot:")
    for w_a in (0.0, 0.25, 0.35, 0.5, 0.75, 1.0):
        for k_name, hk, bk in (("sur", h_ks, b_ks), ("corpus", h_kc, b_kc)):
            hs = w_a * h_a + (1 - w_a) * hk
            bs = w_a * b_a + (1 - w_a) * bk
            for thr in (0.50,):
                print(
                    f"  w_a={w_a:.2f} ks={k_name:6s} thr={thr:.2f}  "
                    f"human flagged {100*(hs>=thr).mean():5.1f}%   bot flagged {100*(bs>=thr).mean():5.1f}%"
                )

    print("\nProposed un-normalized form 0.35*anomaly + 0.45*ks:")
    for k_name, hk, bk in (("sur", h_ks, b_ks), ("corpus", h_kc, b_kc)):
        hs = 0.35 * h_a + 0.45 * hk
        bs = 0.35 * b_a + 0.45 * bk
        for thr in (0.30, 0.40, 0.50):
            print(
                f"  ks={k_name:6s} thr={thr:.2f}  "
                f"human flagged {100*(hs>=thr).mean():5.1f}%   bot flagged {100*(bs>=thr).mean():5.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

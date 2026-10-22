#!/usr/bin/env python3
"""
Track a training run's critic: does it still read its input?

The WGAN-GP critic's gradient with respect to its input is the quantity
the gradient penalty constrains, and it is what tells you whether the
adversary is doing anything. A critic whose score is the same constant
for every input has a gradient of zero, the penalty sits at exactly
lambda_gp, and the penalty is inert.

This script walks a run's checkpoints and records, for each one, the
critic's input gradient norm, the penalty it implies, and the gap
between the critic's score on real traffic and on the generator's own
traffic. A gap of zero with a gradient of zero is a collapsed critic.

The measurement is taken on the corpus the critic trained on, not on a
synthetic batch, because the critic's sensitivity to an input it has
never seen is not the quantity of interest.

USAGE
-----
    .venv/bin/python scripts/analysis/track_critic_sensitivity.py \
        --run /home/img1/gan_runs/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from src.utils.checkpoint_io import load_trained_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import (  # noqa: E402
    DeepPacketInspectionWarden,
    compute_gradient_penalty,
)
from src.stealth.delay_contract import canonical_delay_mapping  # noqa: E402
from src.stealth.gan.generator import TemporalPatternGenerator  # noqa: E402

CORPUS = PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
SEQUENCE = 60
BATCH = 32
LAMBDA_GP = 10.0


def fit(session: dict, length: int = SEQUENCE) -> tuple[np.ndarray, np.ndarray]:
    """A corpus session padded to a fixed length."""
    delays = np.asarray(session["delays"], dtype=np.float32)
    channels = np.asarray(session["channels"])
    n = min(len(delays), length)
    if n == 0:
        return np.full(length, 0.5, dtype=np.float32), np.zeros(length, dtype=np.int64)
    delays = np.concatenate(
        [delays[:n], np.full(length - n, delays[-1], dtype=np.float32)]
    )
    channels = np.concatenate([channels[:n], np.full(length - n, channels[-1])])
    return delays, channels


def gradient_norm(warden, real, fake, real_ch, fake_ch) -> tuple[float, float]:
    """The critic's input gradient norm, and the penalty it implies."""
    penalty = compute_gradient_penalty(
        warden,
        torch.tensor(real, dtype=torch.float32),
        torch.tensor(fake, dtype=torch.float32),
        torch.tensor(real_ch, dtype=torch.long),
        torch.tensor(fake_ch, dtype=torch.long),
        lambda_gp=LAMBDA_GP,
    )
    mean_square = float(penalty) / LAMBDA_GP
    root = float(np.sqrt(max(mean_square, 0.0)))
    norm = 1.0 - root
    if norm < 0:
        norm = 1.0 + root
    return float(norm), float(penalty)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="The run's checkpoint directory.")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the JSON summary. Defaults to the run directory.",
    )
    args = parser.parse_args()

    run = Path(args.run)
    checkpoints = sorted(run.glob("epoch_*.pt"))
    if not checkpoints:
        print(f"ERROR: no checkpoints under {run}")
        return 1

    corpus = json.loads(CORPUS.read_text())
    rng = np.random.default_rng(0)
    fitted = [
        fit(corpus[i]) for i in rng.integers(0, len(corpus), BATCH)
    ]
    real_delays = np.stack([f[0] for f in fitted])
    real_channels = np.stack([f[1] for f in fitted])

    history = []
    print(
        f"{'epoch':>6}  {'||grad_x D||':>12}  {'penalty':>8}  "
        f"{'real':>9}  {'fake':>9}  {'gap':>8}"
    )
    for path in checkpoints:
        checkpoint = load_trained_checkpoint(path, map_location="cpu",)

        critic = DeepPacketInspectionWarden(num_channels=3, hidden_dim=256)
        critic.load_state_dict(checkpoint["warden_state"])
        critic.eval()

        generator = TemporalPatternGenerator(
            latent_dim=128, hidden_dim=256, num_channels=3, max_sequence_length=SEQUENCE
        )
        generator.load_state_dict(checkpoint["generator_state"])
        generator.eval()
        with torch.no_grad():
            schedule = generator(
                torch.randn(BATCH, 128), SEQUENCE, torch.full((BATCH,), 14.0)
            )
        fake_delays = canonical_delay_mapping(schedule.delays.numpy())
        fake_channels = schedule.channel_probs_straight_through().argmax(-1).numpy()

        norm, penalty = gradient_norm(
            critic, real_delays, fake_delays, real_channels, fake_channels
        )
        with torch.no_grad():
            real_score = float(
                critic(
                    torch.tensor(real_delays, dtype=torch.float32),
                    torch.tensor(real_channels, dtype=torch.long),
                )
                .feature_importance["raw_critic_score"]
                .mean()
            )
            fake_score = float(
                critic(
                    torch.tensor(fake_delays, dtype=torch.float32),
                    torch.tensor(fake_channels, dtype=torch.long),
                )
                .feature_importance["raw_critic_score"]
                .mean()
            )

        row = {
            "epoch": int(path.stem.split("_")[-1]),
            "input_gradient_norm": round(norm, 8),
            "gradient_penalty": round(penalty, 6),
            "real_score": round(real_score, 4),
            "fake_score": round(fake_score, 4),
            "score_gap": round(real_score - fake_score, 4),
            "collapsed": bool(norm < 0.01 and abs(real_score - fake_score) < 0.5),
        }
        history.append(row)
        print(
            f"{row['epoch']:>6}  {row['input_gradient_norm']:>12.6f}  "
            f"{row['gradient_penalty']:>8.4f}  {row['real_score']:>9.3f}  "
            f"{row['fake_score']:>9.3f}  {row['score_gap']:>8.3f}"
            + ("   COLLAPSED" if row["collapsed"] else "")
        )

    collapsed = [r["epoch"] for r in history if r["collapsed"]]
    out = Path(args.out) if args.out else run / "critic_sensitivity.json"
    out.write_text(
        json.dumps(
            {
                "schema": "critic-sensitivity-track-v1",
                "run": str(run),
                "lambda_gp": LAMBDA_GP,
                "note": (
                    "The critic's gradient with respect to its input, "
                    "measured on the corpus the critic trained on, and the "
                    "gap between its score on real traffic and on the "
                    "generator's own traffic. A collapsed epoch has a "
                    "gradient near zero and a score gap near zero: the "
                    "adversary is not discriminating anything."
                ),
                "collapsed_epochs": collapsed,
                "final_epoch_collapsed": bool(
                    history and history[-1]["collapsed"]
                ),
                "history": history,
            },
            indent=2,
        )
    )
    print(f"\nCollapsed epochs: {collapsed}")
    print(f"Final epoch collapsed: {bool(history and history[-1]['collapsed'])}")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

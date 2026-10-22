#!/usr/bin/env python3
"""
Verify the repaired generator path.

The generator had three defects, all measured in
`docs/benchmarking/C3_GENERATOR_ADVERSARY.md`:

1. the critic was fed the generator's raw output while the real traffic
   was in seconds, so it never saw the traffic the scheduler sends;
2. the delay mapping left a fresh generator with no gradient at all,
   because the usable delay range is a 1.97 percent slice of the head's
   output range and every output landed at the clip;
3. the critic's score was unbounded, so the Wasserstein loss had no
   anchor and the critic's input gradient collapsed.

This script checks that each is fixed, using the shipped checkpoints.

Outputs:
    storage/benchmarks/generator_path_repair.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import (  # noqa: E402
    DeepPacketInspectionWarden,
    compute_gradient_penalty,
)
from src.stealth.delay_contract import (  # noqa: E402
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    canonical_delay_mapping,
    canonical_delay_mapping_torch,
)
from src.stealth.gan.generator import TemporalPatternGenerator  # noqa: E402

# The measurement is small, and a busy GPU is a reason to run it on
# the CPU rather than to fail.
DEVICE = os.environ.get(
    "DCASS_VERIFY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
SEQ = 60
N = 64
LAMBDA_GP = 10.0


def critic_input_gradient_norm(warden, batch) -> float:
    """The mean norm of the critic's gradient with respect to its input.

    This is the quantity the gradient penalty constrains. A penalty
    sitting at exactly lambda_gp means the norm is zero, and the
    constraint has stopped applying rather than merely being violated.
    """
    real, fake, channels, fake_channels = batch
    penalty = compute_gradient_penalty(
        warden,
        real,
        fake,
        channels,
        fake_channels,
        lambda_gp=LAMBDA_GP,
    )
    mean_square = float(penalty) / LAMBDA_GP
    root = float(np.sqrt(max(mean_square, 0.0)))
    norm = 1.0 - root
    if norm < 0:
        norm = 1.0 + root
    return float(norm)


def corpus_batch(size: int = 32, length: int = 60, seed: int = 0):
    """A batch of the corpus sessions the critic trained on.

    The critic's sensitivity to a synthetic batch it has never seen is
    not the quantity of interest. Its sensitivity to the distribution it
    was trained against is, and that is what the gradient penalty was
    supposed to constrain.
    """
    corpus = json.loads(
        (
            PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
        ).read_text()
    )
    rng = np.random.default_rng(seed)
    picked = [corpus[i] for i in rng.integers(0, len(corpus), size)]
    real, channels = [], []
    for session in picked:
        delays = np.asarray(session["delays"], dtype=np.float32)
        chan = np.asarray(session["channels"])
        n = min(len(delays), length)
        if n == 0:
            continue
        real.append(
            np.concatenate(
                [delays[:n], np.full(length - n, delays[-1], dtype=np.float32)]
            )
        )
        channels.append(np.concatenate([chan[:n], np.full(length - n, chan[-1])]))
    return (
        torch.tensor(np.stack(real), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.stack(channels), dtype=torch.long, device=DEVICE),
    )


def describe_delays(values: np.ndarray) -> dict:
    return {
        "min": round(float(values.min()), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
    }


def main() -> int:
    results: dict = {}

    # --- The generator's output and its gradient path -------------------
    ckpt = torch.load(
        PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    generator = TemporalPatternGenerator(
        latent_dim=128, hidden_dim=256, num_channels=3, max_sequence_length=SEQ
    )
    generator.load_state_dict(ckpt["generator_state"])
    generator.to(DEVICE).eval()

    z = torch.randn(N, 128, device=DEVICE, requires_grad=True)
    tod = torch.full((N,), 14.0, device=DEVICE)
    schedule = generator(z, SEQ, tod)
    raw = schedule.delays
    mapped = canonical_delay_mapping_torch(raw)
    grad = torch.autograd.grad(mapped.sum(), raw, retain_graph=True)[0]

    mapped_np = canonical_delay_mapping(raw.detach().cpu().numpy())
    corpus = json.loads(
        (
            PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
        ).read_text()
    )
    corpus_delays = np.concatenate(
        [np.asarray(s["delays"], dtype=float) for s in corpus]
    )

    results["generator_output"] = {
        "note": (
            "The generator's delays, mapped the way the scheduler maps "
            "them, against the human corpus it is trained against."
        ),
        "raw": {
            "min": round(float(raw.min()), 6),
            "p50": round(float(raw.median()), 6),
            "max": round(float(raw.max()), 6),
        },
        "mapped_seconds": describe_delays(mapped_np),
        "corpus_seconds": describe_delays(corpus_delays),
        "fraction_at_the_ceiling": round(
            float((mapped_np >= GAN_DELAY_MAX - 1e-6).mean()), 6
        ),
        "fraction_at_the_floor": round(
            float((mapped_np <= GAN_DELAY_MIN + 1e-6).mean()), 6
        ),
    }
    results["gradient_path"] = {
        "note": (
            "The gradient that reaches the generator's delay head "
            "through the canonical delay mapping. A zero gradient means "
            "the generator cannot be trained at all."
        ),
        "zero_gradient_pct": round(100 * float((grad == 0).float().mean()), 4),
        "gradient_median": round(float(grad.abs().median()), 6),
    }

    # --- The critic's input gradient ------------------------------------
    critic_state = ckpt.get("warden_state")
    if critic_state is not None:
        critic = DeepPacketInspectionWarden(
            num_channels=3,
            hidden_dim=256,
            bounded_critic=bool(ckpt.get("critic_bounded", False)),
            critic_score_bound=float(ckpt.get("critic_score_bound", 10.0)),
        )
        critic.load_state_dict(critic_state)
        critic.to(DEVICE).eval()

        real_delays, real_channels = corpus_batch(size=N)
        fake_delays = torch.tensor(
            mapped_np, dtype=torch.float32, device=DEVICE
        )
        fake_channels = schedule.channel_probs_straight_through().argmax(-1)

        norm = critic_input_gradient_norm(
            critic, (real_delays, fake_delays, real_channels, fake_channels)
        )
        with torch.no_grad():
            real_score = float(
                critic(real_delays, real_channels)
                .feature_importance["raw_critic_score"]
                .mean()
            )
            fake_score = float(
                critic(fake_delays, fake_channels)
                .feature_importance["raw_critic_score"]
                .mean()
            )
        results["critic"] = {
            "note": (
                "The trained critic's gradient with respect to its "
                "input, the quantity the gradient penalty constrains, "
                "measured on the corpus the critic trained on, together "
                "with the gap between its score on real traffic and on "
                "the generator's own traffic. A collapsed critic has a "
                "gradient near zero and a score gap near zero."
            ),
            "bounded_critic": bool(ckpt.get("critic_bounded", False)),
            "critic_score_bound": ckpt.get("critic_score_bound"),
            "input_gradient_norm": round(norm, 8),
            "gradient_penalty": round(
                float(LAMBDA_GP * (norm - 1.0) ** 2), 6
            ),
            "real_score": round(real_score, 4),
            "fake_score": round(fake_score, 4),
            "score_gap": round(real_score - fake_score, 4),
            "collapsed": bool(norm < 0.01 and abs(real_score - fake_score) < 0.5),
        }

    out = PROJECT_ROOT / "storage" / "benchmarks" / "generator_path_repair.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    g = results["generator_output"]
    print("Generator's mapped delays:")
    print(f"  generated: p10 {g['mapped_seconds']['p10']}  p50 {g['mapped_seconds']['p50']}  "
          f"p90 {g['mapped_seconds']['p90']}  max {g['mapped_seconds']['max']}")
    print(f"  corpus   : p10 {g['corpus_seconds']['p10']}  p50 {g['corpus_seconds']['p50']}  "
          f"p90 {g['corpus_seconds']['p90']}  max {g['corpus_seconds']['max']}")
    print(f"  at the ceiling: {g['fraction_at_the_ceiling'] * 100:.2f}%   "
          f"at the floor: {g['fraction_at_the_floor'] * 100:.2f}%")
    print()
    print(f"Gradient path: {results['gradient_path']['zero_gradient_pct']}% zero")
    if "critic" in results:
        c = results["critic"]
        print(f"Critic: bounded={c['bounded_critic']} bound={c['critic_score_bound']} "
              f"||grad_x D||={c['input_gradient_norm']} penalty={c['gradient_penalty']}")
        print(f"  score on real {c['real_score']}  on the generator's own traffic "
              f"{c['fake_score']}  gap {c['score_gap']}"
              + ("   COLLAPSED" if c["collapsed"] else ""))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

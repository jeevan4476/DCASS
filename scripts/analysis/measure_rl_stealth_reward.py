#!/usr/bin/env python3
"""
Record the reward the RL environment assigns to bot-like and human-like
traffic, under the shipped WGAN-GP critic and under a working detector.

The Component 3 evaluation document quotes this table to show that the
shipped agent was not merely unstealthed but was rewarded for bot-like
traffic: with the collapsed critic the stealth penalty is identically
zero, so the reward reduces to throughput plus diversity and prefers the
faster, more regular traffic.

Outputs:
    storage/benchmarks/rl_stealth_reward_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.gan.trainer import apply_critic_config  # noqa: E402
from src.stealth.rl.environment import StealthEnvironment  # noqa: E402

TRAFFIC = {
    "fixed 3.0 s metronome": [3.0] * 20,
    "fixed 12.0 s": [12.0] * 20,
    "fixed 30.0 s": [30.0] * 20,
}


def load_warden(path: Path, key: str) -> DeepPacketInspectionWarden:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.load_state_dict(ckpt[key] if key in ckpt else ckpt["state_dict"])
    # A checkpoint whose critic was trained with a bounded score is a
    # different model, so the bound travels with the weights.
    apply_critic_config(warden, ckpt)
    warden.eval()
    for p in warden.parameters():
        p.requires_grad_(False)
    return warden


def probe(warden: DeepPacketInspectionWarden, delays: list[float], lambda_stealth: float) -> dict:
    """Run the environment's reward on one fixed delay sequence."""
    env = StealthEnvironment(num_channels=3, warden=warden, lambda_stealth=lambda_stealth)
    state = env.reset([f"carrier_{i:04d}" for i in range(len(delays))])
    total = 0.0
    for i, d in enumerate(delays):
        state, reward, done, _ = env.step({"delay": d, "channel": i % 3})
        total += reward
        if done:
            break
    return {
        "mean_reward": float(total / len(delays)),
        "warden_score": float(env.get_warden_score()),
    }


def main() -> int:
    lambda_stealth = 50.0
    wardens = {
        "shipped_wgan_gp_critic": load_warden(
            PROJECT_ROOT / "storage" / "models" / "gan_generator.pt", "warden_state"
        ),
        "trained_detector": load_warden(
            PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt", "state_dict"
        ),
    }

    results = {}
    for name, warden in wardens.items():
        results[name] = {
            label: probe(warden, delays, lambda_stealth)
            for label, delays in TRAFFIC.items()
        }

    out = PROJECT_ROOT / "storage" / "benchmarks" / "rl_stealth_reward_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "rl-stealth-reward-probe-v1",
                "lambda_stealth": lambda_stealth,
                "note": (
                    "The mean per-step reward the RL environment assigns to "
                    "each fixed delay sequence, under the shipped WGAN-GP critic "
                    "and under a trained traffic detector. The critic returns "
                    "bot_probability = 0.0 for every input, so its stealth "
                    "penalty is identically zero and the reward prefers the "
                    "faster traffic."
                ),
                "wardens": results,
            },
            indent=2,
        )
    )
    print(f"Saved {out}")
    for name, rows in results.items():
        for label, row in rows.items():
            print(f"  {name:24s} {label:26s} score {row['warden_score']:.4f}  reward {row['mean_reward']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

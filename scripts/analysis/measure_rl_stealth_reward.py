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
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint_io import load_trained_checkpoint  # noqa: E402
from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from benchmark_manifest import build_manifest, capture_git_state, write_manifest  # noqa: E402
from src.stealth.gan.trainer import apply_critic_config  # noqa: E402
from src.stealth.rl.environment import StealthEnvironment  # noqa: E402

TRAFFIC = {
    # The metronome row is extended by the environment's rate-limit
    # back-offs: on the i%3 rotation the cooldowns are 6/12/4 s, so the
    # channel-1 steps of a 3.0 s metronome are stretched past their
    # nominal 3.0 s before they are scored. Disclosed on the quoted
    # table (docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md).
    "fixed 3.0 s metronome": [3.0] * 20,
    "fixed 12.0 s": [12.0] * 20,
    # canonical_rl_delay clamps every step into the pacing window
    # [0.5, 20.0] inside env.step, so this row scores as a fixed
    # 20.0 s clock, not a 30 s one.
    "fixed 20.0 s (30 clamped to the canonical window)": [30.0] * 20,
}


def load_warden(path: Path, key: str) -> DeepPacketInspectionWarden:
    ckpt = load_trained_checkpoint(path, map_location="cpu",)
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
    executed = env.transmission_history
    return {
        "mean_reward": float(total / len(delays)),
        "warden_score": float(env.get_warden_score()),
        "requested_steps": len(delays),
        "completed_steps": len(executed),
        "effective_delays_seconds": [float(row.delay_from_previous) for row in executed],
        "effective_channels": [int(row.channel_id) for row in executed],
        "backoff_steps": int(env.rate_limit_backoffs),
    }


def main() -> int:
    git_state = capture_git_state()
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

    benchmark_dir = Path(
        os.environ.get(
            "DCASS_BENCHMARK_DIR", str(PROJECT_ROOT / "storage" / "benchmarks")
        )
    ).expanduser().resolve()
    out = benchmark_dir / "rl_stealth_reward_probe.json"
    manifest_path = benchmark_dir / "rl_stealth_reward_probe_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "rl-stealth-reward-probe-v1",
                "lambda_stealth": lambda_stealth,
                "provenance": {
                    "git": git_state,
                    "device": "cpu",
                },
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
    manifest = build_manifest(
        component="rl_stealth_reward_probe",
        outputs=[out],
        inputs=[
            PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
            PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt",
            PROJECT_ROOT / "scripts" / "analysis" / "measure_rl_stealth_reward.py",
            PROJECT_ROOT / "scripts" / "analysis" / "benchmark_manifest.py",
            PROJECT_ROOT / "src" / "stealth" / "rl" / "environment.py",
            PROJECT_ROOT / "src" / "stealth" / "rl" / "pacing.py",
            PROJECT_ROOT / "src" / "stealth" / "gan" / "trainer.py",
            PROJECT_ROOT / "src" / "stealth" / "delay_contract.py",
            PROJECT_ROOT / "src" / "analysis" / "adversarial" / "warden.py",
        ],
        seeds={"numpy_seed": None, "torch_seed": None},
        extra={
            "lambda_stealth": lambda_stealth,
            "nominal_traffic_cases": list(TRAFFIC),
            "effective_sequences_recorded": True,
        },
        git_state=git_state,
    )
    write_manifest(manifest, manifest_path)
    print(f"Saved {out}")
    print(f"Saved provenance manifest {manifest_path}")
    for name, rows in results.items():
        for label, row in rows.items():
            print(f"  {name:24s} {label:26s} score {row['warden_score']:.4f}  reward {row['mean_reward']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

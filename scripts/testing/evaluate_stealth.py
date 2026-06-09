#!/usr/bin/env python3
"""
Evaluate Stealth Quality of Generated Schedules.

Feeds generated timing schedules through the trained Warden and reports
bot-detection probability, delay distribution statistics, and comparison
against real human traffic.

Usage:
    python scripts/evaluate_stealth.py --mode gan
    python scripts/evaluate_stealth.py --mode rl
    python scripts/evaluate_stealth.py --mode static   # baseline NoiseController
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.stealth_scheduler import StealthScheduler


def load_human_traffic(path: Path, max_sessions: int = 200):
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data[:max_sessions]


def evaluate_warden(warden, delays_list, channels_list):
    """Run Warden on a batch and return average bot probability."""
    max_len = max(len(d) for d in delays_list)
    padded_d = [d + [0.0] * (max_len - len(d)) for d in delays_list]
    padded_c = [c + [0] * (max_len - len(c)) for c in channels_list]

    dt = torch.tensor(padded_d, dtype=torch.float32)
    ct = torch.tensor(padded_c, dtype=torch.long)

    with torch.no_grad():
        verdict = warden(dt, ct)
    return verdict.bot_probability.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate stealth quality")
    parser.add_argument("--mode", choices=["gan", "rl", "static"], default="static")
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--human-data", type=Path,
                        default=PROJECT_ROOT / "data" / "behavioral" / "human_traffic.json")
    parser.add_argument("--gan-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "gan" / "final.pt")
    parser.add_argument("--rl-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "rl" / "ppo_agent_final.pt")
    args = parser.parse_args()

    # --- warden ----------------------------------------------------------
    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.eval()

    # --- generate schedules ---------------------------------------------
    scheduler = StealthScheduler(num_channels=3, device=args.device)
    media_ids = [f"media_{i:04d}" for i in range(20)]

    print(f"\nEvaluating mode={args.mode} over {args.num_sequences} sequences ...\n")

    all_delays, all_channels = [], []
    for _ in range(args.num_sequences):
        result = scheduler.schedule(media_ids, mode=args.mode,
                                    gan_checkpoint=args.gan_checkpoint,
                                    rl_checkpoint=args.rl_checkpoint)
        all_delays.append(result["delays"])
        all_channels.append(result["channels"])

    # --- warden evaluation -----------------------------------------------
    bot_probs = evaluate_warden(warden, all_delays, all_channels)
    mean_bp = np.mean(bot_probs)
    flat_delays = [d for seq in all_delays for d in seq]

    print("=" * 55)
    print(f"  Mode        : {args.mode}")
    print(f"  Sequences   : {args.num_sequences}")
    print(f"  Avg Bot Prob: {mean_bp:.4f}  {'(GOOD)' if mean_bp < 0.4 else '(SUSPICIOUS)' if mean_bp < 0.6 else '(DETECTED)'}")
    print(f"  Delay Mean  : {np.mean(flat_delays):.2f}s")
    print(f"  Delay Std   : {np.std(flat_delays):.2f}s")
    print(f"  Delay CV    : {np.std(flat_delays)/max(np.mean(flat_delays),1e-8):.3f}")
    print("=" * 55)

    # --- compare to human baseline ---------------------------------------
    human = load_human_traffic(args.human_data)
    if human:
        h_delays = [s["delays"] for s in human[:args.num_sequences]]
        h_channels = [s["channels"] for s in human[:args.num_sequences]]
        h_probs = evaluate_warden(warden, h_delays, h_channels)
        h_flat = [d for seq in h_delays for d in seq]
        print(f"\n  Human baseline Bot Prob : {np.mean(h_probs):.4f}")
        print(f"  Human Delay Mean       : {np.mean(h_flat):.2f}s")
        print(f"  Human Delay Std        : {np.std(h_flat):.2f}s")


if __name__ == "__main__":
    main()

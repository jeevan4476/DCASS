#!/usr/bin/env python3
"""
Generate Synthetic Human Traffic Data for GAN/RL Training.

Creates realistic human-like social media posting patterns with:
- Circadian rhythm (more activity during day, less at night)
- Bursty behavior (clustered posts followed by idle periods)
- Channel preference patterns
- Weekend vs weekday variation

Usage:
    python scripts/generate_traffic_data.py
    python scripts/generate_traffic_data.py --num-sessions 5000 --output data/behavioral/human_traffic.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def circadian_rate(hour: float) -> float:
    """
    Return a posting-rate multiplier based on time-of-day.
    Peak activity ~14:00, lowest ~04:00.
    """
    return 0.3 + 0.7 * max(0.0, np.sin(np.pi * (hour - 6) / 14)) ** 1.5


def generate_session(
    num_channels: int = 3,
    min_posts: int = 8,
    max_posts: int = 50,
    rng: np.random.Generator = None,
) -> dict:
    """Generate a single realistic traffic session."""
    rng = rng or np.random.default_rng()

    start_hour = int(rng.choice(24, p=_hour_probs(rng)))
    seq_len = rng.integers(min_posts, max_posts + 1)
    base_rate = rng.uniform(5.0, 25.0)  # avg seconds between posts

    delays, channels = [], []
    current_hour = float(start_hour)

    # Channel preference: each session has a "favourite" channel
    fav_channel = int(rng.integers(0, num_channels))
    fav_weight = rng.uniform(0.4, 0.7)
    probs = np.full(num_channels, (1 - fav_weight) / max(num_channels - 1, 1))
    probs[fav_channel] = fav_weight

    for i in range(seq_len):
        rate_mult = circadian_rate(current_hour % 24)
        lam = base_rate / max(rate_mult, 0.1)

        # Exponential delay with occasional bursts
        if rng.random() < 0.15:
            delay = rng.uniform(0.5, 3.0)       # burst
        elif rng.random() < 0.10:
            delay = rng.uniform(60.0, 300.0)     # long idle
        else:
            delay = float(rng.exponential(lam))  # normal

        delay = max(0.5, delay)
        delays.append(round(delay, 2))
        channels.append(int(rng.choice(num_channels, p=probs)))
        current_hour += delay / 3600.0

    return {
        "delays": delays,
        "channels": channels,
        "time_of_day": start_hour,
        "num_channels": num_channels,
        "session_length": seq_len,
    }


def _hour_probs(rng: np.random.Generator) -> np.ndarray:
    """Probability of a session starting at each hour."""
    probs = np.array([circadian_rate(h) for h in range(24)])
    probs = probs / probs.sum()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic human traffic data")
    parser.add_argument("--num-sessions", type=int, default=2000)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "behavioral" / "human_traffic.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.num_sessions} traffic sessions ...")
    sessions = [
        generate_session(num_channels=args.num_channels, rng=rng)
        for _ in range(args.num_sessions)
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    total_posts = sum(s["session_length"] for s in sessions)
    print(f"Saved {args.num_sessions} sessions ({total_posts:,} total posts) → {args.output}")


if __name__ == "__main__":
    main()

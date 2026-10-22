#!/usr/bin/env python3
"""
Generate a behavioral traffic dataset for training the stealth scheduler.

The training profile in `docker-compose.yml` referenced this script and it
did not exist, so the whole `training` profile was unrunnable. This script
produces the dataset the GAN and RL trainers expect.

The output format matches `HumanTrafficDataset` in
`src/stealth/gan/trainer.py`:

    [
      {"delays": [5.2, 3.1, ...], "channels": [0, 1, ...], "time_of_day": 14},
      ...
    ]

The generated traffic is SYNTHETIC. It models human browsing as a mixture
of regimes rather than reproducing any recorded human trace:

  * active reading or posting, with short, roughly log-normal gaps;
  * a burst of rapid successive posts;
  * a distraction pause;
  * stepping away from the keyboard for a long while;
  * channel switching between platforms, with runs of the same platform.

This matters for how the result may be read. A detector or a generator
trained on this corpus learns the corpus's structure, not human behaviour.
The scripts that consume it say so in their own output.

Usage:
    python scripts/stealth/generate_traffic_dataset.py \\
        --num-sessions 2000 --output storage/data/behavioral/human_traffic.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT = Path("storage/data/behavioral/human_traffic.json")


def sample_session(
    rng: np.random.Generator, num_channels: int, min_len: int, max_len: int
) -> dict:
    """Sample one synthetic browsing session."""
    length = int(rng.integers(min_len, max_len + 1))
    delays = np.empty(length, dtype=np.float64)
    channels = np.empty(length, dtype=np.int64)

    # A session is a sequence of "platform episodes": the user stays on
    # one platform for a while, then switches.
    current_channel = int(rng.integers(0, num_channels))
    pos = 0
    while pos < length:
        episode = int(rng.integers(4, 25))
        episode = min(episode, length - pos)
        channels[pos : pos + episode] = current_channel

        # Within an episode, the gap between posts is a mixture of
        # regimes.
        n = episode
        active = rng.weibull(rng.uniform(0.7, 1.4), n) * rng.uniform(2.0, 7.0)
        burst = rng.random(n) < 0.12
        active[burst] = rng.uniform(0.4, 1.5, int(burst.sum()))
        pause = rng.random(n) < 0.06
        active[pause] = rng.uniform(30.0, 240.0, int(pause.sum()))
        away = rng.random(n) < 0.02
        active[away] = rng.uniform(600.0, 3600.0, int(away.sum()))

        delays[pos : pos + episode] = np.clip(active, 0.4, None)
        pos += episode
        if rng.random() < 0.6:
            current_channel = int(rng.integers(0, num_channels))

    return {
        "delays": [round(float(d), 2) for d in delays],
        "channels": [int(c) for c in channels],
        "time_of_day": int(rng.integers(0, 24)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sessions", type=int, default=2000)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--min-len", type=int, default=15)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sessions = [
        sample_session(rng, args.num_channels, args.min_len, args.max_len)
        for _ in range(args.num_sessions)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sessions))

    delays = np.concatenate([np.asarray(s["delays"]) for s in sessions])
    print(f"Wrote {len(sessions)} sessions to {out_path}")
    print(
        f"  delays: n={len(delays)} mean={delays.mean():.2f} "
        f"median={np.median(delays):.2f} p95={np.percentile(delays, 95):.2f} "
        f"max={delays.max():.2f}"
    )
    print(
        f"  sequence length: {min(len(s['delays']) for s in sessions)}"
        f"-{max(len(s['delays']) for s in sessions)}"
    )
    print("  NOTE: this corpus is synthetic, not recorded human browsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

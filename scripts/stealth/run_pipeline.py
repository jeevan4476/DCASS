"""Interactive entry point for the whole pipeline: gan/ -> scheduler/ -> path_agent/.

Run from the repo root:

    python run_pipeline.py

It asks for three things - number of packets, number of channels, and a
date - then generates that many GAN timestamps on that date and prints the
channel chosen for each. Congestion is currently simulated (no real
measurement source or send mechanism wired up yet - see
INTEGRATION_GUIDE.md section 8).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from path_agent.config import Config
from path_agent.path_selector import PathSelector
from scheduler import SimulatedCongestionProvider, plan_sends

REPO_ROOT = Path(__file__).resolve().parent
GAN_CKPT = REPO_ROOT / "gan" / "gan_ckpt.pt"


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def get_timestamps(n: int, date: str):
    """n randomized timestamps within the given day ('YYYY-MM-DD')."""
    start = pd.Timestamp(date)
    end = start + pd.Timedelta(days=1)

    if GAN_CKPT.exists():
        from gan.gan import generate_timestamps
        return generate_timestamps(n, start=start, end=end, ckpt=str(GAN_CKPT))

    print(f"(no GAN checkpoint at {GAN_CKPT}, using stand-in timestamps)")
    rng = np.random.default_rng(0)
    gaps = rng.uniform(0.5, 3.0, size=n)
    return np.cumsum(gaps)


def pick_checkpoint(num_paths: int, cfg: Config) -> str:
    """Use the checkpoint trained specifically for this many paths if we have
    one (path_agent/runs/dqn_n{num_paths}.pt); otherwise fall back to the
    default checkpoint - the network scores each path independently, so it
    still works at a different path count, just without N-specific tuning.
    """
    candidate = Path(cfg.checkpoint_path).parent / f"dqn_n{num_paths}.pt"
    return str(candidate) if candidate.exists() else cfg.checkpoint_path


def format_timestamp(ts) -> str:
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"t={ts:6.2f}s"


def main():
    print("=== Congestion-aware routing pipeline ===")
    n_packets = ask_int("How many packets to schedule", 15)
    num_paths = ask_int("How many channels/paths", 4)
    date = ask_str("Date to generate timestamps for (YYYY-MM-DD)", "2026-09-15")

    cfg = Config(num_paths=num_paths)
    ckpt = pick_checkpoint(num_paths, cfg)
    print(f"using checkpoint: {ckpt}\n")

    selector = PathSelector(checkpoint_path=ckpt, num_paths=num_paths)
    congestion = SimulatedCongestionProvider(config=cfg, seed=1)

    timestamps = get_timestamps(n_packets, date)

    def log_send(entry):
        print(f"{format_timestamp(entry.timestamp)}  -> channel {entry.channel}")

    schedule = plan_sends(timestamps, selector, congestion, send_fn=log_send)

    print(f"\nplanned {len(schedule)} sends across {num_paths} channels")


if __name__ == "__main__":
    main()

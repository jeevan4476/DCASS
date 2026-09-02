"""End-to-end pipeline: GAN timestamps -> scheduler -> chosen channel.

Uses the real GAN (gan/gan.py) if a trained checkpoint is present at
gan/gan_ckpt.pt, otherwise falls back to a lightweight stand-in so this
still runs standalone before the GAN is trained.

Congestion is supplied by SimulatedCongestionProvider (path_agent's
simulator) since a real congestion source and the actual send mechanism
aren't wired up yet - see INTEGRATION_GUIDE.md section 8 for how to swap
those in.
"""

from pathlib import Path

import numpy as np

from path_agent.config import Config
from path_agent.path_selector import PathSelector
from scheduler import SimulatedCongestionProvider, plan_sends

REPO_ROOT = Path(__file__).resolve().parent.parent
GAN_CKPT = REPO_ROOT / "gan" / "gan_ckpt.pt"


def get_timestamps(n=15, start="2026-09-15", end="2026-09-16", seed=0):
    """Real GAN-generated timestamps when a trained checkpoint exists,
    otherwise n evenly-randomized second offsets as a stand-in.
    """
    if GAN_CKPT.exists():
        from gan.gan import generate_timestamps
        return generate_timestamps(n, start=start, end=end, ckpt=str(GAN_CKPT), seed=seed)

    print(f"no GAN checkpoint at {GAN_CKPT}, using stand-in timestamps")
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.5, 3.0, size=n)
    return np.cumsum(gaps)


def format_timestamp(ts) -> str:
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"t={ts:6.2f}s"


def main():
    cfg = Config()
    selector = PathSelector(checkpoint_path=cfg.checkpoint_path)
    congestion = SimulatedCongestionProvider(config=cfg, seed=1)

    timestamps = get_timestamps()

    def log_send(entry):
        print(f"{format_timestamp(entry.timestamp)}  -> channel {entry.channel}")

    schedule = plan_sends(timestamps, selector, congestion, send_fn=log_send)

    print(f"\nplanned {len(schedule)} sends across {cfg.num_paths} channels")


if __name__ == "__main__":
    main()

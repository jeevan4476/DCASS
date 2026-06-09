#!/usr/bin/env python3
# scripts/docker_orchestrate.py
"""
Docker Pipeline Orchestrator for DCASS.

Runs the full pipeline using docker compose:

  1. Generate synthetic traffic data   (training profile)
  2. Train GAN stealth scheduler       (training profile)  [optional]
  3. Train RL stealth policy           (training profile)  [optional]
  4. Send sequence (auto fallback)     (default profile)
  5. Receiver picks up packets         (default profile)

Usage:
    python scripts/docker_orchestrate.py                  # send only (auto fallback)
    python scripts/docker_orchestrate.py --full-pipeline  # gen + train + send
    python scripts/docker_orchestrate.py --send-only      # explicit send-only
    python scripts/docker_orchestrate.py --gen-data       # generate traffic data only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run(cmd: str, check: bool = True) -> int:
    """Run a shell command, streaming output."""
    print(f"\n{'─' * 60}")
    print(f"  ▶  {cmd}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if check and result.returncode != 0:
        print(f"\n✗ Command failed (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)
    return result.returncode


def build():
    """Build Docker images."""
    run("docker compose build")


def generate_traffic_data():
    """Step 0: Generate synthetic human traffic data."""
    run("docker compose --profile training run --rm dcass-gen-traffic")


def train_gan(epochs: int = 50):
    """Step 1: Train GAN stealth scheduler."""
    run(f"GAN_EPOCHS={epochs} docker compose --profile training run --rm dcass-train-gan")


def train_rl(episodes: int = 1000):
    """Step 2: Train RL stealth policy."""
    run(f"RL_EPISODES={episodes} docker compose --profile training run --rm dcass-train-rl")


def send_receive(mode: str = "auto", seq_length: int = 20):
    """Step 3: Run sender + receiver (auto fallback)."""
    run(
        f"DCASS_MODE={mode} DCASS_SEQ_LENGTH={seq_length} "
        f"docker compose up --abort-on-container-exit dcass-sender dcass-receiver"
    )


def cleanup():
    """Stop and remove all containers."""
    run("docker compose down --remove-orphans", check=False)


def main():
    parser = argparse.ArgumentParser(
        description="DCASS Docker Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Run everything: gen-data → train-gan → train-rl → send",
    )
    parser.add_argument(
        "--gen-data", action="store_true",
        help="Generate synthetic traffic data only",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Run data generation + GAN + RL training (no send)",
    )
    parser.add_argument(
        "--send-only", action="store_true",
        help="Run sender + receiver only (auto fallback)",
    )
    parser.add_argument(
        "--mode", type=str, default="auto",
        choices=["auto", "rl", "gan", "static"],
        help="Sender scheduling mode (default: auto)",
    )
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--gan-epochs", type=int, default=50)
    parser.add_argument("--rl-episodes", type=int, default=1000)
    parser.add_argument("--build", action="store_true", help="Force rebuild images")
    parser.add_argument("--cleanup", action="store_true", help="Stop and remove containers")

    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    if args.build:
        build()

    if args.full_pipeline:
        build()
        generate_traffic_data()
        train_gan(args.gan_epochs)
        train_rl(args.rl_episodes)
        send_receive(args.mode, args.seq_length)
    elif args.gen_data:
        generate_traffic_data()
    elif args.train:
        generate_traffic_data()
        train_gan(args.gan_epochs)
        train_rl(args.rl_episodes)
    elif args.send_only or not any([args.full_pipeline, args.gen_data, args.train]):
        # Default: just send (auto fallback)
        send_receive(args.mode, args.seq_length)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# scripts/runtime/run_pipeline.py
"""
Phase F — Unified pipeline CLI.

Runs one end-to-end round-trip through the full DCASS stack:

    encode -> schedule (GAN/RL/static) -> transmit through shared channel
    -> reassemble -> decode -> report whether decoded == original.

Thin wrapper around `src.stealth.pipeline.run_pipeline`. See
docs/GAN_RL_INTEGRATION_PLAN.md Phase F.3 for the design rationale.

Usage:
    python scripts/runtime/run_pipeline.py \\
        --message "Meet at the cafe at noon" \\
        --sender alice --recipient bob \\
        --mode auto --speed-multiplier 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DCASS end-to-end pipeline round-trip (Phase F).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--message", required=True, help="Secret message to send")
    parser.add_argument("--sender", required=True, help="Sender username")
    parser.add_argument("--recipient", required=True, help="Recipient username")
    parser.add_argument(
        "--mode",
        choices=["static", "gan", "rl", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=100.0,
        help="Divide real delays by this. 1.0 = real wall clock; 100 = fast test (default)",
    )
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--base-delay", type=float, default=3.0)
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=None,
        help="Override the default `storage/shared_channel/` root",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep the session's packets+manifest after the run finishes",
    )
    args = parser.parse_args()

    result = run_pipeline(
        message=args.message,
        sender=args.sender,
        recipient=args.recipient,
        mode=args.mode,
        speed_multiplier=args.speed_multiplier,
        shared_dir=args.shared_dir,
        num_channels=args.num_channels,
        base_delay=args.base_delay,
        cleanup=not args.no_cleanup,
    )

    print(result.summary())

    # Exit non-zero on round-trip failure so CI / shell scripts can gate on it.
    sys.exit(0 if result.decoded_ok else 1)


if __name__ == "__main__":
    main()

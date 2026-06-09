#!/usr/bin/env python3
# scripts/run_sender.py
"""
Alice (Sender) Script for DCASS Dockerized Simulation.

Uses the StealthScheduler to produce a timed schedule and writes
media-packet metadata to the shared channel directory that Bob
(receiver) watches.

Modes
-----
static – Handcrafted NoiseController (no models needed) **[default — review demo]**
gan    – GAN only, static fallback if checkpoint is missing  [future review]
rl     – RL only, static fallback if checkpoint is missing   [future review]
auto   – Try RL → GAN → static  (cascade)                    [future review]
train  – Train the RL agent in-container                     [future review]
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Optional
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.stealth_scheduler import StealthScheduler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("dcass.sender")


# ═══════════════════════════════════════════════════════════════════════════
# Packet writer
# ═══════════════════════════════════════════════════════════════════════════

class PacketWriter:
    """Writes packet metadata JSON files to the shared channel directory."""

    def __init__(self, shared_dir: Path):
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.transmission_count = 0

    def write_packet(
        self,
        media_id: str,
        channel_id: int,
        sequence_number: int,
        delay: float,
        mode_used: str,
    ) -> Path:
        """
        Write a single packet metadata file.

        Returns the path of the written file.
        """
        metadata = {
            "media_id": media_id,
            "channel_id": channel_id,
            "sequence_number": sequence_number,
            "delay_seconds": round(delay, 3),
            "timestamp": time.time(),
            "mode_used": mode_used,
        }

        filename = f"{media_id}_{channel_id}_{sequence_number:04d}.json"
        file_path = self.shared_dir / filename

        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.transmission_count += 1
        log.info(
            "TX #%d  seq=%d  media=%s  ch=%d  delay=%.1fs  [%s]",
            self.transmission_count,
            sequence_number,
            media_id,
            channel_id,
            delay,
            mode_used,
        )
        return file_path

    def write_manifest(self, schedule: dict, message: str):
        """Write a session manifest for debugging / auditing."""
        manifest = {
            "message": message,
            "mode_requested": schedule.get("mode_requested", "unknown"),
            "mode_used": schedule.get("mode_used", "unknown"),
            "total_items": len(schedule["items"]),
            "total_delay_seconds": round(sum(schedule["delays"]), 2),
            "timestamp": time.time(),
        }
        path = self.shared_dir / "_manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Manifest written → %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Core sender logic
# ═══════════════════════════════════════════════════════════════════════════

def resolve_mode_cascade(
    scheduler: StealthScheduler,
    media_ids: List[str],
    requested_mode: str,
    base_delay: float,
    gan_checkpoint: Optional[Path],
    rl_checkpoint: Optional[Path],
) -> dict:
    """
    Try the requested mode; if it falls back to static, try the next
    dynamic mode before giving up.

    Cascade order for 'auto':  rl → gan → static
    For explicit modes (rl / gan) the StealthScheduler already handles
    fallback to static internally.
    """
    if requested_mode == "auto":
        # --- Try RL first ------------------------------------------------
        log.info("Auto mode: attempting RL scheduling …")
        schedule = scheduler.schedule(
            media_ids,
            mode="rl",
            base_delay=base_delay,
            rl_checkpoint=rl_checkpoint,
        )
        if schedule["mode_used"] == "rl":
            schedule["mode_requested"] = "auto→rl"
            return schedule

        # --- RL unavailable, try GAN -------------------------------------
        log.info("Auto mode: RL unavailable, attempting GAN scheduling …")
        schedule = scheduler.schedule(
            media_ids,
            mode="gan",
            base_delay=base_delay,
            gan_checkpoint=gan_checkpoint,
        )
        if schedule["mode_used"] == "gan":
            schedule["mode_requested"] = "auto→gan"
            return schedule

        # --- Both unavailable, static fallback ---------------------------
        log.warning(
            "Auto mode: no trained checkpoints found — "
            "falling back to STATIC (NoiseController)."
        )
        schedule["mode_requested"] = "auto→static"
        return schedule

    # Explicit mode — StealthScheduler handles its own fallback
    schedule = scheduler.schedule(
        media_ids,
        mode=requested_mode,
        base_delay=base_delay,
        gan_checkpoint=gan_checkpoint,
        rl_checkpoint=rl_checkpoint,
    )
    schedule["mode_requested"] = requested_mode
    if schedule["mode_used"] != requested_mode:
        log.warning(
            "Requested mode '%s' unavailable — fell back to '%s'.",
            requested_mode,
            schedule["mode_used"],
        )
    return schedule


def send_sequence(
    scheduler: StealthScheduler,
    writer: PacketWriter,
    media_ids: List[str],
    mode: str,
    base_delay: float,
    gan_checkpoint: Optional[Path],
    rl_checkpoint: Optional[Path],
    dry_run: bool = False,
):
    """Schedule + transmit a media sequence through the shared channel."""

    # 1. Produce schedule (dynamic → static cascade)
    schedule = resolve_mode_cascade(
        scheduler, media_ids, mode, base_delay,
        gan_checkpoint, rl_checkpoint,
    )

    items = schedule["items"]
    delays = schedule["delays"]
    channels = schedule["channels"]
    mode_used = schedule["mode_used"]

    log.info("=" * 60)
    log.info("SCHEDULE READY  |  mode_used=%s  |  items=%d", mode_used, len(items))
    log.info("  Total estimated time: %.1fs", sum(delays))
    log.info("=" * 60)

    # 2. Write manifest
    writer.write_manifest(schedule, message="(sequence)")

    # 3. Transmit packets with the scheduled delays
    for idx, (media_id, delay, channel) in enumerate(zip(items, delays, channels)):
        if media_id is None:
            # Noise-injected idle gap — just wait
            if not dry_run:
                time.sleep(delay)
            continue

        writer.write_packet(
            media_id=media_id,
            channel_id=channel,
            sequence_number=idx,
            delay=delay,
            mode_used=mode_used,
        )

        if not dry_run and delay > 0:
            log.debug("Sleeping %.1fs before next packet …", delay)
            time.sleep(delay)

    log.info(
        "Sequence complete!  %d packets transmitted  |  mode=%s",
        writer.transmission_count,
        mode_used,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Training (deferred — kept for later use)
# ═══════════════════════════════════════════════════════════════════════════

def train_agent_in_container(episodes: int):
    """Train the RL agent inside the Docker container. For later use."""
    from src.stealth.rl.agent import PPOAgent, PPOConfig
    from src.stealth.rl.environment import StealthEnvironment
    from src.analysis.adversarial.warden import DeepPacketInspectionWarden

    log.info("Training RL agent for %d episodes …", episodes)

    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.eval()

    env = StealthEnvironment(num_channels=3, warden=warden, lambda_stealth=50.0)
    config = PPOConfig(state_dim=env.state_dim, device="cpu")
    agent = PPOAgent(env, config)

    rng = np.random.default_rng(42)

    def media_gen():
        n = rng.integers(10, 30)
        return [f"media_{i:03d}" for i in range(n)]

    agent.train(num_episodes=episodes, media_sequence_generator=media_gen, log_interval=10)

    ckpt_dir = Path("/app/checkpoints/rl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "ppo_agent_final.pt"
    agent.save(save_path)
    log.info("Training complete. Agent saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DCASS Sender (Alice) — stealth sequence transmitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shared-dir", type=str, default="/app/shared_channel",
        help="Shared directory for transmissions (mounted Docker volume)",
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["auto", "rl", "gan", "static", "train"],
        default="static",
        help=(
            "Scheduling mode. Default 'static' uses the handcrafted "
            "NoiseController (no model checkpoints required). "
            "'auto' tries rl→gan→static cascade. 'train' trains the RL agent."
        ),
    )
    parser.add_argument(
        "--base-delay", type=float, default=3.0,
        help="Base inter-item delay in seconds (for static mode)",
    )
    parser.add_argument(
        "--num-channels", type=int, default=3,
        help="Number of distribution channels",
    )
    parser.add_argument(
        "--profile", type=str, default="casual",
        help="Activity profile for static noise (casual / stealth / burst)",
    )
    parser.add_argument(
        "--gan-checkpoint", type=str, default=None,
        help="Path to trained GAN checkpoint (.pt)",
    )
    parser.add_argument(
        "--rl-checkpoint", type=str, default=None,
        help="Path to trained RL (PPO) checkpoint (.pt)",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of training episodes (only for --mode train)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=20,
        help="Fallback number of media items when --no-encode is set",
    )
    parser.add_argument(
        "--message", type=str, default="Meet at the cafe at noon",
        help="Secret message to encode into a media sequence",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Skip SemanticEncoder and transmit dummy media_NNN IDs "
             "(sniff test only; Bob won't be able to decode)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip real delays (write packets immediately)",
    )

    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DCASS Sender (Alice)")
    print("  Mode:", args.mode)
    print("=" * 60)

    # ── Training mode ───────────────────────────────────────────────────
    if args.mode == "train":
        train_agent_in_container(episodes=args.episodes)
        print("\n[Sender] Done!")
        return

    # ── Transmission mode ───────────────────────────────────────────────
    scheduler = StealthScheduler(
        num_channels=args.num_channels,
        device="cpu",
        profile=args.profile,
    )
    writer = PacketWriter(shared_dir=Path(args.shared_dir))

    log.info('Secret message: "%s"', args.message)

    if args.no_encode:
        media_ids = [f"media_{i:03d}" for i in range(args.sequence_length)]
        log.warning("--no-encode: transmitting %d dummy IDs (Bob cannot decode)",
                    len(media_ids))
    else:
        from src.engine.encoder import SemanticEncoder
        log.info("Loading SemanticEncoder and indices...")
        encoder = SemanticEncoder(expand_synonyms=True)
        encoder.load()
        log.info("Encoding message into media sequence...")
        encode_result = encoder.encode(args.message)
        media_ids = encode_result.media_ids
        log.info("Encoder produced %d media items: %s",
                 len(media_ids), encode_result.modality_breakdown)

    log.info("Media sequence length: %d", len(media_ids))

    gan_ckpt = Path(args.gan_checkpoint) if args.gan_checkpoint else None
    rl_ckpt = Path(args.rl_checkpoint) if args.rl_checkpoint else None

    send_sequence(
        scheduler=scheduler,
        writer=writer,
        media_ids=media_ids,
        mode=args.mode,
        base_delay=args.base_delay,
        gan_checkpoint=gan_ckpt,
        rl_checkpoint=rl_ckpt,
        dry_run=args.dry_run,
    )

    print("\n[Sender] Done!")


if __name__ == "__main__":
    main()

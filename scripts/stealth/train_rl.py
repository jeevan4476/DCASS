#!/usr/bin/env python3
# scripts/stealth/train_rl.py
"""
Phase B — RL training against the frozen GAN Warden.

Loads the Warden trained in Phase A (scripts/stealth/train_gan.py), freezes
it, and trains a PPO agent (src.stealth.rl.agent.PPOAgent) whose reward
penalizes that Warden's detection probability. This is the piece that was
missing: the pre-existing in-sender training path
(scripts/runtime/run_sender.py:train_agent_in_container) built a
RANDOM-weight Warden, so its reward signal carried no real stealth
information. --warden-checkpoint here is REQUIRED and actually used.

See docs/GAN_RL_INTEGRATION_PLAN.md (Phase B) for the full rationale,
including the lambda-stealth curriculum and the delay-semantics fix this
script assumes (src/stealth/rl/environment.py).

Usage:
    python scripts/stealth/train_rl.py \\
        --warden-checkpoint storage/models/gan_generator.pt \\
        --episodes 500 --lambda-stealth 50 \\
        --out storage/models/rl_agent.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.stealth.rl.environment import StealthEnvironment


def load_frozen_warden(
    checkpoint_path: Path, num_channels: int, device: str
) -> DeepPacketInspectionWarden:
    """Load the Phase-A Warden and freeze it for use as a reward model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    hidden_dim = getattr(config, "hidden_dim", 256) if config is not None and not isinstance(config, dict) else (config or {}).get("hidden_dim", 256)

    warden = DeepPacketInspectionWarden(num_channels=num_channels, hidden_dim=hidden_dim)
    warden.load_state_dict(ckpt["warden_state"])
    warden.to(device)
    warden.eval()
    for p in warden.parameters():
        p.requires_grad_(False)
    return warden


def make_lambda_schedule(target: float, warmup_fraction: float, num_episodes: int):
    """
    Linear ramp from lambda=target*0.2 up to `target` over the first
    `warmup_fraction` of episodes, then hold at `target`.

    See Risk R2 in docs/GAN_RL_INTEGRATION_PLAN.md: advantage normalization
    partially absorbs this, but it still changes the reward's gradient
    direction early in training, preventing the stealth penalty from
    crushing a policy that hasn't learned to transmit at all yet.
    """
    warmup_episodes = max(1, int(num_episodes * warmup_fraction))
    start = target * 0.2

    def schedule(episode: int) -> float:
        if episode >= warmup_episodes:
            return target
        frac = episode / warmup_episodes
        return start + frac * (target - start)

    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO agent against the frozen GAN Warden (Phase B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--warden-checkpoint",
        type=Path,
        required=True,
        help="Path to the Phase-A GAN checkpoint (must contain 'warden_state')",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--lambda-stealth", type=float, default=50.0)
    parser.add_argument(
        "--lambda-warmup-fraction",
        type=float,
        default=0.2,
        help="Fraction of episodes over which lambda_stealth ramps up from 20%% to the target",
    )
    parser.add_argument("--min-seq-len", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=60)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "storage" / "models" / "rl_agent.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    if not args.warden_checkpoint.exists():
        print(f"ERROR: --warden-checkpoint not found at {args.warden_checkpoint}")
        print("Run scripts/stealth/train_gan.py first (Phase A).")
        sys.exit(1)

    print("=" * 60)
    print("  DCASS RL Training (Phase B)")
    print(f"  Warden checkpoint: {args.warden_checkpoint}")
    print(f"  Episodes: {args.episodes}  lambda_stealth target: {args.lambda_stealth}")
    print("=" * 60)

    warden = load_frozen_warden(args.warden_checkpoint, args.num_channels, args.device)
    print(
        f"✓ Loaded frozen Warden ({sum(p.numel() for p in warden.parameters()):,} params, "
        "requires_grad=False)"
    )

    env = StealthEnvironment(
        num_channels=args.num_channels,
        warden=warden,
        lambda_stealth=args.lambda_stealth,
    )
    config = PPOConfig(state_dim=env.state_dim, device=args.device)
    agent = PPOAgent(env, config)

    lambda_schedule = make_lambda_schedule(
        target=args.lambda_stealth,
        warmup_fraction=args.lambda_warmup_fraction,
        num_episodes=args.episodes,
    )

    rng = np.random.default_rng(args.seed)

    def media_sequence_generator():
        n = rng.integers(args.min_seq_len, args.max_seq_len + 1)
        return [f"media_{i:03d}" for i in range(n)]

    print(f"Training PPO agent for {args.episodes} episodes " f"(seq_len in [{args.min_seq_len}, {args.max_seq_len}])...")

    for episode in range(args.episodes):
        env.lambda_stealth = lambda_schedule(episode)

        agent.collect_rollout(media_sequence_generator())
        metrics = agent.update()

        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(agent.episode_rewards[-args.log_interval :])
            avg_warden = np.mean(agent.warden_scores[-args.log_interval :])
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"lambda={env.lambda_stealth:.1f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Warden Score: {avg_warden:.3f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out)
    print(f"\n✓ RL agent saved to {args.out}")
    print("  Next: run the scheduler end-to-end (see docs/GAN_RL_INTEGRATION_PLAN.md)")


if __name__ == "__main__":
    main()

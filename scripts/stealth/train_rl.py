#!/usr/bin/env python3
"""
DCASS PPO Reinforcement Learning Training Script.

Trains an Actor-Critic PPO agent inside StealthEnvironment against the
pre-trained Warden discriminator, optimizing closed-loop channel selection
and dynamic inter-packet delays.

Usage:
    .venv/bin/python scripts/stealth/train_rl.py --episodes 500
"""

import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.stealth.rl.environment import StealthEnvironment
from src.analysis.adversarial.warden import DeepPacketInspectionWarden


def main():
    parser = argparse.ArgumentParser(description="DCASS PPO Reinforcement Learning Training")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of transmission channels")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-stealth", type=float, default=100.0, help="Stealth penalty weight")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging interval in episodes")
    args = parser.parse_args()

    print("=" * 80)
    print(" DCASS PPO REINFORCEMENT LEARNING CLOSED-LOOP TRAINING")
    print("=" * 80)
    print(f"• Hardware Device:      {args.device.upper()}")
    if args.device == "cuda":
        print(f"• Active GPU:           {torch.cuda.get_device_name(0)}")
    print(f"• Target Episodes:      {args.episodes}")
    print(f"• Number of Channels:   {args.num_channels}")
    print(f"• Stealth Lambda:       {args.lambda_stealth}")
    print(f"• Learning Rate:        {args.lr}")

    models_dir = PROJECT_ROOT / "storage" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rl_dir = models_dir / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Warden Discriminator
    warden = DeepPacketInspectionWarden(num_channels=args.num_channels, hidden_dim=256)
    warden_ckpt = models_dir / "warden.pt"
    gan_ckpt = models_dir / "gan_generator.pt"

    if warden_ckpt.exists():
        print(f"• Loading pre-trained Warden from: {warden_ckpt.name}")
        ckpt = torch.load(warden_ckpt, map_location=args.device, weights_only=False)
        warden.load_state_dict(ckpt.get("warden_state", ckpt))
    elif gan_ckpt.exists():
        print(f"• Loading pre-trained Warden from: {gan_ckpt.name}")
        ckpt = torch.load(gan_ckpt, map_location=args.device, weights_only=False)
        if "warden_state" in ckpt:
            warden.load_state_dict(ckpt["warden_state"])
    else:
        print("• Initializing fresh Warden discriminator")

    warden.to(args.device)
    warden.eval()

    # 2. Initialize Stealth Environment
    env = StealthEnvironment(
        num_channels=args.num_channels,
        warden=warden,
        lambda_stealth=args.lambda_stealth,
        max_sequence_length=100
    )

    # 3. Initialize PPO Agent
    config = PPOConfig(
        state_dim=env.state_dim,
        hidden_dim=256,
        learning_rate=args.lr,
        gamma=0.99,
        epsilon_clip=0.2,
        batch_size=args.batch_size,
        num_epochs=4,
        device=args.device
    )
    agent = PPOAgent(env=env, config=config)
    param_count = sum(p.numel() for p in agent.actor_critic.parameters())
    print(f"• Actor-Critic Parameters: {param_count:,}")
    print("-" * 80)

    # 4. Media Sequence Generator
    rng = np.random.default_rng(42)

    def media_generator():
        seq_len = rng.integers(10, 40)
        return [f"media_item_{i:04d}" for i in range(seq_len)]

    # 5. Training Loop
    t_start = time.time()
    print("Starting PPO Policy Optimization Loop...")

    agent.train(
        num_episodes=args.episodes,
        media_sequence_generator=media_generator,
        log_interval=args.log_interval
    )

    elapsed = time.time() - t_start
    print("-" * 80)
    print(f"✅ PPO Training completed in {elapsed:.2f}s ({args.episodes / elapsed:.1f} episodes/sec)")

    # 6. Save Checkpoints
    primary_save = models_dir / "rl_agent.pt"
    backup_save = rl_dir / "ppo_agent_final.pt"

    agent.save(primary_save)
    agent.save(backup_save)
    print(f"• Master RL Agent Checkpoint: {primary_save}")
    print(f"• Backup Checkpoint:          {backup_save}")
    print("=" * 80)


if __name__ == "__main__":
    main()

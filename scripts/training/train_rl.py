#!/usr/bin/env python3
"""
Train the RL Stealth Policy Agent (PPO).

Trains the PPOAgent inside the StealthEnvironment, using a pre-trained
(or fresh) Warden as the adversarial detector.

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py --episodes 2000 --device cuda
    python scripts/train_rl.py --warden-checkpoint models/gan/final.pt
"""

import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.stealth.rl.environment import StealthEnvironment
from src.analysis.adversarial.warden import DeepPacketInspectionWarden

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train RL stealth policy agent")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lambda-stealth", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models" / "rl")
    parser.add_argument("--warden-checkpoint", type=Path, default=None,
                        help="Path to a trained Warden .pt file (from GAN training)")
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- warden ----------------------------------------------------------
    warden = DeepPacketInspectionWarden(num_channels=args.num_channels)
    if args.warden_checkpoint and args.warden_checkpoint.exists():
        print(f"Loading trained Warden from {args.warden_checkpoint}")
        ckpt = torch.load(args.warden_checkpoint, map_location=args.device)
        warden.load_state_dict(ckpt["warden_state"])
    else:
        print("Using fresh (untrained) Warden")
    warden.eval()

    # --- environment -----------------------------------------------------
    env = StealthEnvironment(
        num_channels=args.num_channels,
        warden=warden,
        lambda_stealth=args.lambda_stealth,
    )

    # --- agent -----------------------------------------------------------
    config = PPOConfig(state_dim=env.state_dim, device=args.device, learning_rate=args.lr)
    agent = PPOAgent(env, config)
    print(f"Actor-Critic params: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

    # --- media sequence generator ----------------------------------------
    rng = np.random.default_rng(42)

    def media_gen():
        length = rng.integers(10, 30)
        return [f"media_{i:04d}" for i in range(length)]

    # --- train -----------------------------------------------------------
    agent.train(num_episodes=args.episodes, media_sequence_generator=media_gen,
                log_interval=args.log_interval)

    # --- save ------------------------------------------------------------
    save_path = args.checkpoint_dir / "ppo_agent_final.pt"
    agent.save(save_path)
    print(f"\nTraining complete. Agent saved to {save_path}")


if __name__ == "__main__":
    main()

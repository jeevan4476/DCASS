#!/usr/bin/env python3
# scripts/run_sender.py
"""
Alice (Sender) Script for DCASS Dockerized Simulation.

Uses the RL agent to schedule transmissions and sends media files to
the shared channel monitored by Bob (receiver).
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stealth.rl.environment import StealthEnvironment
from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.analysis.adversarial.warden import DeepPacketInspectionWarden


class SimulatedSender:
    """
    Simulated sender for Docker environment.

    Uses RL agent to schedule transmissions and writes packet metadata
    to shared directory for receiver to pick up.
    """

    def __init__(
        self,
        shared_dir: Path,
        num_channels: int = 3,
        use_rl: bool = True,
        agent_checkpoint: Optional[Path] = None
    ):
        """
        Initialize sender.

        Args:
            shared_dir: Shared directory for transmitting packets
            num_channels: Number of distribution channels
            use_rl: Whether to use RL agent (vs. random scheduling)
            agent_checkpoint: Path to pre-trained agent checkpoint
        """
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        self.num_channels = num_channels
        self.use_rl = use_rl

        # Initialize Warden
        print("[Sender] Loading Warden...")
        self.warden = DeepPacketInspectionWarden(num_channels=num_channels)
        self.warden.eval()

        # Initialize environment
        print("[Sender] Creating RL environment...")
        self.env = StealthEnvironment(
            num_channels=num_channels,
            warden=self.warden,
            lambda_stealth=50.0
        )

        # Initialize agent (if using RL)
        self.agent: Optional[PPOAgent] = None
        if use_rl:
            print("[Sender] Initializing PPO agent...")
            config = PPOConfig(
                state_dim=self.env.state_dim,
                device="cpu"
            )
            self.agent = PPOAgent(self.env, config)

            # Load checkpoint if provided
            if agent_checkpoint is not None and agent_checkpoint.exists():
                print(f"[Sender] Loading agent from {agent_checkpoint}")
                self.agent.load(agent_checkpoint)

        self.transmission_count = 0

    def send_packet(
        self,
        media_id: str,
        channel_id: int,
        sequence_number: int
    ):
        """
        Send a packet by writing metadata to shared directory.

        Args:
            media_id: Media item identifier
            channel_id: Channel to send on
            sequence_number: Sequence number for reassembly
        """
        # Create packet metadata
        metadata = {
            "media_id": media_id,
            "channel_id": channel_id,
            "sequence_number": sequence_number,
            "timestamp": time.time()
        }

        # Write to shared directory
        filename = f"{media_id}_{channel_id}_{sequence_number:04d}.json"
        file_path = self.shared_dir / filename

        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.transmission_count += 1

        print(
            f"[Sender] Sent packet {sequence_number}: {media_id} "
            f"(channel={channel_id}) -> {filename}"
        )

    def send_sequence_rl(self, media_sequence: List[str]):
        """
        Send media sequence using RL agent scheduling.

        Args:
            media_sequence: List of media IDs to transmit
        """
        print(f"[Sender] Sending {len(media_sequence)} items using RL agent...")

        # Reset environment
        state = self.env.reset(media_sequence)
        done = False
        step = 0

        while not done:
            # Select action using agent
            if self.agent is not None:
                action, _, _ = self.agent.select_action(state)
            else:
                # Fallback to random action
                action = {
                    "delay": np.random.uniform(5, 15),
                    "channel": np.random.randint(0, self.num_channels)
                }

            # Take action in environment
            next_state, reward, done, info = self.env.step(action)

            # Actually send the packet (write to shared dir)
            if info.get("queue_remaining", 0) < len(media_sequence):
                # A packet was sent
                sent_index = len(media_sequence) - info["queue_remaining"] - 1
                media_id = media_sequence[sent_index]

                self.send_packet(
                    media_id=media_id,
                    channel_id=action["channel"],
                    sequence_number=sent_index
                )

                # Apply delay
                print(f"[Sender] Waiting {action['delay']:.1f}s before next transmission...")
                time.sleep(action['delay'])

            state = next_state
            step += 1

        # Report final metrics
        warden_score = self.env.get_warden_score()
        print(f"[Sender] Sequence complete!")
        print(f"  Total transmissions: {self.transmission_count}")
        print(f"  Total time: {self.env.current_time:.1f}s")
        print(f"  Warden score: {warden_score:.3f}")

    def send_sequence_random(self, media_sequence: List[str]):
        """
        Send media sequence with random scheduling (baseline).

        Args:
            media_sequence: List of media IDs to transmit
        """
        print(f"[Sender] Sending {len(media_sequence)} items with random scheduling...")

        for idx, media_id in enumerate(media_sequence):
            # Random delay and channel
            delay = np.random.uniform(5, 15)
            channel = np.random.randint(0, self.num_channels)

            # Send packet
            self.send_packet(
                media_id=media_id,
                channel_id=channel,
                sequence_number=idx
            )

            # Wait
            time.sleep(delay)

        print(f"[Sender] Sequence complete! ({self.transmission_count} packets)")

    def train_agent(self, num_episodes: int = 100):
        """
        Train the RL agent in the simulation environment.

        Args:
            num_episodes: Number of training episodes
        """
        if self.agent is None:
            print("[Sender] Error: RL agent not initialized")
            return

        print(f"[Sender] Training RL agent for {num_episodes} episodes...")

        # Define media sequence generator
        def generate_sequence():
            length = np.random.randint(10, 30)
            return [f"media_{i:03d}" for i in range(length)]

        # Train
        self.agent.train(
            num_episodes=num_episodes,
            media_sequence_generator=generate_sequence,
            log_interval=10
        )

        # Save trained agent
        checkpoint_dir = Path("checkpoints/rl")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "trained_agent.pt"

        self.agent.save(checkpoint_path)

        print(f"[Sender] Training complete! Agent saved to {checkpoint_path}")


def main():
    """Main entry point for sender."""
    parser = argparse.ArgumentParser(
        description="DCASS Sender (Alice)"
    )
    parser.add_argument(
        "--shared-dir",
        type=str,
        default="/app/shared_channel",
        help="Shared directory for transmissions"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rl", "random", "train"],
        default="rl",
        help="Transmission mode: rl (use agent), random (baseline), train (train agent)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to agent checkpoint"
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Meet at the cafe at noon",
        help="Secret message to encode and send"
    )

    args = parser.parse_args()

    # Create sender
    sender = SimulatedSender(
        shared_dir=Path(args.shared_dir),
        num_channels=3,
        use_rl=(args.mode in ["rl", "train"]),
        agent_checkpoint=Path(args.checkpoint) if args.checkpoint else None
    )

    print("=" * 60)
    print("DCASS Sender (Alice)")
    print("=" * 60)

    if args.mode == "train":
        # Training mode
        sender.train_agent(num_episodes=args.episodes)

    else:
        # Transmission mode
        # In a real scenario, we would encode the message first
        # For simulation, use a dummy sequence
        media_sequence = [f"media_{i:03d}" for i in range(20)]

        print(f"Secret message: \"{args.message}\"")
        print(f"Media sequence length: {len(media_sequence)}")

        if args.mode == "rl":
            sender.send_sequence_rl(media_sequence)
        else:
            sender.send_sequence_random(media_sequence)

    print("\n[Sender] Done!")


if __name__ == "__main__":
    main()

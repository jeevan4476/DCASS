# src/stealth/rl/environment.py
"""
RL Environment for DCASS Stealth Optimization.

This module implements a Gym-style environment for training an RL agent to
optimize transmission scheduling while evading the Warden.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from collections import deque

from ...analysis.adversarial.warden import DeepPacketInspectionWarden


@dataclass
class ChannelState:
    """State of a single distribution channel."""

    channel_id: int
    rate_limit: float  # Max transmissions per minute
    last_transmission_time: float  # Seconds since start
    transmission_count: int  # Total transmissions sent

    def can_send(self, current_time: float, window: float = 60.0) -> bool:
        """Check if channel can send based on rate limit."""
        # Simple rate limiting: check if enough time has passed
        time_since_last = current_time - self.last_transmission_time
        return time_since_last >= (60.0 / self.rate_limit)


@dataclass
class TransmissionRecord:
    """Record of a single transmission."""

    media_id: str
    channel_id: int
    timestamp: float
    delay_from_previous: float


class StealthEnvironment:
    """
    RL Environment for steganography transmission scheduling.

    The agent must schedule transmissions of media items across multiple
    channels to maximize throughput while minimizing detection by the Warden.

    State Space:
        - Queue size (remaining media items)
        - Current time of day (hour)
        - Channel states (last transmission time, rate limits)
        - Recent transmission history (for temporal patterns)

    Action Space:
        - Delay duration (continuous, in seconds)
        - Target channel (discrete)
        - Which media item to send next (typically FIFO from queue)

    Reward:
        R = Throughput (items/minute) - λ * Warden_Score

        Where:
        - Throughput encourages faster transmission
        - Warden_Score penalizes suspicious patterns
        - λ controls stealth vs. speed trade-off

    Args:
        num_channels: Number of distribution channels available
        warden: Pre-trained Warden for evaluation
        max_sequence_length: Maximum media sequence length
        lambda_stealth: Stealth penalty coefficient (default: 100.0)
        channel_rate_limits: Rate limits per channel (items/minute)
        max_episode_time: Maximum episode duration in seconds

    Example:
        >>> warden = DeepPacketInspectionWarden(num_channels=3)
        >>> env = StealthEnvironment(num_channels=3, warden=warden)
        >>> state = env.reset(media_sequence=["img1", "img2", "img3"])
        >>> action = {"delay": 5.0, "channel": 0}
        >>> next_state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        num_channels: int = 3,
        warden: Optional[DeepPacketInspectionWarden] = None,
        max_sequence_length: int = 100,
        lambda_stealth: float = 100.0,
        channel_rate_limits: Optional[list[float]] = None,
        max_episode_time: float = 3600.0,  # 1 hour max
        warden_window_size: int = 20,  # Number of recent transmissions to evaluate
    ):
        self.num_channels = num_channels
        self.max_sequence_length = max_sequence_length
        self.lambda_stealth = lambda_stealth
        self.max_episode_time = max_episode_time
        self.warden_window_size = warden_window_size

        # Initialize Warden
        if warden is None:
            self.warden = DeepPacketInspectionWarden(num_channels=num_channels)
            self.warden.eval()  # Freeze in eval mode
        else:
            self.warden = warden
            self.warden.eval()

        # Channel configuration
        if channel_rate_limits is None:
            # Default: channels have different rate limits
            channel_rate_limits = [
                10.0,  # Channel 0: 10 items/minute
                5.0,  # Channel 1: 5 items/minute
                15.0,  # Channel 2: 15 items/minute
            ]
        self.base_rate_limits = channel_rate_limits

        # State variables
        self.channels: list[ChannelState] = []
        self.media_queue: deque[str] = deque()
        self.transmission_history: list[TransmissionRecord] = []
        self.current_time: float = 0.0
        self.start_hour: int = 0

        # Episode tracking
        self.episode_step: int = 0
        self.total_reward: float = 0.0
        self.is_done: bool = False

        # State/action space dimensions
        self.state_dim = self._compute_state_dim()
        self.action_dim = 1 + 1  # delay (continuous) + channel (discrete)

    def _compute_state_dim(self) -> int:
        """Compute dimensionality of state vector."""
        # Queue size (1) + time-of-day cyclical encoding (sin + cos = 2)
        # + channel states (num_channels * 3) + history features (4:
        # avg delay, delay std, channel diversity, tx rate).
        return 1 + 2 + (self.num_channels * 3) + 4

    def reset(
        self, media_sequence: list[str], start_hour: Optional[int] = None
    ) -> np.ndarray:
        """
        Reset environment for a new episode.

        Args:
            media_sequence: List of media IDs to transmit
            start_hour: Starting hour of day [0-23] (random if None)

        Returns:
            Initial state vector
        """
        # Reset channels
        self.channels = [
            ChannelState(
                channel_id=i,
                rate_limit=self.base_rate_limits[i],
                last_transmission_time=-60.0,  # Allow immediate first transmission
                transmission_count=0,
            )
            for i in range(self.num_channels)
        ]

        # Reset queue
        self.media_queue = deque(media_sequence)

        # Reset transmission history
        self.transmission_history = []

        # Reset time
        self.current_time = 0.0
        self.start_hour = (
            start_hour if start_hour is not None else np.random.randint(0, 24)
        )

        # Reset episode tracking
        self.episode_step = 0
        self.total_reward = 0.0
        self.is_done = False

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Construct state vector from current environment state.

        Returns:
            State vector as numpy array
        """
        state_components = []

        # Queue size (normalized)
        queue_size = len(self.media_queue) / self.max_sequence_length
        state_components.append(queue_size)

        # Current time of day (cyclical encoding)
        current_hour = (self.start_hour + (self.current_time / 3600.0)) % 24
        hour_sin = np.sin(2 * np.pi * current_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * current_hour / 24.0)
        state_components.extend([hour_sin, hour_cos])

        # Channel states
        for channel in self.channels:
            time_since_last = self.current_time - channel.last_transmission_time
            state_components.extend(
                [
                    channel.rate_limit / 20.0,  # Normalize by max rate
                    np.clip(
                        time_since_last / 60.0, 0, 1
                    ),  # Time since last (normalized)
                    channel.transmission_count / max(len(self.transmission_history), 1),
                ]
            )

        # Transmission history features
        if len(self.transmission_history) > 0:
            recent_history = self.transmission_history[-10:]

            # Average delay
            avg_delay = np.mean([r.delay_from_previous for r in recent_history])
            state_components.append(np.clip(avg_delay / 60.0, 0, 1))

            # Delay variance
            delay_std = np.std([r.delay_from_previous for r in recent_history])
            state_components.append(np.clip(delay_std / 30.0, 0, 1))

            # Channel diversity (unique channels used)
            unique_channels = len(set(r.channel_id for r in recent_history))
            state_components.append(unique_channels / self.num_channels)

            # Transmission rate (items per minute)
            if len(recent_history) > 1:
                time_span = recent_history[-1].timestamp - recent_history[0].timestamp
                tx_rate = (
                    len(recent_history) / (time_span / 60.0) if time_span > 0 else 0
                )
                state_components.append(np.clip(tx_rate / 10.0, 0, 1))
            else:
                state_components.append(0.0)
        else:
            # No history yet
            state_components.extend([0.0, 0.0, 0.0, 0.0])

        # Pad to fixed size if needed
        while len(state_components) < self.state_dim:
            state_components.append(0.0)

        return np.array(state_components[: self.state_dim], dtype=np.float32)

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary with keys:
                - "delay": Delay in seconds before transmission
                - "channel": Channel index to use

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Parse action
        delay = float(action["delay"])
        channel_id = int(action["channel"]) % self.num_channels  # Ensure valid channel

        # Advance time
        self.current_time += delay

        # Check episode timeout
        if self.current_time >= self.max_episode_time:
            self.is_done = True
            return self._get_state(), -50.0, True, {"reason": "timeout"}

        # Check if queue is empty
        if len(self.media_queue) == 0:
            self.is_done = True
            return self._get_state(), 0.0, True, {"reason": "queue_empty"}

        # Check channel rate limit
        channel = self.channels[channel_id]
        if not channel.can_send(self.current_time):
            # Penalize for violating rate limit
            reward = -10.0
            return self._get_state(), reward, False, {"reason": "rate_limit_violation"}

        # Send media item
        media_id = self.media_queue.popleft()

        # Record transmission. The first item has no predecessor, so the
        # action delay is recorded as-is.
        record = TransmissionRecord(
            media_id=media_id,
            channel_id=channel_id,
            timestamp=self.current_time,
            delay_from_previous=delay,
        )
        self.transmission_history.append(record)

        # Update channel state
        channel.last_transmission_time = self.current_time
        channel.transmission_count += 1

        # Compute reward
        reward = self._compute_reward()
        self.total_reward += reward

        # Check if episode is complete
        done = len(self.media_queue) == 0
        self.is_done = done

        # Advance episode
        self.episode_step += 1

        # Prepare info
        info = {
            "queue_remaining": len(self.media_queue),
            "current_time": self.current_time,
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
        }

        # Get next state
        next_state = self._get_state()

        return next_state, reward, done, info

    def _compute_reward(self) -> float:
        """
        Compute reward for the current state.

        Reward = Throughput - λ * Warden_Score

        Returns:
            Reward value (scalar)
        """
        # Throughput component (items per minute)
        if self.current_time > 0:
            throughput = (len(self.transmission_history) / self.current_time) * 60.0
        else:
            throughput = 0.0

        # Warden detection penalty
        warden_penalty = 0.0

        # Evaluate recent transmission window
        if len(self.transmission_history) >= self.warden_window_size:
            recent = self.transmission_history[-self.warden_window_size :]

            # Extract delays and channels
            delays = [r.delay_from_previous for r in recent]
            channels = [r.channel_id for r in recent]

            # Convert to tensors
            device = next(self.warden.parameters()).device
            delays_tensor = torch.tensor([delays], dtype=torch.float32, device=device)
            channels_tensor = torch.tensor([channels], dtype=torch.long, device=device)

            # Get Warden verdict
            with torch.no_grad():
                verdict = self.warden(delays_tensor, channels_tensor)
                bot_probability = float(
                    verdict.bot_probability[0].item()
                    if verdict.bot_probability.dim() > 0
                    else verdict.bot_probability.item()
                )

            # Penalty scales with detection probability
            warden_penalty = bot_probability * self.lambda_stealth

        # Path diversity & entropy reward
        diversity_bonus = 0.0
        if len(self.transmission_history) >= 2:
            if (
                self.transmission_history[-1].channel_id
                != self.transmission_history[-2].channel_id
            ):
                diversity_bonus += 2.0  # Encourage multi-platform channel hopping

            # Channel distribution Shannon entropy bonus
            channel_counts = np.bincount(
                [r.channel_id for r in self.transmission_history],
                minlength=self.num_channels,
            )
            probs = channel_counts / len(self.transmission_history)
            entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-6, 1.0)))
            max_entropy = np.log2(self.num_channels)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            diversity_bonus += 3.0 * normalized_entropy

        # Combined reward
        reward = throughput - warden_penalty + diversity_bonus

        return reward

    def get_action_mask(self, projected_time: Optional[float] = None) -> np.ndarray:
        """
        Get binary mask for available channels (1 = can send, 0 = cooling down).
        """
        t = self.current_time if projected_time is None else projected_time
        mask = np.array(
            [1.0 if ch.can_send(t) else 0.0 for ch in self.channels], dtype=np.float32
        )
        if np.sum(mask) == 0:
            mask = np.ones(self.num_channels, dtype=np.float32)
        return mask

    def get_warden_score(self) -> float:
        """
        Get current Warden detection score for the full transmission history.

        Returns:
            Bot probability [0, 1]
        """
        if len(self.transmission_history) == 0:
            return 0.0

        history = self.transmission_history[-self.warden_window_size :]

        delays = [r.delay_from_previous for r in history]
        channels = [r.channel_id for r in history]

        # Pad if needed
        while len(delays) < self.warden_window_size:
            delays.append(0.0)
            channels.append(0)

        device = next(self.warden.parameters()).device
        delays_tensor = torch.tensor([delays], dtype=torch.float32, device=device)
        channels_tensor = torch.tensor([channels], dtype=torch.long, device=device)

        with torch.no_grad():
            verdict = self.warden(delays_tensor, channels_tensor)
            return float(
                verdict.bot_probability[0].item()
                if verdict.bot_probability.dim() > 0
                else verdict.bot_probability.item()
            )

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current environment state.

        Args:
            mode: Rendering mode ("human" or "ansi")

        Returns:
            String representation if mode=="ansi", else None
        """
        lines = []
        lines.append(f"=== Stealth Environment (Step {self.episode_step}) ===")
        lines.append(
            f"Time: {self.current_time:.1f}s | Queue: {len(self.media_queue)} items"
        )
        lines.append(f"Total Reward: {self.total_reward:.2f}")

        # Channel states
        lines.append("\nChannels:")
        for ch in self.channels:
            time_since = self.current_time - ch.last_transmission_time
            lines.append(
                f"  [{ch.channel_id}] Rate: {ch.rate_limit:.1f}/min | "
                f"Last: {time_since:.1f}s ago | Count: {ch.transmission_count}"
            )

        # Recent transmissions
        if len(self.transmission_history) > 0:
            lines.append(f"\nRecent Transmissions (last 5):")
            for record in self.transmission_history[-5:]:
                lines.append(
                    f"  {record.timestamp:.1f}s: {record.media_id} on channel {record.channel_id} "
                    f"(delay: {record.delay_from_previous:.1f}s)"
                )

        # Warden score
        warden_score = self.get_warden_score()
        lines.append(f"\nWarden Detection Score: {warden_score:.3f}")

        output = "\n".join(lines)

        if mode == "human":
            print(output)
            return None
        else:
            return output


if __name__ == "__main__":
    # Quick test
    print("Testing StealthEnvironment...")

    # Create environment
    warden = DeepPacketInspectionWarden(num_channels=3)
    env = StealthEnvironment(num_channels=3, warden=warden, lambda_stealth=50.0)

    # Reset with a media sequence
    media_sequence = [f"media_{i:03d}" for i in range(20)]
    state = env.reset(media_sequence, start_hour=14)

    print(f"✓ Initial state shape: {state.shape}")
    print(f"✓ State dim: {env.state_dim}")

    # Take a few random actions
    done = False
    total_reward = 0.0

    for step in range(10):
        if done:
            break

        # Random action
        action = {"delay": np.random.uniform(5, 15), "channel": np.random.randint(0, 3)}

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step}: delay={action['delay']:.1f}s, channel={action['channel']}, reward={reward:.2f}"
        )

    # Render final state
    print("\nFinal State:")
    env.render()

    print(f"\n✓ Environment test complete!")
    print(f"✓ Total reward: {total_reward:.2f}")

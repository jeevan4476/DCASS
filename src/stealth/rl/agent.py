# src/stealth/rl/agent.py
"""
PPO RL Agent for DCASS Stealth Optimization.

This module implements a Proximal Policy Optimization (PPO) agent that learns
to schedule transmissions to maximize throughput while evading the Warden.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path

from .environment import StealthEnvironment


@dataclass
class PPOConfig:
    """
    Configuration for PPO agent.
    """
    state_dim: int = 21
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE lambda parameter
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.05  # Increased for maximum channel path entropy
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RolloutBuffer:
    """Buffer for storing episode rollouts."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[dict] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.masks.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    The Actor outputs a policy for selecting actions (delay and channel).
    The Critic estimates the value function V(s).

    Architecture:
        State → Shared MLP → Actor Head (delay, channel) + Critic Head (value)

    Args:
        state_dim: Dimension of state vector
        num_channels: Number of channels (for discrete action)
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        state_dim: int,
        num_channels: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_channels = num_channels

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: outputs delay distribution and channel logits
        # Delay: Gaussian distribution (mean, log_std)
        self.delay_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive delays
        )

        self.delay_log_std = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Channel: Categorical distribution
        self.channel_logits = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels)
        )

        # Critic head: state value V(s)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.

        Args:
            state: State tensor, shape (batch_size, state_dim)

        Returns:
            Tuple of (delay_mean, delay_std, channel_logits, value)
        """
        # Shared features
        features = self.shared(state)

        # Actor outputs
        delay_mean = self.delay_mean(features)  # (batch_size, 1)
        delay_log_std = self.delay_log_std(features)  # (batch_size, 1)
        delay_std = torch.exp(delay_log_std).clamp(min=0.1, max=10.0)  # Bounded std

        channel_logits = self.channel_logits(features)  # (batch_size, num_channels)

        # Critic output
        value = self.value(features)  # (batch_size, 1)

        return delay_mean, delay_std, channel_logits, value

    def act(
        self,
        state: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, float, float]:
        """
        Sample an action from the policy with optional action masking.

        Args:
            state: State tensor, shape (batch_size, state_dim) or (state_dim,)
            channel_mask: Optional binary mask (1=valid, 0=masked), shape (batch_size, num_channels)

        Returns:
            Tuple of (action_dict, log_prob, value)
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        delay_mean, delay_std, channel_logits, value = self.forward(state)

        # Apply action masking for cooling-down channels
        if channel_mask is not None:
            if channel_mask.dim() == 1:
                channel_mask = channel_mask.unsqueeze(0)
            channel_logits = channel_logits + (1.0 - channel_mask) * -1e9

        # Sample delay from Gaussian
        delay_dist = Normal(delay_mean, delay_std)
        delay_sample = delay_dist.sample()

        # Clamp to a physically valid positive delay BEFORE computing log_prob so
        # the stored log_prob matches the action actually executed in the environment.
        # (If we clamped after, old_log_probs would reflect the raw sample while
        #  evaluate() would compute log_prob of the clamped action, breaking PPO.)
        delay_val = max(0.5, delay_sample.item())
        delay_clamped = torch.tensor([[delay_val]], dtype=delay_mean.dtype, device=delay_mean.device)
        delay_log_prob = delay_dist.log_prob(delay_clamped).sum(dim=-1)

        # Sample channel from Categorical
        channel_dist = Categorical(logits=channel_logits)
        channel_sample = channel_dist.sample()
        channel_log_prob = channel_dist.log_prob(channel_sample)

        # Total log probability
        total_log_prob = delay_log_prob + channel_log_prob

        # Construct action
        action = {
            "delay": delay_val,
            "channel": channel_sample.item()
        }

        return action, total_log_prob.item(), value.item()

    def evaluate(
        self,
        states: torch.Tensor,
        delay_actions: torch.Tensor,
        channel_actions: torch.Tensor,
        channel_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states, shape (batch_size, state_dim)
            delay_actions: Batch of delay actions, shape (batch_size, 1)
            channel_actions: Batch of channel actions, shape (batch_size,)
            channel_masks: Optional batch of channel masks, shape (batch_size, num_channels)

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        delay_mean, delay_std, channel_logits, values = self.forward(states)

        if channel_masks is not None:
            channel_logits = channel_logits + (1.0 - channel_masks) * -1e9

        # Delay distribution
        delay_dist = Normal(delay_mean, delay_std)
        delay_log_probs = delay_dist.log_prob(delay_actions).sum(dim=-1)
        delay_entropy = delay_dist.entropy().sum(dim=-1)

        # Channel distribution
        channel_dist = Categorical(logits=channel_logits)
        channel_log_probs = channel_dist.log_prob(channel_actions)
        channel_entropy = channel_dist.entropy()

        # Combined
        total_log_probs = delay_log_probs + channel_log_probs
        total_entropy = delay_entropy + channel_entropy

        return total_log_probs, values.squeeze(-1), total_entropy


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for stealth scheduling.

    Learns to maximize reward (throughput - stealth penalty) through
    policy gradient optimization with clipped objective.
    """

    def __init__(
        self,
        env: StealthEnvironment,
        config: Optional[PPOConfig] = None,
        actor_critic: Optional[ActorCritic] = None
    ):
        self.env = env
        self.config = config or PPOConfig(state_dim=env.state_dim)
        self.device = torch.device(self.config.device)

        # Create or use provided actor-critic
        if actor_critic is None:
            self.actor_critic = ActorCritic(
                state_dim=self.config.state_dim,
                num_channels=env.num_channels,
                hidden_dim=self.config.hidden_dim
            )
        else:
            self.actor_critic = actor_critic

        self.actor_critic.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.warden_scores: List[float] = []

    def select_action(
        self,
        state: np.ndarray,
        channel_mask: Optional[np.ndarray] = None
    ) -> Tuple[dict, float, float]:
        """
        Select an action using the current policy.

        Args:
            state: Current state
            channel_mask: Optional available channel binary mask

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask_tensor = None
        if channel_mask is not None:
            mask_tensor = torch.FloatTensor(channel_mask).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.act(state_tensor, channel_mask=mask_tensor)

        return action, log_prob, value

    def collect_rollout(
        self,
        media_sequence: list[str],
        max_steps: int = 1000
    ) -> float:
        """
        Collect a full episode rollout.

        Args:
            media_sequence: Media items to transmit
            max_steps: Maximum steps per episode

        Returns:
            Total episode reward
        """
        state = self.env.reset(media_sequence)
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # Select action with action masking
            mask = self.env.get_action_mask()
            action, log_prob, value = self.select_action(state, channel_mask=mask)

            # Store in buffer
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.masks.append(mask)
            self.buffer.log_probs.append(log_prob)
            self.buffer.values.append(value)

            # Take action in environment
            next_state, reward, done, info = self.env.step(action)

            # Store reward and done
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Record metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.warden_scores.append(self.env.get_warden_score())

        return episode_reward

    def compute_gae(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE-lambda) and returns.

        Returns:
            Tuple of (returns_tensor, normalized_advantages_tensor)
        """
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.config.gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer.values)]

        adv_tensor = torch.FloatTensor(advantages).to(self.device)
        ret_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return ret_tensor, adv_tensor

    def update(self) -> dict[str, float]:
        """
        Update policy using PPO with GAE-lambda advantages and action mask replay.

        Returns:
            Dictionary of training metrics
        """
        # Compute GAE advantages and returns
        returns, advantages = self.compute_gae()

        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)

        delay_actions = torch.FloatTensor([
            a["delay"] for a in self.buffer.actions
        ]).unsqueeze(1).to(self.device)

        channel_actions = torch.LongTensor([
            a["channel"] for a in self.buffer.actions
        ]).to(self.device)

        channel_masks = torch.FloatTensor(np.array(self.buffer.masks)).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)

        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.config.num_epochs):
            # Evaluate current policy with replayed masks
            log_probs, values, entropy = self.actor_critic.evaluate(
                states, delay_actions, channel_actions, channel_masks=channel_masks
            )

            # Ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.epsilon_clip,
                1.0 + self.config.epsilon_clip
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss +
                self.config.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.config.num_epochs,
            "value_loss": total_value_loss / self.config.num_epochs,
            "entropy": total_entropy / self.config.num_epochs
        }

    def train(
        self,
        num_episodes: int,
        media_sequence_generator: Optional[callable] = None,
        log_interval: int = 10
    ) -> List[float]:
        """
        Train the agent for multiple episodes.

        Args:
            num_episodes: Number of episodes to train
            media_sequence_generator: Function that generates media sequences
            log_interval: Episodes between logging

        Returns:
            List of episode rewards
        """
        if media_sequence_generator is None:
            # Default: generate random sequences
            def default_generator():
                length = np.random.randint(10, 30)
                return [f"media_{i:03d}" for i in range(length)]
            media_sequence_generator = default_generator

        print(f"Training PPO agent for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Generate media sequence
            media_sequence = media_sequence_generator()

            # Collect rollout
            self.collect_rollout(media_sequence)

            # Update policy
            metrics = self.update()

            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_length = np.mean(self.episode_lengths[-log_interval:])
                avg_warden = np.mean(self.warden_scores[-log_interval:])

                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Warden Score: {avg_warden:.3f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f}"
                )

        return self.episode_rewards

    def save(self, path: Path | str):
        """Save agent checkpoint."""
        checkpoint = {
            "actor_critic_state": self.actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "warden_scores": self.warden_scores
        }
        torch.save(checkpoint, str(path))
        print(f"Agent saved to {path}")

    def load(self, path: Path | str):
        """Load agent checkpoint into existing instance."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

        # If config is present in checkpoint, recreate actor_critic if dimensions differ
        if "config" in checkpoint and isinstance(checkpoint["config"], PPOConfig):
            cfg = checkpoint["config"]
            if (self.config.hidden_dim != cfg.hidden_dim or
                self.config.state_dim != cfg.state_dim):
                self.config = cfg
                self.actor_critic = ActorCritic(
                    state_dim=cfg.state_dim,
                    num_channels=self.env.num_channels,
                    hidden_dim=cfg.hidden_dim
                ).to(self.device)
                self.optimizer = optim.Adam(
                    self.actor_critic.parameters(),
                    lr=self.config.learning_rate
                )

        self.actor_critic.load_state_dict(checkpoint["actor_critic_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.warden_scores = checkpoint.get("warden_scores", [])
        print(f"Agent loaded from {path}")

    @classmethod
    def load_from_file(
        cls,
        path: Path | str,
        env: StealthEnvironment,
        device: Optional[str] = None
    ) -> PPOAgent:
        """Create and load a PPOAgent directly from a checkpoint file."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(path), map_location=dev, weights_only=False)
        cfg = checkpoint.get("config", PPOConfig(state_dim=env.state_dim, device=dev))
        if isinstance(cfg, PPOConfig):
            cfg.device = dev
        agent = cls(env=env, config=cfg)
        agent.load(path)
        return agent


if __name__ == "__main__":
    # Quick test
    print("Testing PPOAgent...")

    from ...analysis.adversarial.warden import DeepPacketInspectionWarden

    # Create environment
    warden = DeepPacketInspectionWarden(num_channels=3)
    env = StealthEnvironment(num_channels=3, warden=warden)

    # Create agent
    config = PPOConfig(state_dim=env.state_dim, device="cpu")
    agent = PPOAgent(env, config)

    print(f"✓ Actor-Critic parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

    # Test action selection
    state = env.reset([f"media_{i}" for i in range(10)])
    action, log_prob, value = agent.select_action(state)

    print(f"✓ Sample action: {action}")
    print(f"✓ Log prob: {log_prob:.4f}")
    print(f"✓ Value: {value:.4f}")

    # Test short training
    print("\nTesting short training...")
    agent.train(num_episodes=5, log_interval=2)

    print("\n✓ PPO agent test complete!")

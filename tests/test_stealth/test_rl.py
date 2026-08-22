"""
Unit and integration tests for PPO Reinforcement Learning closed-loop controller.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.stealth.rl.environment import StealthEnvironment
from src.stealth.rl.agent import PPOAgent, PPOConfig, ActorCritic
from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.stealth_scheduler import StealthScheduler


def test_rl_environment_lifecycle():
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=64)
    env = StealthEnvironment(num_channels=3, warden=warden, max_sequence_length=20)
    
    media = [f"item_{i}" for i in range(10)]
    state = env.reset(media, start_hour=14)
    
    assert isinstance(state, np.ndarray)
    assert len(state) == env.state_dim
    assert not env.is_done
    assert len(env.media_queue) == 10

    # Step action
    action = {"delay": 2.5, "channel": 0}
    next_state, reward, done, info = env.step(action)

    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert len(env.media_queue) == 9
    assert len(env.transmission_history) == 1


def test_actor_critic_forward_and_act():
    state_dim = 21
    num_channels = 3
    net = ActorCritic(state_dim=state_dim, num_channels=num_channels, hidden_dim=64)

    state = torch.randn(1, state_dim)
    delay_mean, delay_std, channel_logits, value = net.forward(state)

    assert delay_mean.shape == (1, 1)
    assert delay_std.shape == (1, 1)
    assert channel_logits.shape == (1, num_channels)
    assert value.shape == (1, 1)

    action, log_prob, val = net.act(state)
    assert "delay" in action
    assert "channel" in action
    assert 0 <= action["channel"] < num_channels
    assert action["delay"] > 0.0


def test_ppo_agent_train_step(tmp_path):
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=64)
    env = StealthEnvironment(num_channels=3, warden=warden, max_sequence_length=15)
    config = PPOConfig(
        state_dim=env.state_dim,
        hidden_dim=64,
        batch_size=8,
        num_epochs=2,
        device="cpu"
    )
    agent = PPOAgent(env=env, config=config)

    # Collect rollout on 5 items
    items = [f"m_{i}" for i in range(5)]
    reward = agent.collect_rollout(items)
    assert len(agent.buffer) > 0

    # Train step
    metrics = agent.update()
    assert "policy_loss" in metrics
    assert "value_loss" in metrics

    # Save and load
    save_path = tmp_path / "test_rl_agent.pt"
    agent.save(str(save_path))
    assert save_path.exists()

    agent_loaded = PPOAgent.load_from_file(str(save_path), env=env, device="cpu")
    assert agent_loaded.config.state_dim == env.state_dim


def test_stealth_scheduler_rl_mode(tmp_path):
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=64)
    env = StealthEnvironment(num_channels=3, warden=warden, max_sequence_length=15)
    config = PPOConfig(state_dim=env.state_dim, hidden_dim=64, device="cpu")
    agent = PPOAgent(env=env, config=config)
    save_path = tmp_path / "rl_agent.pt"
    agent.save(str(save_path))

    scheduler = StealthScheduler(num_channels=3)
    items = [f"doc_{i}" for i in range(6)]
    sched = scheduler.schedule(items, mode="rl", rl_checkpoint=save_path)

    assert len(sched["items"]) == 6
    assert len(sched["delays"]) == 6
    assert len(sched["channels"]) == 6
    assert sched["mode_used"] == "rl"

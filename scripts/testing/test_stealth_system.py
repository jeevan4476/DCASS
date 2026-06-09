#!/usr/bin/env python3
# scripts/test_stealth_system.py
"""
Integration test for DCASS Stealth System.

Tests all components:
1. GAN Generator
2. Adversarial Warden
3. RL Environment
4. PPO Agent
5. End-to-end training pipeline
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stealth.gan.generator import TemporalPatternGenerator, sample_latent
from src.analysis.adversarial.warden import DeepPacketInspectionWarden, compute_warden_loss
from src.stealth.rl.environment import StealthEnvironment
from src.stealth.rl.agent import PPOAgent, PPOConfig


def test_generator():
    """Test GAN Generator."""
    print("\n" + "=" * 60)
    print("TEST 1: Temporal Pattern Generator")
    print("=" * 60)

    generator = TemporalPatternGenerator(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3
    )

    # Test generation
    z = sample_latent(batch_size=4, latent_dim=128)
    time_of_day = torch.randint(0, 24, (4,)).float()

    schedule = generator(z, sequence_length=20, time_of_day=time_of_day)

    print(f"✓ Generator created")
    print(f"✓ Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"✓ Output delays shape: {schedule.delays.shape}")
    print(f"✓ Output channels shape: {schedule.channel_logits.shape}")
    print(f"✓ Sample delays: {schedule.delays[0, :5].tolist()}")
    print(f"✓ Sample channels: {schedule.sample_channels()[0, :5].tolist()}")

    return generator


def test_warden(generator):
    """Test Adversarial Warden."""
    print("\n" + "=" * 60)
    print("TEST 2: Adversarial Warden")
    print("=" * 60)

    warden = DeepPacketInspectionWarden(
        num_channels=3,
        hidden_dim=256
    )

    # Generate fake traffic
    z = sample_latent(batch_size=8, latent_dim=128)
    time_of_day = torch.randint(0, 24, (8,)).float()
    fake_schedule = generator(z, sequence_length=20, time_of_day=time_of_day)

    fake_delays = fake_schedule.delays
    fake_channels = fake_schedule.sample_channels()

    # Create real human-like traffic
    real_delays = torch.abs(torch.randn(8, 20) * 5 + 10)
    real_channels = torch.randint(0, 3, (8, 20))

    # Test Warden
    real_verdict = warden(real_delays, real_channels)
    fake_verdict = warden(fake_delays, fake_channels)

    print(f"✓ Warden created")
    print(f"✓ Parameters: {sum(p.numel() for p in warden.parameters()):,}")
    print(f"✓ Real traffic bot probability: {real_verdict.bot_probability.mean():.3f}")
    print(f"✓ Fake traffic bot probability: {fake_verdict.bot_probability.mean():.3f}")
    print(f"✓ Warden loss: {compute_warden_loss(real_verdict, fake_verdict):.4f}")

    # Check detection
    print(f"✓ Suspicious real samples: {real_verdict.is_suspicious().sum()}/{len(real_verdict.bot_probability)}")
    print(f"✓ Suspicious fake samples: {fake_verdict.is_suspicious().sum()}/{len(fake_verdict.bot_probability)}")

    return warden


def test_rl_environment(warden):
    """Test RL Environment."""
    print("\n" + "=" * 60)
    print("TEST 3: RL Environment")
    print("=" * 60)

    env = StealthEnvironment(
        num_channels=3,
        warden=warden,
        lambda_stealth=50.0
    )

    # Reset environment
    media_sequence = [f"media_{i:03d}" for i in range(20)]
    state = env.reset(media_sequence, start_hour=14)

    print(f"✓ Environment created")
    print(f"✓ State dimension: {env.state_dim}")
    print(f"✓ Initial state shape: {state.shape}")

    # Take a few steps
    total_reward = 0.0
    for step in range(5):
        action = {
            "delay": np.random.uniform(5, 15),
            "channel": np.random.randint(0, 3)
        }

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"  Step {step + 1}: delay={action['delay']:.1f}s, "
            f"channel={action['channel']}, reward={reward:.2f}"
        )

        if done:
            break

    print(f"✓ Total reward: {total_reward:.2f}")
    print(f"✓ Warden score: {env.get_warden_score():.3f}")

    return env


def test_ppo_agent(env):
    """Test PPO Agent."""
    print("\n" + "=" * 60)
    print("TEST 4: PPO Agent")
    print("=" * 60)

    config = PPOConfig(
        state_dim=env.state_dim,
        device="cpu",
        batch_size=32,
        num_epochs=2
    )

    agent = PPOAgent(env, config)

    print(f"✓ Agent created")
    print(f"✓ Actor-Critic parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

    # Test action selection
    state = env.reset([f"media_{i}" for i in range(10)])
    action, log_prob, value = agent.select_action(state)

    print(f"✓ Sample action: {action}")
    print(f"✓ Log probability: {log_prob:.4f}")
    print(f"✓ Value estimate: {value:.4f}")

    # Test short training
    print("\n  Training for 5 episodes...")
    rewards = agent.train(num_episodes=5, log_interval=2)

    print(f"✓ Training complete")
    print(f"✓ Average reward: {np.mean(rewards):.2f}")
    print(f"✓ Final reward: {rewards[-1]:.2f}")

    return agent


def test_end_to_end():
    """Full integration test."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Integration")
    print("=" * 60)

    # 1. Create Generator
    print("\n[1/5] Creating Generator...")
    generator = TemporalPatternGenerator(num_channels=3)

    # 2. Create Warden
    print("[2/5] Creating Warden...")
    warden = DeepPacketInspectionWarden(num_channels=3)

    # 3. Create Environment
    print("[3/5] Creating RL Environment...")
    env = StealthEnvironment(num_channels=3, warden=warden)

    # 4. Create Agent
    print("[4/5] Creating PPO Agent...")
    config = PPOConfig(state_dim=env.state_dim, device="cpu")
    agent = PPOAgent(env, config)

    # 5. Run full episode
    print("[5/5] Running full episode...")
    media_sequence = [f"media_{i:03d}" for i in range(15)]
    episode_reward = agent.collect_rollout(media_sequence, max_steps=100)

    print(f"\n✓ Episode completed!")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Episode length: {agent.episode_lengths[-1]}")
    print(f"  Warden score: {agent.warden_scores[-1]:.3f}")

    # 6. Update policy
    print("\n  Updating policy...")
    metrics = agent.update()

    print(f"✓ Policy updated!")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("DCASS Stealth System Integration Test")
    print("=" * 60)

    try:
        # Test individual components
        generator = test_generator()
        warden = test_warden(generator)
        env = test_rl_environment(warden)
        agent = test_ppo_agent(env)

        # Test full integration
        test_end_to_end()

        # Summary
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe DCASS Stealth System is ready for:")
        print("  1. GAN training (Generator vs. Warden)")
        print("  2. RL training (PPO agent optimization)")
        print("  3. Docker simulation (Alice-Bob network)")
        print("\nNext steps:")
        print("  - Train Warden on real human traffic data")
        print("  - Train RL agent for 1000+ episodes")
        print("  - Deploy in Docker sandbox")
        print("  - Integrate with real API channels")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

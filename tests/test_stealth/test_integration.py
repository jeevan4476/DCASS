# tests/test_stealth/test_integration.py
"""
Integration test for the GAN -> frozen Warden -> RL -> StealthScheduler
pipeline described in docs/GAN_RL_INTEGRATION_PLAN.md.

Trains a tiny GAN (2 epochs, synthetic data) and a tiny RL agent (3
episodes) against the resulting frozen Warden, then runs the scheduler
end-to-end and checks the schedule's shape/value invariants. Deliberately
small so this runs in CI in a few seconds; it is not meant to produce a
meaningful Warden AUC, only to exercise the full wiring.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.gan.trainer import GANTrainer, HumanTrafficDataset, TrainingConfig
from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.stealth.rl.environment import StealthEnvironment
from src.stealth.stealth_scheduler import StealthScheduler


def _train_tiny_gan(tmp_path: Path) -> Path:
    config = TrainingConfig(
        latent_dim=16,
        hidden_dim=32,
        num_channels=3,
        max_sequence_length=20,
        batch_size=4,
        num_epochs=2,
        warden_steps=1,
        use_gradient_penalty=True,
        device="cpu",
        checkpoint_dir=tmp_path / "gan_ckpts",
        log_interval=100,
    )

    # No data file -> HumanTrafficDataset falls back to synthetic data.
    dataset = HumanTrafficDataset(tmp_path / "nonexistent.json", config.max_sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    trainer = GANTrainer(config)
    trainer.train(loader, num_epochs=2)

    gan_checkpoint = tmp_path / "gan_generator.pt"
    torch.save(
        {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "generator_state": trainer.generator.state_dict(),
            "warden_state": trainer.warden.state_dict(),
            "config": trainer.config,
            "metrics_history": trainer.metrics_history,
        },
        gan_checkpoint,
    )
    return gan_checkpoint


def _train_tiny_rl(gan_checkpoint: Path, tmp_path: Path) -> Path:
    ckpt = torch.load(gan_checkpoint, map_location="cpu", weights_only=False)
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=32)
    warden.load_state_dict(ckpt["warden_state"])
    warden.eval()
    for p in warden.parameters():
        p.requires_grad_(False)

    env = StealthEnvironment(num_channels=3, warden=warden, lambda_stealth=10.0)
    config = PPOConfig(state_dim=env.state_dim, hidden_dim=32, device="cpu")
    agent = PPOAgent(env, config)

    rng = np.random.default_rng(0)

    def media_gen():
        n = rng.integers(10, 15)
        return [f"media_{i:03d}" for i in range(n)]

    agent.train(num_episodes=3, media_sequence_generator=media_gen, log_interval=100)

    rl_checkpoint = tmp_path / "rl_agent.pt"
    agent.save(rl_checkpoint)
    return rl_checkpoint


def test_gan_rl_scheduler_end_to_end(tmp_path):
    """
    GAN (frozen Warden) -> RL -> StealthScheduler.schedule() should produce
    a valid schedule whose length matches the input media_ids, with the
    RL policy actually loaded (mode_used == "rl").
    """
    gan_checkpoint = _train_tiny_gan(tmp_path)
    assert gan_checkpoint.exists()

    rl_checkpoint = _train_tiny_rl(gan_checkpoint, tmp_path)
    assert rl_checkpoint.exists()

    scheduler = StealthScheduler(num_channels=3)
    media_ids = [f"doc_{i}" for i in range(12)]
    schedule = scheduler.schedule(
        media_ids,
        mode="rl",
        rl_checkpoint=rl_checkpoint,
        gan_checkpoint=gan_checkpoint,
    )

    assert schedule["mode_used"] == "rl"
    assert len(schedule["items"]) == len(media_ids)
    assert len(schedule["delays"]) == len(media_ids)
    assert len(schedule["channels"]) == len(media_ids)
    assert all(d >= 0 for d in schedule["delays"])
    assert all(0 <= c < 3 for c in schedule["channels"])


def test_gan_scheduler_end_to_end(tmp_path):
    """GAN-only mode should also produce a schedule matching the payload length."""
    gan_checkpoint = _train_tiny_gan(tmp_path)

    scheduler = StealthScheduler(num_channels=3)
    media_ids = [f"doc_{i}" for i in range(8)]
    schedule = scheduler.schedule(media_ids, mode="gan", gan_checkpoint=gan_checkpoint)

    assert schedule["mode_used"] == "gan"
    assert len(schedule["delays"]) == len(media_ids)
    assert len(schedule["channels"]) == len(media_ids)
    assert all(d >= 0 for d in schedule["delays"])

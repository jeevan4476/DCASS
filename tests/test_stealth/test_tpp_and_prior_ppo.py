"""Unit tests for the Autoregressive Log-Normal Mixture TPP, Prior-Guided PPO,
scaled Warden features, and StatisticalTrafficDetector.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analysis.adversarial.statistical_detector import StatisticalTrafficDetector
from src.analysis.adversarial.warden import DeepPacketInspectionWarden
from src.stealth.rl.tpp_ppo import PriorGuidedPPOAgent, PriorGuidedPPOConfig
from src.stealth.stealth_scheduler import CHANNEL_COOLDOWNS, StealthScheduler
from src.stealth.tpp import (
    TPPConfig,
    TPPTrainingConfig,
    TemporalPointProcessGenerator,
    train_tpp_model,
)


def _count_violations(delays: list[float], channels: list[int]) -> int:
    ts = np.cumsum(np.asarray(delays, dtype=np.float64))
    last: dict[int, float] = {}
    violations = 0
    for idx, ch in enumerate(channels):
        if ch in last and (ts[idx] - last[ch]) < CHANNEL_COOLDOWNS[ch] - 1e-12:
            violations += 1
        last[ch] = float(ts[idx])
    return violations


def _synthetic_sessions(n: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    sessions = []
    for i in range(n):
        hour = int(rng.integers(0, 24))
        scale = 4.0 if 9 <= hour <= 20 else 15.0
        length = int(rng.integers(20, 45))
        delays = np.clip(rng.lognormal(np.log(scale), 0.9, size=length), 0.3, 100.0)
        channels = rng.integers(0, 3, size=length)
        sessions.append(
            {
                "delays": delays.tolist(),
                "channels": channels.tolist(),
                "time_of_day": hour,
                "contributor_id": f"user-{i % 6}",
            }
        )
    return sessions


def test_tpp_training_generation_cooldowns_and_checkpoint(tmp_path):
    sessions = _synthetic_sessions(36, seed=11)
    model, summary = train_tpp_model(
        train_sessions=sessions[:24],
        val_sessions=sessions[24:],
        model_config=TPPConfig(hidden_dim=32, num_components=4, num_layers=1),
        train_config=TPPTrainingConfig(epochs=2, batch_size=8, device="cpu", seed=11),
        cooldowns=CHANNEL_COOLDOWNS,
        out_path=tmp_path / "tpp.pt",
    )
    assert summary["best_epoch"] >= 1
    assert np.isfinite(summary["best_val_loss"])

    reloaded = TemporalPointProcessGenerator.load_checkpoint(tmp_path / "tpp.pt", device="cpu")
    delays, channels = reloaded.generate_routed_schedule(
        num_items=35,
        time_of_day=14.0,
        cooldowns=CHANNEL_COOLDOWNS,
        pause_cap_s=60.0,
        rng=np.random.default_rng(123),
        device="cpu",
    )
    assert len(delays) == 35
    assert len(channels) == 35
    assert all(d >= reloaded.min_delay for d in delays)
    assert _count_violations(delays, channels) == 0


def test_prior_guided_ppo_training_and_scheduler_integration(tmp_path):
    tpp = TemporalPointProcessGenerator(
        TPPConfig(hidden_dim=32, num_components=4, num_layers=1, max_delay=60.0)
    )
    tpp_path = tmp_path / "tpp.pt"
    tpp.save_checkpoint(tpp_path)

    warden = DeepPacketInspectionWarden(
        num_channels=3, hidden_dim=32, num_transformer_layers=1, scaled_features=True
    )
    warden.eval()

    ppo_cfg = PriorGuidedPPOConfig(
        num_channels=3,
        num_components=4,
        hidden_dim=32,
        pause_cap_s=60.0,
        ppo_epochs=2,
        minibatch_size=16,
        device="cpu",
    )
    agent = PriorGuidedPPOAgent(tpp=tpp, config=ppo_cfg, cooldowns=CHANNEL_COOLDOWNS)
    history = agent.train(
        warden=warden,
        num_updates=2,
        episodes_per_update=2,
        carriers_per_episode=15,
        seed=5,
    )
    assert len(history) == 2
    assert np.isfinite(history[-1]["mean_reward"])

    ppo_path = tmp_path / "tpp_ppo.pt"
    agent.save_checkpoint(ppo_path)

    scheduler = StealthScheduler(num_channels=3, device="cpu", seed=9)
    media = [f"m_{i}" for i in range(30)]

    tpp_sched = scheduler.schedule(
        media, mode="tpp", tpp_checkpoint=tpp_path, time_of_day=14.0
    )
    assert tpp_sched["mode_used"] == "tpp"
    assert len(tpp_sched["delays"]) == 30
    assert _count_violations(tpp_sched["delays"], tpp_sched["channels"]) == 0

    ppo_sched = scheduler.schedule(
        media, mode="tpp_ppo", tpp_ppo_checkpoint=ppo_path, time_of_day=3.0
    )
    assert ppo_sched["mode_used"] == "tpp_ppo"
    assert len(ppo_sched["delays"]) == 30
    assert _count_violations(ppo_sched["delays"], ppo_sched["channels"]) == 0


def test_scaled_warden_and_statistical_detector_stability():
    warden = DeepPacketInspectionWarden(
        num_channels=3,
        hidden_dim=32,
        num_transformer_layers=1,
        scaled_features=True,
        ignore_channels=True,
    )
    warden.eval()

    # Heavy-tailed sequence with 300s pause and padding
    delays = torch.zeros(2, 64)
    delays[0, :20] = torch.linspace(0.5, 300.0, 20)
    delays[1, :15] = torch.linspace(1.0, 120.0, 15)
    mask = torch.zeros(2, 64)
    mask[0, :20] = 1.0
    mask[1, :15] = 1.0
    ch_a = torch.zeros(2, 64, dtype=torch.long)
    ch_b = torch.ones(2, 64, dtype=torch.long)

    with torch.no_grad():
        va = warden(delays, ch_a, mask=mask)
        vb = warden(delays, ch_b, mask=mask)
    assert torch.isfinite(va.bot_probability).all()
    # Timing-only mode must be invariant to channel IDs
    assert torch.allclose(va.bot_probability, vb.bot_probability, atol=1e-6)

    # Statistical detector sanity check
    humans = _synthetic_sessions(30, seed=1)
    bots = [
        {"delays": [3.0] * 30, "channels": [0] * 30}
        for _ in range(30)
    ]
    det = StatisticalTrafficDetector(ignore_channels=True, max_iter=30, seed=1)
    det.fit(humans + bots, [0] * len(humans) + [1] * len(bots))
    p_human = det.score_sequence(humans[0]["delays"])
    p_bot = det.score_sequence(bots[0]["delays"])
    assert p_bot > p_human

#!/usr/bin/env python3
"""
Train the WGAN-GP pair against two training sets and record what happens.

The shipped generator was trained against `data/human_traffic.json`, which
is not in the repository, so `HumanTrafficDataset` fell back to a synthetic
corpus: exponential delays with a rate drawn per session, uniformly random
channels per step, and a time of day drawn independently of the delays. The
detector, by contrast, was trained on the traffic corpus in
`storage/data/traffic/`, which has a different structure.

This script runs the same WGAN-GP training against both and records, per
epoch: the two losses, the gradient penalty, the generator's output spread,
and the critic's sensitivity to its input. The critic's sensitivity is the
probe the diagnostic used: the spread of its mean anomaly score across six
deliberately different delay sequences. A critic that cannot tell a
metronome from an exponential process is not measuring anything.

Outputs:
    storage/benchmarks/wgan_training_data_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402
from src.stealth.gan.trainer import (  # noqa: E402
    GANTrainer,
    HumanTrafficDataset,
    TrainingConfig,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
FALLBACK = PROJECT_ROOT / "data" / "human_traffic.json"  # does not exist
SEQ_LEN = 60
EPOCHS = 6


def build_probe_inputs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The diagnostic's six probes: delays with different statistical character."""
    rng = np.random.default_rng(0)
    n = 50
    constant = np.full(n, 3.0, dtype=np.float32)
    jitter = np.clip(3.0 + rng.normal(0, 1.5, n), 0.2, None).astype(np.float32)
    poisson = rng.exponential(scale=3.0, size=n).astype(np.float32)
    bimodal = np.concatenate([np.full(25, 0.5), np.full(25, 60.0)]).astype(np.float32)
    tight = np.full(n, 0.5, dtype=np.float32)
    ramp = np.linspace(0.5, 30.0, n, dtype=np.float32)
    return {
        "constant_3s": (constant, np.zeros(n, dtype=np.int64)),
        "jittered_3s": (jitter, rng.integers(0, 3, n)),
        "exponential": (poisson, rng.integers(0, 3, n)),
        "bimodal_burst_pause": (bimodal, rng.integers(0, 3, n)),
        "constant_0p5s": (tight, np.zeros(n, dtype=np.int64)),
        "ramp": (ramp, rng.integers(0, 3, n)),
    }


def critic_sensitivity(warden: DeepPacketInspectionWarden) -> float:
    """The spread of the mean anomaly score across the six probes.

    A trained critic that has collapsed returns the same value for every
    input, so this goes to zero as the collapse sets in.
    """
    values = []
    for delays, channels in build_probe_inputs().values():
        d = torch.tensor(delays).unsqueeze(0).to(DEVICE)
        c = torch.tensor(channels).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            verdict = warden(d, c)
        values.append(float(verdict.anomaly_scores.mean()))
    return float(np.std(values))


def generator_spread(trainer: GANTrainer) -> float:
    """The standard deviation of the generator's raw delay output.

    The shipped generator's raw output had a standard deviation of about
    0.002, which is a quarter of one percent of the unit output range: a
    near-constant clock.
    """
    z = torch.randn(64, trainer.config.latent_dim, device=trainer.device)
    tod = torch.full((64,), 14.0, device=trainer.device)
    with torch.no_grad():
        schedule = trainer.generator(z, SEQ_LEN, tod)
    return float(schedule.delays.std())


def run_arm(name: str, data_path: Path) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    config = TrainingConfig(
        latent_dim=128,
        hidden_dim=256,
        num_channels=3,
        max_sequence_length=SEQ_LEN,
        num_epochs=EPOCHS,
        batch_size=64,
        lambda_gp=10.0,
        generator_lr=1e-4,
        warden_lr=2e-4,
        warden_steps=5,
        device=DEVICE,
    )
    dataset = HumanTrafficDataset(data_path, config.max_sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    trainer = GANTrainer(config)

    history = []
    for epoch in range(EPOCHS):
        metrics = trainer.train(loader, num_epochs=1)
        m = metrics[-1]
        row = {
            "epoch": epoch,
            "generator_loss": round(float(m.generator_loss), 4),
            "warden_loss": round(float(m.warden_loss), 4),
            "gradient_penalty": round(float(m.gradient_penalty), 4)
            if m.gradient_penalty is not None
            else None,
            "real_bot_prob": round(float(m.real_bot_prob), 6),
            "fake_bot_prob": round(float(m.fake_bot_prob), 6),
            "generator_delay_std": round(generator_spread(trainer), 6),
            "critic_anomaly_spread": round(critic_sensitivity(trainer.warden), 8),
        }
        history.append(row)
        print(
            f"  [{name}] epoch {epoch}: gen {row['generator_loss']:>10}  "
            f"warden {row['warden_loss']:>10}  gp {row['gradient_penalty']}  "
            f"gen_std {row['generator_delay_std']:.5f}  "
            f"critic_spread {row['critic_anomaly_spread']:.2e}"
        )
    return {
        "data_path": str(data_path.relative_to(PROJECT_ROOT))
        if data_path.is_relative_to(PROJECT_ROOT)
        else str(data_path),
        "data_exists": data_path.exists(),
        "history": history,
    }


def main() -> int:
    print("=" * 78)
    print(" WGAN-GP TRAINING: WHICH TRAINING SET?")
    print("=" * 78)
    print(f"Device: {DEVICE}")
    print()

    arms = {}
    for name, path in (("synthetic_fallback", FALLBACK), ("traffic_corpus", CORPUS)):
        print(f"Training against {name} ({path.name})...")
        arms[name] = run_arm(name, path)
        print()

    out = PROJECT_ROOT / "storage" / "benchmarks" / "wgan_training_data_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "wgan-training-data-probe-v1",
                "epochs": EPOCHS,
                "note": (
                    "The same WGAN-GP training run against the synthetic "
                    "fallback the shipped generator was trained on and "
                    "against the traffic corpus the detector was trained on. "
                    "'critic_anomaly_spread' is the spread of the critic's "
                    "mean anomaly score across the diagnostic's six delay "
                    "probes; it falls to zero as the critic collapses. "
                    "'generator_delay_std' is the standard deviation of the "
                    "generator's raw delay output."
                ),
                "arms": arms,
            },
            indent=2,
        )
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

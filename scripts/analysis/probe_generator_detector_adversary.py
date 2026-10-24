#!/usr/bin/env python3
"""
Can the generator be trained to look human to a detector that works?

The WGAN-GP training diverges on the traffic corpus as well as on the
synthetic fallback, so the data is not the cause. The divergence has a
specific mechanism: the critic's gradient with respect to its input
collapses to 1.3e-5, at which the gradient penalty sits at exactly
lambda_gp and is inert, so nothing constrains the critic's scale and the
generator's loss chases it away.

The probe in `probe_critic_input_gradient.py` rules out the
architectural hypothesis the diagnostic proposed: a pre-LN residual stack
does not give the critic a larger input gradient than the shipped
post-LN one.

This script tests the alternative that does not need the WGAN-GP critic
at all. The generator's objective becomes: minimise the bot probability
that a *working* detector assigns to the generated traffic. That
objective is bounded, so the race between the generator and the critic
that diverges cannot happen. The detector plays the adversary's role,
which is the role it was always meant to play, and the generated traffic
is then measured with a *different* detector, one the generator never
trained against.

Outputs:
    storage/benchmarks/generator_detector_adversary_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from src.utils.checkpoint_io import load_trained_checkpoint
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402
from src.stealth.gan.generator import TemporalPatternGenerator  # noqa: E402
from src.stealth.stealth_scheduler import (  # noqa: E402
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    GAN_DELAY_OFFSET,
    GAN_DELAY_SCALE,
    canonical_delay_mapping,
)


def map_delays_torch(raw: torch.Tensor) -> torch.Tensor:
    """The canonical delay mapping, differentiably.

    The scheduler's mapping is a numpy clip, so no gradient survives it.
    The generator's usable delay range is a narrow slice of its output
    range, and the mapping saturates outside it, so the gradient has to
    reach the generator through the mapping for the objective to mean
    anything at all.
    """
    return torch.clamp(
        (raw - 0.5) * GAN_DELAY_SCALE + GAN_DELAY_OFFSET,
        GAN_DELAY_MIN,
        GAN_DELAY_MAX,
    )
from src.stealth.gan.trainer import HumanTrafficDataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json"
SEQ_LEN = 60
EPOCHS = 40
LAMBDA_MATCH = 2.0
BATCH = 64
LATENT = 128


def enable_input_gradients(model: DeepPacketInspectionWarden) -> None:
    """Put the frozen adversary in a mode where gradients reach its input.

    cuDNN's RNN backward refuses to run unless the module is in training
    mode, but the adversary's dropout must stay off, or the signal the
    generator is chasing would be noise.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def load_detector(path: Path, key: str) -> DeepPacketInspectionWarden:
    ckpt = load_trained_checkpoint(path, map_location=DEVICE,)
    model = DeepPacketInspectionWarden(num_channels=3, hidden_dim=256)
    model.load_state_dict(ckpt[key] if key in ckpt else ckpt["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def batch_mask(seq_lengths: torch.Tensor, length: int) -> torch.Tensor:
    positions = torch.arange(length, device=seq_lengths.device).unsqueeze(0)
    return (positions < seq_lengths.unsqueeze(1)).float()


def generator_spread(generator, n: int = 64) -> float:
    z = torch.randn(n, LATENT, device=DEVICE)
    tod = torch.full((n,), 14.0, device=DEVICE)
    with torch.no_grad():
        schedule = generator(z, SEQ_LEN, tod)
    return float(schedule.delays.std())


def main() -> int:
    print("=" * 78)
    print(" GENERATOR TRAINED AGAINST A WORKING DETECTOR")
    print("=" * 78)
    print(f"Device: {DEVICE}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    adversary = load_detector(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt", "state_dict"
    )
    enable_input_gradients(adversary)
    evaluator = load_detector(
        PROJECT_ROOT / "storage" / "models" / "traffic_detector_eval.pt", "state_dict"
    )

    dataset = HumanTrafficDataset(CORPUS, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    generator = TemporalPatternGenerator(
        latent_dim=LATENT, hidden_dim=256, num_channels=3, max_sequence_length=SEQ_LEN
    ).to(DEVICE)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    history = []
    for epoch in range(EPOCHS):
        generator.train()
        total, batches, adv_scores = 0.0, 0, []
        for real in loader:
            delays = real["delays"].to(DEVICE)
            time_of_day = real["time_of_day"].to(DEVICE)
            seq_lengths = real["sequence_length"].to(DEVICE)
            mask = batch_mask(seq_lengths, delays.size(1))

            b = delays.size(0)
            z = torch.randn(b, LATENT, device=DEVICE)
            schedule = generator(z, delays.size(1), time_of_day)
            channel_probs = schedule.channel_probs_straight_through()

            # The adversary must see the delays the scheduler will send,
            # not the generator's raw output. The canonical mapping is
            # applied first, so the objective is the traffic that actually
            # goes on the wire.
            mapped_t = map_delays_torch(schedule.delays)

            # The adversary's parameters are frozen, so the gradient flows
            # to the generator's input and not into the adversary.
            verdict = adversary(mapped_t, channel_probs, mask=mask)
            # The bounded objective: push the adversary's bot probability
            # down. There is no unbounded score to run away with.
            loss = verdict.bot_probability.mean()

            # A distribution-matching term, so the generator does not
            # simply find one degenerate shape the detector happens to
            # like. It compares the log-delay distributions, which are
            # bounded, rather than the Wasserstein distance between two
            # unbounded critic scores.
            real_mask = mask > 0.5
            gen_log = torch.log(mapped_t.clamp(min=0.05))
            real_log = torch.log(delays.clamp(min=0.05))
            gen_vals = gen_log[real_mask]
            real_vals = real_log[real_mask]
            gen_sorted = torch.sort(gen_vals).values
            real_sorted = torch.sort(real_vals).values
            n = min(gen_sorted.numel(), real_sorted.numel())
            if n > 1:
                match = (gen_sorted[:n] - real_sorted[:n]).abs().mean()
                loss = loss + LAMBDA_MATCH * match

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss)
            batches += 1
            adv_scores.append(float(verdict.bot_probability.mean()))

        generator.eval()
        z = torch.randn(64, LATENT, device=DEVICE)
        tod = torch.full((64,), 14.0, device=DEVICE)
        with torch.no_grad():
            schedule = generator(z, SEQ_LEN, tod)
            mapped_eval = map_delays_torch(schedule.delays)
            held = evaluator(
                mapped_eval, schedule.channel_probs_straight_through()
            )
        mapped = canonical_delay_mapping(schedule.delays.detach().cpu().numpy())
        row = {
            "epoch": epoch,
            "adversary_bot_probability": round(float(np.mean(adv_scores)), 6),
            "held_out_evaluator_bot_probability": round(float(held.bot_probability.mean()), 6),
            "generator_delay_std": round(generator_spread(generator), 5),
            "mapped_delay_median": round(float(np.median(mapped)), 4),
            "mapped_delay_p95": round(float(np.percentile(mapped, 95)), 4),
            "mapped_delay_max": round(float(mapped.max()), 4),
        }
        history.append(row)
        print(
            f"  epoch {epoch}: adversary {row['adversary_bot_probability']:.4f}  "
            f"held-out {row['held_out_evaluator_bot_probability']:.4f}  "
            f"gen_std {row['generator_delay_std']:.4f}  "
            f"mapped median {row['mapped_delay_median']:.2f}s "
            f"p95 {row['mapped_delay_p95']:.2f}s"
        )

    out = PROJECT_ROOT / "storage" / "benchmarks" / "generator_detector_adversary_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "generator-detector-adversary-probe-v1",
                "epochs": EPOCHS,
                "adversary": "storage/models/traffic_detector.pt",
                "evaluator": "storage/models/traffic_detector_eval.pt",
                "note": (
                    "The generator's objective is the adversary's bot "
                    "probability, which is bounded, so the unbounded "
                    "Wasserstein race that diverged cannot happen. The "
                    "held-out evaluator is a detector the generator never "
                    "trained against."
                ),
                "history": history,
            },
            indent=2,
        )
    )
    ckpt_path = PROJECT_ROOT / "storage" / "models" / "gan_generator_detector_adversary.pt"
    torch.save(
        {"generator_state": generator.state_dict(), "config": {"latent_dim": LATENT}},
        ckpt_path,
    )
    print(f"\nSaved {out}")
    print(f"Saved the generator to {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

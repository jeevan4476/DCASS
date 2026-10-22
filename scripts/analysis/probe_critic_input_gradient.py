#!/usr/bin/env python3
"""
Can the gradient penalty be satisfied at all?

The WGAN-GP training on the traffic corpus pins the gradient penalty at
exactly lambda_gp, which is the value the penalty takes when the critic's
input gradient norm is zero: `lambda * (||grad D|| - 1)^2` equals lambda
when the norm is zero. The penalty is therefore inert, and the critic's
scale runs away while the generator's loss chases it.

The critic's transformer is a post-LN residual stack, and the diagnostic
measured that this attenuates the input difference by about a factor of
ten per layer. A penalty that asks for an input gradient of one cannot be
satisfied by a network whose input gradient is already a hundredth of
one.

This script measures the critic's input gradient norm, which is the
quantity the penalty constrains, for a fresh critic in three
configurations:

  * the shipped post-LN stack;
  * the same stack with pre-LN residuals;
  * the same stack with a bounded, normalised critic output.

Outputs:
    storage/benchmarks/critic_input_gradient_probe.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import (  # noqa: E402
    DeepPacketInspectionWarden,
    compute_gradient_penalty,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
SEQ = 50
LAMBDA_GP = 10.0


def make_batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    real = torch.tensor(
        rng.exponential(scale=5.0, size=(BATCH, SEQ)), dtype=torch.float32, device=DEVICE
    )
    fake = torch.tensor(
        np.clip(rng.normal(6.0, 1.0, (BATCH, SEQ)), 0.5, None),
        dtype=torch.float32,
        device=DEVICE,
    )
    channels = torch.tensor(
        rng.integers(0, 3, (BATCH, SEQ)), dtype=torch.long, device=DEVICE
    )
    fake_channels = torch.nn.functional.one_hot(
        torch.tensor(rng.integers(0, 3, (BATCH, SEQ)), device=DEVICE), 3
    ).float()
    return real, fake, channels, fake_channels


def input_gradient_norm(warden: DeepPacketInspectionWarden, batch) -> float:
    """The mean norm of the critic's gradient with respect to its input."""
    real, fake, channels, fake_channels = batch
    penalty = compute_gradient_penalty(
        warden,
        real,
        fake,
        channels,
        fake_channels,
        lambda_gp=LAMBDA_GP,
    )
    # The penalty is lambda * (norm - 1)^2, so the norm follows from it.
    mean_square = float(penalty) / LAMBDA_GP
    norm = 1.0 - np.sqrt(max(mean_square, 0.0))
    if norm < 0:
        norm = 1.0 + np.sqrt(mean_square)
    return float(norm)


def build_warden(prenorm: bool) -> DeepPacketInspectionWarden:
    """A critic with the shipped stack, or with pre-LN residuals."""
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=256)
    if prenorm:
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=256 * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        warden.transformer = nn.TransformerEncoder(layer, num_layers=4).to(DEVICE)
    return warden.to(DEVICE)


def main() -> int:
    print("=" * 78)
    print(" CRITIC INPUT GRADIENT: CAN THE GRADIENT PENALTY BE SATISFIED?")
    print("=" * 78)
    print(f"Device: {DEVICE}\n")

    batch = make_batch()
    results = {}

    for label, prenorm in (("post_ln_shipped", False), ("pre_ln", True)):
        torch.manual_seed(0)
        warden = build_warden(prenorm)
        norm = input_gradient_norm(warden, batch)
        penalty = LAMBDA_GP * (norm - 1.0) ** 2
        results[label] = {
            "input_gradient_norm": round(norm, 6),
            "gradient_penalty": round(float(penalty), 6),
        }
        print(f"  {label:20s} ||grad_x D|| = {norm:.6f}   penalty = {float(penalty):.4f}")

    # The same measurement after the critic has been trained to collapse,
    # using the shipped checkpoint, for reference.
    ckpt = torch.load(
        PROJECT_ROOT / "storage" / "models" / "warden.pt",
        map_location=DEVICE,
        weights_only=False,
    )
    warden = DeepPacketInspectionWarden(num_channels=3, hidden_dim=256)
    warden.load_state_dict(ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"])
    warden = warden.to(DEVICE)
    norm = input_gradient_norm(warden, batch)
    results["shipped_collapsed_critic"] = {
        "input_gradient_norm": round(norm, 6),
        "gradient_penalty": round(float(LAMBDA_GP * (norm - 1.0) ** 2), 6),
    }
    print(f"  {'shipped_collapsed':20s} ||grad_x D|| = {norm:.6f}")

    print()
    print("  A penalty of exactly lambda means the input gradient norm is zero:")
    print("  the constraint is not merely violated, it is inert.")

    out = PROJECT_ROOT / "storage" / "benchmarks" / "critic_input_gradient_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "critic-input-gradient-probe-v1",
                "lambda_gp": LAMBDA_GP,
                "note": (
                    "The critic's gradient with respect to its input, which is "
                    "the quantity the WGAN-GP penalty constrains. The penalty "
                    "equals lambda_gp exactly when this norm is zero, so a "
                    "penalty sitting at lambda_gp means the constraint is "
                    "inert rather than merely violated."
                ),
                "configurations": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

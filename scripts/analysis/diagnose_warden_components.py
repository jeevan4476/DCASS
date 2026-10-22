#!/usr/bin/env python3
"""
B2 diagnostic, stage 3: isolate WHICH component destroys the warden's
input sensitivity.

Stage 2 showed the shipped checkpoint attenuates any input difference by
roughly 10x per transformer layer, so the anomaly head reads an essentially
constant vector. Two candidate causes:

  (a) the anomaly head is untrained and stuck at its random init, and
  (b) the trained transformer is a constant-output shortcut.

This script separates them by swapping components:

  * shipped transformer + shipped anomaly head
  * shipped transformer + FRESH anomaly head
  * FRESH transformer + shipped anomaly head
  * FRESH transformer + FRESH anomaly head

and, for each, reports how much the anomaly score and the raw critic score
move between two maximally different delay sequences.

The gradient test in stage 1 showed the anomaly head receives no gradient
from the WGAN-GP objective, which predicts case 2 varies and case 1 does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402


def probes(n: int = 50):
    a = np.full(n, 3.0, dtype=np.float32)
    b = np.concatenate([np.full(n // 2, 0.5), np.full(n // 2, 60.0)]).astype(np.float32)
    return a, b


def measure(warden: DeepPacketInspectionWarden, a: np.ndarray, b: np.ndarray) -> dict:
    device = next(warden.parameters()).device
    c = np.zeros(len(a), dtype=np.int64)
    d_a = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    d_b = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
    c_t = torch.tensor(c, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        va = warden(d_a, c_t)
        vb = warden(d_b, c_t)
    return {
        "anomaly_a": float(va.anomaly_scores.mean()),
        "anomaly_b": float(vb.anomaly_scores.mean()),
        "anomaly_diff": float((va.anomaly_scores - vb.anomaly_scores).abs().max()),
        "raw_critic_a": float(va.feature_importance["raw_critic_score"]),
        "raw_critic_b": float(vb.feature_importance["raw_critic_score"]),
        "raw_critic_diff": float(
            (va.feature_importance["raw_critic_score"] - vb.feature_importance["raw_critic_score"]).abs().max()
        ),
        "bot_prob_a": float(va.bot_probability),
        "bot_prob_b": float(vb.bot_probability),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shipped_state = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in torch.load(
            PROJECT_ROOT / "storage" / "models" / "warden.pt",
            map_location=device,
            weights_only=False,
        ).items()
    }

    a, b = probes()
    report: dict = {"device": device, "cases": {}}

    def build(use_shipped_transformer: bool, use_shipped_anomaly: bool) -> DeepPacketInspectionWarden:
        torch.manual_seed(0)
        w = DeepPacketInspectionWarden(num_channels=3).to(device)
        if use_shipped_transformer:
            w.transformer.load_state_dict(
                {
                    k[len("transformer.") :]: v
                    for k, v in shipped_state.items()
                    if k.startswith("transformer.")
                }
            )
        if use_shipped_anomaly:
            w.anomaly_head.load_state_dict(
                {
                    k[len("anomaly_head.") :]: v
                    for k, v in shipped_state.items()
                    if k.startswith("anomaly_head.")
                }
            )
        w.eval()
        return w

    for st in (True, False):
        for sa in (True, False):
            key = f"transformer={'shipped' if st else 'fresh'}, anomaly={'shipped' if sa else 'fresh'}"
            w = build(st, sa)
            report["cases"][key] = measure(w, a, b)

    # --- gradient check on the shipped checkpoint --------------------------
    torch.manual_seed(0)
    g = DeepPacketInspectionWarden(num_channels=3).to(device)
    g.load_state_dict(shipped_state, strict=False)
    B, T = 8, 20
    real_d = torch.abs(torch.randn(B, T, device=device) * 5 + 10)
    real_c = torch.randint(0, 3, (B, T), device=device)
    fake_d = torch.abs(torch.randn(B, T, device=device) * 2 + 5)
    fake_c = torch.randint(0, 3, (B, T), device=device)
    from src.analysis.adversarial.warden import (
        compute_gradient_penalty,
        compute_warden_loss,
    )

    rv = g(real_d, real_c)
    fv = g(fake_d, torch.nn.functional.one_hot(fake_c, 3).float())
    total = compute_warden_loss(rv, fv) + compute_gradient_penalty(
        g, real_d, fake_d, real_c, torch.nn.functional.one_hot(fake_c, 3).float()
    )
    total.backward()
    no_grad = [n for n, p in g.named_parameters() if p.grad is None]
    report["anomaly_head_parameters_without_gradient"] = no_grad
    report["anomaly_head_has_any_gradient"] = any(
        "anomaly_head" in n for n in no_grad
    ) is False

    out_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "storage" / "benchmarks" / "warden_diagnostic_components.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'case':58s} {'anom_a':>9s} {'anom_b':>9s} {'anom_diff':>11s} {'critic_diff':>12s}")
    for key, m in report["cases"].items():
        print(
            f"{key:58s} {m['anomaly_a']:9.6f} {m['anomaly_b']:9.6f} "
            f"{m['anomaly_diff']:11.3e} {m['raw_critic_diff']:12.3e}"
        )
    print()
    print("anomaly-head parameters with NO gradient:", no_grad)
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

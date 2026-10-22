#!/usr/bin/env python3
"""
B2 diagnostic, stage 2: locate the exact layer where the warden's input
signal is lost.

Stage 1 established that the shipped warden produces the same anomaly score
and the same bot probability for six deliberately different delay sequences.
This script walks the layers and reports, for each, the spread of the
output across the six probes, so the first layer whose output stops varying
is the point of collapse.

Also inspects the checkpoint's weight statistics to see whether the collapse
is in the weights (a degenerate checkpoint) or in the forward pass (a bug).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from src.utils.checkpoint_io import load_trained_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import DeepPacketInspectionWarden  # noqa: E402


def probe_inputs(rng: np.random.Generator, n: int = 50):
    return {
        "constant_3s": (np.full(n, 3.0, dtype=np.float32), np.zeros(n, dtype=np.int64)),
        "jittered": (
            np.clip(3.0 + rng.normal(0, 1.5, n), 0.2, None).astype(np.float32),
            rng.integers(0, 3, n),
        ),
        "exponential": (
            rng.exponential(scale=3.0, size=n).astype(np.float32),
            rng.integers(0, 3, n),
        ),
        "bimodal": (
            np.concatenate([np.full(25, 0.5), np.full(25, 60.0)]).astype(np.float32),
            rng.integers(0, 3, n),
        ),
        "constant_0p5s": (np.full(n, 0.5, dtype=np.float32), np.zeros(n, dtype=np.int64)),
        "ramp": (np.linspace(0.5, 30.0, n).astype(np.float32), rng.integers(0, 3, n)),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warden = DeepPacketInspectionWarden(num_channels=3).to(device)
    ckpt_path = PROJECT_ROOT / "storage" / "models" / "warden.pt"
    state = load_trained_checkpoint(ckpt_path, map_location=device,)
    warden.load_state_dict(state, strict=False)
    warden.eval()

    report: dict = {"device": device, "checkpoint": str(ckpt_path)}

    # ---- weight statistics -------------------------------------------
    weight_report = {}
    for name, p in warden.named_parameters():
        weight_report[name] = {
            "shape": list(p.shape),
            "abs_mean": float(p.detach().abs().mean()),
            "abs_max": float(p.detach().abs().max()),
            "std": float(p.detach().std()) if p.numel() > 1 else 0.0,
            "all_zero": bool(torch.all(p == 0)),
        }
    report["weights"] = weight_report

    # ---- per-layer spread across probes ------------------------------
    rng = np.random.default_rng(0)
    probes = probe_inputs(rng)

    captured: dict[str, list[torch.Tensor]] = {}

    def record(tag: str, tensor: torch.Tensor) -> None:
        captured.setdefault(tag, []).append(tensor.detach().float().cpu().reshape(-1).clone())

    hooks = []

    def mk_hook(tag: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            record(tag, t)
        return hook

    for tag, module in [
        ("delay_embedding", warden.delay_embedding),
        ("input_projection", warden.input_projection),
        ("bilstm", warden.bilstm),
        ("transformer", warden.transformer),
        ("anomaly_head", warden.anomaly_head),
        ("classification_head", warden.classification_head),
    ]:
        hooks.append(module.register_forward_hook(mk_hook(tag)))

    per_probe = {}
    with torch.no_grad():
        for name, (delays, channels) in probes.items():
            d_t = torch.tensor(delays, dtype=torch.float32, device=device).unsqueeze(0)
            c_t = torch.tensor(channels, dtype=torch.long, device=device).unsqueeze(0)
            verdict = warden(d_t, c_t)
            per_probe[name] = {
                "bot_probability": float(verdict.bot_probability.item()),
                "anomaly_mean": float(verdict.anomaly_scores.mean().item()),
                "raw_critic": float(verdict.feature_importance["raw_critic_score"].item()),
            }

    for h in hooks:
        h.remove()

    layer_spread = {}
    for tag, tensors in captured.items():
        stacked = torch.stack(tensors)  # (n_probes, n_elements)
        # Spread of the per-probe MEAN: how much does the summary statistic
        # move between probes?
        means = stacked.mean(dim=1)
        # Spread of the per-probe STANDARD DEVIATION: how much does the
        # internal structure move?
        stds = stacked.std(dim=1)
        layer_spread[tag] = {
            "mean_of_means": float(means.mean()),
            "spread_of_means": float(means.max() - means.min()),
            "spread_of_stds": float(stds.max() - stds.min()),
            "max_abs_element_spread": float((stacked.max(dim=0).values - stacked.min(dim=0).values).max()),
        }

    report["per_probe"] = per_probe
    report["layer_spread"] = layer_spread

    # ---- where does the signal die? ----------------------------------
    collapse_point = None
    for tag in [
        "delay_embedding",
        "input_projection",
        "bilstm",
        "transformer",
        "anomaly_head",
        "classification_head",
    ]:
        s = layer_spread[tag]
        flat = s["spread_of_means"] < 1e-9 or s["max_abs_element_spread"] < 1e-9
        if flat and collapse_point is None:
            collapse_point = tag
    report["collapse_point"] = collapse_point

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "storage" / "benchmarks" / "warden_diagnostic_layers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Layer spread across the six probes:")
    print(f"{'layer':22s} {'spread(means)':>14s} {'spread(stds)':>14s} {'max elem spread':>16s}")
    for tag, s in layer_spread.items():
        print(
            f"{tag:22s} {s['spread_of_means']:14.6e} {s['spread_of_stds']:14.6e} "
            f"{s['max_abs_element_spread']:16.6e}"
        )
    print(f"\ncollapse point: {collapse_point}")

    print("\nKey weight statistics:")
    for name in [
        "anomaly_head.4.weight",
        "anomaly_head.4.bias",
        "classification_head.5.weight",
        "classification_head.5.bias",
        "transformer.layers.0.linear1.weight",
    ]:
        if name in weight_report:
            w = weight_report[name]
            print(
                f"  {name:36s} shape={w['shape']} abs_mean={w['abs_mean']:.6f} "
                f"abs_max={w['abs_max']:.6f} all_zero={w['all_zero']}"
            )
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

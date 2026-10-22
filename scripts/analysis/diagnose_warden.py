#!/usr/bin/env python3
"""
B2 diagnostic: why does the shipped warden return a constant anomaly score
and bot_probability = 0.0?

Runs the shipped warden checkpoint against several clearly different delay
sequences and reports, per input:
  * the mean anomaly score and its per-timestep spread
  * the bot probability and the raw critic score
  * intermediate activations (stat features, delay embedding, LSTM output,
    transformer output) and their spreads

If the warden's outputs are genuinely constant across very different
inputs, the diagnosis is a collapsed or degenerate checkpoint. If they vary
but the benchmark still records 0.62493, the bug is in the benchmark's use
of the model.

Writes a JSON report to the given path.
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


def build_probe_inputs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Delay sequences with deliberately different statistical character."""
    rng = np.random.default_rng(0)
    n = 50

    # 1. Perfect metronome, 3.0 s
    constant = np.full(n, 3.0, dtype=np.float32)

    # 2. Heavy jitter around 3.0 s
    jitter = (3.0 + rng.normal(0, 1.5, n)).astype(np.float32)
    jitter = np.clip(jitter, 0.2, None)

    # 3. Exponential (Poisson-like)
    poisson = rng.exponential(scale=3.0, size=n).astype(np.float32)

    # 4. Bimodal: 0.5 s bursts separated by 60 s pauses
    bimodal = np.concatenate([np.full(25, 0.5), np.full(25, 60.0)]).astype(np.float32)

    # 5. The 50 ms-scale constant the pre-remediation GAN path produced
    tight = np.full(n, 0.5, dtype=np.float32)

    # 6. Ramp
    ramp = np.linspace(0.5, 30.0, n).astype(np.float32)

    return {
        "constant_3s": (constant, np.zeros(n, dtype=np.int64)),
        "jittered_3s": (jitter, rng.integers(0, 3, n)),
        "exponential": (poisson, rng.integers(0, 3, n)),
        "bimodal_burst_pause": (bimodal, rng.integers(0, 3, n)),
        "constant_0p5s": (tight, np.zeros(n, dtype=np.int64)),
        "ramp": (ramp, rng.integers(0, 3, n)),
    }


def probe(warden: DeepPacketInspectionWarden, delays: np.ndarray, channels: np.ndarray) -> dict:
    device = next(warden.parameters()).device
    d_t = torch.tensor(delays, dtype=torch.float32, device=device).unsqueeze(0)
    c_t = torch.tensor(channels, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        stat = warden.extract_statistical_features(d_t)
        log_delays = torch.log(d_t + 1.0).unsqueeze(-1)
        delay_embed = warden.delay_embedding(log_delays)
        chan_embed = warden.channel_embedding(c_t)
        combined = torch.cat([delay_embed, chan_embed, stat.unsqueeze(1).repeat(1, d_t.size(1), 1)], dim=2)
        seq = warden.input_projection(combined)
        lstm_out, _ = warden.bilstm(seq)
        transformer_out = warden.transformer(lstm_out)
        anomaly = warden.anomaly_head(transformer_out).squeeze(-1)
        mean_pool = transformer_out.mean(dim=1)
        max_pool = transformer_out.max(dim=1)[0]
        global_features = torch.cat([mean_pool, max_pool], dim=1)
        raw_critic = warden.classification_head(global_features).squeeze(-1)
        verdict = warden(d_t, c_t)

    def stats(x: torch.Tensor) -> dict:
        flat = x.flatten().float()
        return {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        }

    return {
        "bot_probability": float(verdict.bot_probability.item()),
        "raw_critic_score": float(raw_critic.item()),
        "anomaly_mean": float(anomaly.mean().item()),
        "anomaly_std_over_time": float(anomaly.std().item()),
        "anomaly_min": float(anomaly.min().item()),
        "anomaly_max": float(anomaly.max().item()),
        "stat_features": stats(stat),
        "delay_embedding": stats(delay_embed),
        "lstm_output": stats(lstm_out),
        "transformer_output": stats(transformer_out),
        "anomaly_head_output": stats(anomaly),
        "classification_head_output": stats(raw_critic),
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "storage" / "benchmarks" / "warden_diagnostic.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    warden = DeepPacketInspectionWarden(num_channels=3).to(device)
    ckpt_path = PROJECT_ROOT / "storage" / "models" / "warden.pt"
    report: dict = {
        "device": device,
        "checkpoint": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
    }

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "warden_state" in ckpt:
            state = ckpt["warden_state"]
            report["checkpoint_kind"] = "training_checkpoint"
            report["checkpoint_epoch"] = ckpt.get("epoch")
            report["checkpoint_global_step"] = ckpt.get("global_step")
            report["checkpoint_metrics_tail"] = (
                ckpt.get("metrics_history", [])[-3:]
                if isinstance(ckpt.get("metrics_history"), list)
                else None
            )
        else:
            state = ckpt
            report["checkpoint_kind"] = "raw_state_dict"
        missing, unexpected = warden.load_state_dict(state, strict=False)
        report["missing_keys"] = list(missing)
        report["unexpected_keys"] = list(unexpected)
    else:
        report["checkpoint_kind"] = "random_init"
    warden.eval()

    probes = build_probe_inputs()
    report["probes"] = {}
    for name, (delays, channels) in probes.items():
        report["probes"][name] = probe(warden, delays, channels)

    anomalies = [p["anomaly_mean"] for p in report["probes"].values()]
    bots = [p["bot_probability"] for p in report["probes"].values()]
    report["summary"] = {
        "anomaly_mean_min": min(anomalies),
        "anomaly_mean_max": max(anomalies),
        "anomaly_spread": max(anomalies) - min(anomalies),
        "bot_probability_min": min(bots),
        "bot_probability_max": max(bots),
        "bot_probability_spread": max(bots) - min(bots),
        "anomaly_constant_across_inputs": (max(anomalies) - min(anomalies)) < 1e-6,
        "bot_probability_constant_across_inputs": (max(bots) - min(bots)) < 1e-6,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"checkpoint: {report['checkpoint_kind']} ({ckpt_path})")
    print(f"missing keys: {report.get('missing_keys')}")
    print(f"unexpected keys: {report.get('unexpected_keys')}")
    print()
    print(f"{'probe':24s} {'anomaly':>10s} {'anom_std':>10s} {'bot_p':>8s} {'raw_critic':>12s}")
    for name, p in report["probes"].items():
        print(
            f"{name:24s} {p['anomaly_mean']:10.6f} {p['anomaly_std_over_time']:10.3e} "
            f"{p['bot_probability']:8.4f} {p['raw_critic_score']:12.4f}"
        )
    print()
    print(json.dumps(report["summary"], indent=2))
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

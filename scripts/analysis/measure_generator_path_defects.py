#!/usr/bin/env python3
"""
Measure the generator path's two structural defects.

Both are recorded here because the diagnosis in
`docs/benchmarking/C3_GENERATOR_ADVERSARY.md` rests on them.

1. **The critic never sees the deployed traffic.** The trainer gives the
   critic the generator's raw output while the real traffic is in
   seconds, so the critic discriminates a unit range rather than a
   timing shape.

2. **The delay mapping gives a fresh generator no gradient.** The
   generator's delay head ends in a Softplus, so its raw output is
   unbounded, and the usable delay range is a narrow slice of it. Every
   output of a fresh generator lands at the mapping's upper clip, where
   the gradient is zero.

Outputs:
    storage/benchmarks/generator_path_defects.json
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

from src.stealth.gan.generator import TemporalPatternGenerator  # noqa: E402
from src.stealth.stealth_scheduler import (  # noqa: E402
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    GAN_DELAY_OFFSET,
    GAN_DELAY_SCALE,
    canonical_delay_mapping,
)

SEQ = 60
N = 256


def main() -> int:
    results: dict = {}

    # --- 1. The unit mismatch -------------------------------------------
    ckpt = load_trained_checkpoint(
        PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
        map_location="cpu",)
    shipped = TemporalPatternGenerator(
        latent_dim=128, hidden_dim=256, num_channels=3, max_sequence_length=SEQ
    )
    shipped.load_state_dict(ckpt["generator_state"])
    shipped.eval()
    z = torch.randn(N, 128)
    tod = torch.full((N,), 14.0)
    with torch.no_grad():
        schedule = shipped(z, SEQ, tod)
    raw = schedule.delays.numpy()
    mapped = canonical_delay_mapping(raw)

    corpus = json.loads(
        (PROJECT_ROOT / "storage" / "data" / "traffic" / "human_traffic.json").read_text()
    )
    real_delays = np.concatenate(
        [np.asarray(s["delays"], dtype=float) for s in corpus[:2000]]
    )

    results["unit_mismatch"] = {
        "note": (
            "The trainer gives the critic the generator's raw output and "
            "the corpus delays in seconds. The critic therefore never sees "
            "the traffic the scheduler sends."
        ),
        "critic_sees_real_seconds": {
            "min": round(float(real_delays.min()), 4),
            "p50": round(float(np.percentile(real_delays, 50)), 4),
            "max": round(float(real_delays.max()), 4),
        },
        "critic_sees_fake_raw": {
            "min": round(float(raw.min()), 6),
            "p50": round(float(np.median(raw)), 6),
            "max": round(float(raw.max()), 6),
        },
        "deployed_mapped_seconds": {
            "min": round(float(mapped.min()), 4),
            "p50": round(float(np.median(mapped)), 4),
            "max": round(float(mapped.max()), 4),
        },
        "shipped_raw_std": round(float(raw.std()), 6),
    }

    # --- 2. The mapping's conditioning ----------------------------------
    torch.manual_seed(42)
    fresh = TemporalPatternGenerator(
        latent_dim=128, hidden_dim=256, num_channels=3, max_sequence_length=SEQ
    )
    z = torch.randn(N, 128, requires_grad=True)
    schedule = fresh(z, SEQ, tod)
    raw_t = schedule.delays
    mapped_t = torch.clamp(
        (raw_t - 0.5) * GAN_DELAY_SCALE + GAN_DELAY_OFFSET, GAN_DELAY_MIN, GAN_DELAY_MAX
    )
    at_floor = (mapped_t <= GAN_DELAY_MIN + 1e-9).float()
    at_ceiling = (mapped_t >= GAN_DELAY_MAX - 1e-9).float()
    grad = torch.autograd.grad(mapped_t.sum(), raw_t, retain_graph=True)[0]

    usable_fraction = (GAN_DELAY_MAX - GAN_DELAY_OFFSET) / GAN_DELAY_SCALE
    results["mapping_conditioning"] = {
        "note": (
            "The generator's delay head ends in a Softplus, so its raw "
            "output is unbounded. The usable delay range corresponds to a "
            "narrow slice of it, and a fresh generator's every output "
            "lands at the upper clip, where the gradient is zero."
        ),
        "usable_raw_band": [
            round(0.5, 6),
            round(0.5 + usable_fraction, 6),
        ],
        "usable_band_as_pct_of_softplus_range": round(100 * usable_fraction, 4),
        "fresh_generator": {
            "raw_p50": round(float(raw_t.median()), 6),
            "mapped_at_floor_pct": round(100 * float(at_floor.mean()), 4),
            "mapped_at_ceiling_pct": round(100 * float(at_ceiling.mean()), 4),
            "zero_gradient_pct": round(100 * float((grad == 0).float().mean()), 4),
        },
    }

    out = PROJECT_ROOT / "storage" / "benchmarks" / "generator_path_defects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("Unit mismatch:")
    u = results["unit_mismatch"]
    print(
        f"  critic sees real: {u['critic_sees_real_seconds']['min']} to "
        f"{u['critic_sees_real_seconds']['max']} s"
    )
    print(
        f"  critic sees fake: {u['critic_sees_fake_raw']['min']:.4f} to "
        f"{u['critic_sees_fake_raw']['max']:.4f} (raw, never mapped)"
    )
    print(
        f"  deployed mapped : {u['deployed_mapped_seconds']['min']} to "
        f"{u['deployed_mapped_seconds']['max']} s"
    )
    m = results["mapping_conditioning"]
    f = m["fresh_generator"]
    print()
    print("Mapping conditioning:")
    print(f"  usable raw band {m['usable_raw_band']} = {m['usable_band_as_pct_of_softplus_range']}% of the range")
    print(f"  fresh generator: {f['mapped_at_ceiling_pct']}% at the ceiling, {f['zero_gradient_pct']}% zero gradient")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

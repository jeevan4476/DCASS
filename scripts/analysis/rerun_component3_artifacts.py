#!/usr/bin/env python3
"""
Re-run every artifact that depends on the shipped generator.

The generator was retrained after its three structural defects were
repaired, so the measurements that read its traffic have to be recorded
again against the new checkpoint. This script runs them in order and
reports which ones moved.

The order matters. Component 3 first, because the soft-branch design
comparison and the score distributions read the results it writes.

USAGE
-----
    .venv/bin/python scripts/analysis/rerun_component3_artifacts.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS = PROJECT_ROOT / "scripts" / "analysis"

#: The scripts, in dependency order, and the artifact each one writes.
STEPS: list[tuple[str, str]] = [
    ("verify_generator_path.py", "storage/benchmarks/generator_path_repair.json"),
    ("benchmark_3_traffic_evasion.py", "storage/benchmarks/component_3_results.json"),
    (
        "benchmark_3b_warden_score_distributions.py",
        "storage/benchmarks/component_3b_warden_score_summary.json",
    ),
    (
        "check_detector_generalization.py",
        "storage/benchmarks/traffic_detector_generalization.json",
    ),
    (
        "compare_soft_branch_designs.py",
        "storage/benchmarks/soft_branch_design_comparison.json",
    ),
    ("measure_ks_null.py", "storage/benchmarks/traffic_ks_null_band.json"),
    ("measure_rl_stealth_reward.py", "storage/benchmarks/rl_stealth_reward_probe.json"),
]


def snapshot(path: Path) -> dict | None:
    """A small, comparable summary of an artifact, for the before/after."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload


def summarise(payload: dict | None) -> str:
    """A one-line summary of the numbers that matter."""
    if payload is None:
        return "no artifact"
    if "schedulers" in payload:
        rows = []
        for row in payload["schedulers"]:
            rows.append(
                f"{row.get('scheduler')}: {row.get('interception_rate')}% "
                f"ks {row.get('ks_statistic')} lat {row.get('mean_latency')}"
            )
        return " | ".join(rows)
    return f"{len(payload)} keys"


def main() -> int:
    print("=" * 78)
    print(" RE-RUNNING THE GENERATOR-DEPENDENT ARTIFACTS")
    print("=" * 78)

    failures = []
    for script, artifact in STEPS:
        path = PROJECT_ROOT / artifact
        before = snapshot(path)
        print(f"\n--- {script} -> {artifact}")
        result = subprocess.run(
            [sys.executable, str(ANALYSIS / script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    FAILED (exit {result.returncode})")
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            failures.append(script)
            continue
        after = snapshot(path)
        print(f"    before: {summarise(before)}")
        print(f"    after : {summarise(after)}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All generator-dependent artifacts re-recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Dry-run the Component 3 benchmark into a scratch directory.

The benchmark writes to fixed paths under storage/benchmarks. This harness
imports it, redirects every output path into a scratch directory, runs it,
and prints the new scheduler table next to the archived pre-remediation
artifact so the difference is visible before anything is overwritten.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/opencode/c3_dryrun")
BENCH = PROJECT_ROOT / "storage" / "benchmarks"

spec = importlib.util.spec_from_file_location(
    "bench_c3", PROJECT_ROOT / "scripts" / "analysis" / "benchmark_3_traffic_evasion.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["bench_c3"] = mod
spec.loader.exec_module(mod)

(SCRATCH / "plots").mkdir(parents=True, exist_ok=True)
(SCRATCH / "tables").mkdir(parents=True, exist_ok=True)
mod.PLOTS_DIR = SCRATCH / "plots"
mod.TABLES_DIR = SCRATCH / "tables"
mod.BENCHMARK_DIR = SCRATCH

mod.run_component3()

new = json.loads((SCRATCH / "component_3_results.json").read_text())
old = json.loads(
    (BENCH / "archive" / "2026-10-21-pre-remediation" / "component_3_results.json").read_text()
)


def rows(result: dict) -> dict:
    return {s["scheduler"]: s for s in result["schedulers"]}


new_rows, old_rows = rows(new), rows(old)
print()
print("=" * 108)
print(f"{'scheduler':24s} {'int old':>8s} {'int new':>8s} {'KS old':>8s} {'KS new':>8s} "
      f"{'lat old':>8s} {'lat new':>8s} {'anom old':>9s} {'anom new':>9s}")
print("=" * 108)
for name in new_rows:
    o, n = old_rows.get(name, {}), new_rows[name]
    print(
        f"{name:24s} {o.get('interception_rate', float('nan')):8.1f} {n['interception_rate']:8.1f} "
        f"{o.get('ks_statistic', float('nan')):8.4f} {n['ks_statistic']:8.4f} "
        f"{o.get('mean_latency', float('nan')):8.2f} {n['mean_latency']:8.2f} "
        f"{o.get('mean_neural_anomaly', float('nan')):9.6f} {n['mean_neural_anomaly']:9.6f}"
    )
print()
print("rules:")
for name in new_rows:
    o = old_rows.get(name, {}).get("rules", [])
    n = new_rows[name]["rules"]
    from collections import Counter

    print(f"  {name:24s} old {dict(Counter(o))}  new {dict(Counter(n))}")
print()
print("diurnal:")
for d_old, d_new in zip(old.get("diurnal_breakdown", []), new.get("diurnal_breakdown", [])):
    print(
        f"  {d_new['regime']:24s} old mean_lat {d_old['mean_latency']:6.2f}  "
        f"new mean_lat {d_new['mean_latency']:6.2f}   "
        f"old raw {d_old['mean_raw']:.6f}  new raw {d_new['mean_raw']:.6f}"
    )
print()
print("human self-test:", old.get("human_self_test", {}).get("interception_rate"),
      "->", new.get("human_self_test", {}).get("interception_rate"))
print("scratch output:", SCRATCH)

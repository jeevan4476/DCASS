#!/usr/bin/env python3
"""
Component 6 LaTeX compile check.

The paper's LaTeX source is expected to live at the project root, because
the generated tables are included with root-relative paths such as
``\\input{storage/benchmarks/tables/table1_steganalysis.tex}``. This script
verifies that every generated table actually compiles under those exact
paths.

It stages the harness at the project root, compiles it, and reports the
result. The staged file and all build output are removed afterwards, so
nothing is left behind in the repository.

Usage:
    python scripts/analysis/check_latex_compile.py [--tectonic PATH]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS = PROJECT_ROOT / "docs" / "benchmarking" / "latex" / "tables_check.tex"
STAGED_NAME = "_dcass_tables_check.tex"

# The tables this check covers, in the order the harness includes them.
TABLES = [
    "table1_steganalysis.tex",
    "table2_distributional_drift.tex",
    "table2b_cluster_uniformity.tex",
    "table3_traffic_evasion.tex",
    "table3b_warden_score_quantiles.tex",
    "table4a_capacity.tex",
    "table4b_codec_robustness.tex",
    "table4c_ecc_failure_modes.tex",
    "table5_system_performance.tex",
]


def find_tectonic(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("tectonic")
    if found:
        return found
    for candidate in ("/tmp/opencode/tectonic",):
        if Path(candidate).exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", help="path to a tectonic binary")
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()

    if not HARNESS.exists():
        print(f"ERROR: harness missing at {HARNESS}")
        return 2

    tectonic = find_tectonic(args.tectonic)
    if tectonic is None:
        print("ERROR: no LaTeX engine found.")
        print("Install tectonic, or pass --tectonic /path/to/tectonic.")
        print("The check is skipped rather than reported as passing.")
        return 3

    missing = [
        t for t in TABLES if not (PROJECT_ROOT / "storage" / "benchmarks" / "tables" / t).exists()
    ]
    if missing:
        print("ERROR: these tables are referenced by the harness but not on disk:")
        for t in missing:
            print(f"  {t}")
        return 2

    staged = PROJECT_ROOT / STAGED_NAME
    build_dir = Path(tempfile.mkdtemp(prefix="dcass_texbuild_"))
    try:
        shutil.copyfile(HARNESS, staged)
        proc = subprocess.run(
            [
                tectonic,
                "--outdir",
                str(build_dir),
                "--keep-logs",
                str(staged),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        pdf = build_dir / f"{STAGED_NAME[:-4]}.pdf"
        ok = proc.returncode == 0 and pdf.exists()

        if ok:
            print(f"COMPILE OK: {len(TABLES)} tables, {pdf.stat().st_size} byte PDF")
        else:
            print("COMPILE FAILED")
            print("--- stdout ---")
            print(proc.stdout[-4000:])
            print("--- stderr ---")
            print(proc.stderr[-4000:])

        log = build_dir / f"{STAGED_NAME[:-4]}.log"
        if log.exists():
            errors = [
                line
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.startswith("!")
            ]
            if errors:
                print(f"--- {len(errors)} LaTeX error line(s) ---")
                for line in errors[:40]:
                    print(line)
        return 0 if ok else 1
    finally:
        staged.unlink(missing_ok=True)
        if args.keep_build:
            print(f"build output kept in {build_dir}")
        else:
            shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

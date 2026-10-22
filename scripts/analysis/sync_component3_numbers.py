#!/usr/bin/env python3
"""
Apply the recorded Component 3 numbers to the documents that quote them.

The generator is retrained from time to time, so the WGAN and Hybrid rows
in the recorded table change. The numbers are read from the artifact
rather than copied by hand.

The rows are located by their leading text rather than by their whole
line, so the script can be run repeatedly: it replaces whatever the row
currently says with what the artifact says, and it fails loudly if it
cannot find a row at all, because a document that has drifted is worse
than a loud error.

The prose that quotes the same numbers is not handled here. It is
edited by hand, because prose needs a sentence and not a number.

USAGE
-----
    .venv/bin/python scripts/analysis/sync_component3_numbers.py
"""

from __future__ import annotations

import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "storage" / "benchmarks" / "component_3_results.json"

#: Which row of which file is which scheduler's row. The key is a
#: substring that identifies the line, and the value is the line's
#: template.
ROWS: list[tuple[str, str, str]] = [
    (
        "docs/benchmarking/03_TRAFFIC_TIMING_EVASION.md",
        "| DCASS Temporal WGAN |",
        "| DCASS Temporal WGAN | {interception:.1f}% | {not_flagged:.1f}% | {entropy:.4f} | "
        "{ks:.4f} | {p} | {latency:.2f} | {cooldowns} | soft 50/50 |",
    ),
    (
        "docs/benchmarking/03_TRAFFIC_TIMING_EVASION.md",
        "| DCASS Hybrid Engine |",
        "| DCASS Hybrid Engine | {interception:.1f}% | {not_flagged:.1f}% | {entropy:.4f} | "
        "{ks:.4f} | {p} | {latency:.2f} | {cooldowns} | soft 50/50 |",
    ),
    (
        "docs/benchmarking/BENCHMARKING_REPORT.md",
        "| DCASS Temporal WGAN |",
        "| DCASS Temporal WGAN | {interception:.1f} | {not_flagged:.1f} | {entropy:.4f} | "
        "{ks:.4f} | {p} | {latency:.2f} |",
    ),
    (
        "docs/benchmarking/BENCHMARKING_REPORT.md",
        "| DCASS Hybrid Engine |",
        "| DCASS Hybrid Engine | {interception:.1f} | {not_flagged:.1f} | {entropy:.4f} | "
        "{ks:.4f} | {p} | {latency:.2f} |",
    ),
    (
        "docs/benchmarking/BENCHMARKING_REPORT.md",
        "DCASS Temporal WGAN &",
        "DCASS Temporal WGAN & {interception:.1f} & {not_flagged:.1f} & {entropy:.4f} & "
        "{ks:.4f} & {p} & {latency:.2f} & {cooldowns} & soft 50/50 \\\\",
    ),
    (
        "docs/benchmarking/BENCHMARKING_REPORT.md",
        "DCASS Hybrid Engine &",
        "DCASS Hybrid Engine & {interception:.1f} & {not_flagged:.1f} & {entropy:.4f} & "
        "{ks:.4f} & {p} & {latency:.2f} & {cooldowns} & soft 50/50 \\\\",
    ),
    (
        "docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md",
        "| DCASS Temporal WGAN |",
        "| DCASS Temporal WGAN | {interception:.1f}% | {not_flagged:.1f}% | {entropy:.4f} bits | "
        "{ks:.4f} | {p} | {latency:.2f} s | soft 50/50 |",
    ),
    (
        "docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md",
        "| DCASS Hybrid Engine |",
        "| DCASS Hybrid Engine | {interception:.1f}% | {not_flagged:.1f}% | {entropy:.4f} bits | "
        "{ks:.4f} | {p} | {latency:.2f} s | soft 50/50 |",
    ),
    (
        "docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md",
        "| DCASS Temporal WGAN |",
        "| DCASS Temporal WGAN | {interception:.1f}% | {ks:.4f} | {latency:.2f} s | "
        "{cooldowns} | {band} |",
    ),
    (
        "docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md",
        "| DCASS Hybrid Engine |",
        "| DCASS Hybrid Engine | {interception:.1f}% | {ks:.4f} | {latency:.2f} s | "
        "{cooldowns} | {band} |",
    ),
    (
        "docs/benchmarking/BENCHMARK_PAPER_WRITEUP.md",
        "| **DCASS WGAN** |",
        "| **DCASS WGAN** | **{interception:.1f}%**       | **{not_flagged:.1f}%**         | "
        "{entropy:.3f}                | {latency:.2f}           | soft 50/50 |",
    ),
    (
        "docs/benchmarking/BENCHMARK_PAPER_WRITEUP.md",
        "| **DCASS Hybrid** |",
        "| **DCASS Hybrid** | **{interception:.1f}%**     | **{not_flagged:.1f}%**         | "
        "{entropy:.3f}                | {latency:.2f}           | soft 50/50 |",
    ),
]

#: The rows that carry the two generators' KS distances in a different
#: shape.
KS_ROWS: list[tuple[str, str, str]] = [
    (
        "docs/modules/05_WGAN_GP_TRAFFIC_MIMICRY.md",
        "| Two-sample KS to surrogate |",
        "| Two-sample KS to surrogate | **{wgan_ks:.4f}** (WGAN), **{hybrid_ks:.4f}** (Hybrid), "
        "$p < 10^-4$ |",
    ),
]


def load_rows() -> dict:
    payload = json.loads(RESULTS.read_text())
    return {row["scheduler"]: row for row in payload["schedulers"]}


def fmt_p(value: float) -> str:
    """A p-value in the form the documents use.

    Anything below a thousandth is written in scientific notation, so a
    small p-value is never rounded to a flat zero.
    """
    if value == 0:
        return "0"
    if value < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}e{exponent}"
    return f"{value:.3f}"


def score_band(row: dict) -> str:
    scores = row.get("warden_scores") or [0.0]
    return f"{min(scores):.4f} to {max(scores):.4f}"


def main() -> int:
    rows = load_rows()
    wgan = rows["DCASS Temporal WGAN"]
    hybrid = rows["DCASS Hybrid Engine"]

    missing: list[str] = []
    applied = 0

    for path, marker, template in ROWS:
        file = PROJECT_ROOT / path
        lines = file.read_text().split("\n")
        found = False
        for index, line in enumerate(lines):
            if not line.startswith(marker):
                continue
            scheduler = wgan if "WGAN" in marker else hybrid
            lines[index] = template.format(
                interception=scheduler["interception_rate"],
                not_flagged=100 - scheduler["interception_rate"],
                entropy=scheduler["delay_entropy"],
                ks=scheduler["ks_statistic"],
                p=fmt_p(scheduler["ks_pvalue"]),
                latency=scheduler["mean_latency"],
                cooldowns=scheduler.get("cooldown_violations", ""),
                band=score_band(scheduler),
            )
            found = True
            applied += 1
        if not found:
            missing.append(f"{path}: {marker}")
        file.write_text("\n".join(lines))

    for path, marker, template in KS_ROWS:
        file = PROJECT_ROOT / path
        lines = file.read_text().split("\n")
        found = False
        for index, line in enumerate(lines):
            if not line.startswith(marker):
                continue
            lines[index] = template.format(
                wgan_ks=wgan["ks_statistic"], hybrid_ks=hybrid["ks_statistic"]
            )
            found = True
            applied += 1
        if not found:
            missing.append(f"{path}: {marker}")
        file.write_text("\n".join(lines))

    if missing:
        print("NOT FOUND:")
        for item in missing:
            print(f"  {item}")
        return 1

    print(f"Applied {applied} row replacements.")
    print(f"  WGAN  : {wgan['interception_rate']}%  ks {wgan['ks_statistic']:.4f}  "
          f"lat {wgan['mean_latency']:.2f}  scores {score_band(wgan)}")
    print(f"  Hybrid: {hybrid['interception_rate']}%  ks {hybrid['ks_statistic']:.4f}  "
          f"lat {hybrid['mean_latency']:.2f}  scores {score_band(hybrid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

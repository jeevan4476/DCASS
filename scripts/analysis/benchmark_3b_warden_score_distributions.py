#!/usr/bin/env python3
"""
Component 3b: Warden Score Distributions and the Two PPO Operating Points

Component 3 records interception rates per scheduler but not the shape of the
score distributions that produce them. This derived benchmark reads the stored
warden scores from component_3_results.json and the conservative baseline run,
summarizes their quantiles, and plots the distributions the interception rates
are thresholded from.

The flag threshold is read from the Component 3 artifact rather than assumed,
because the soft branch's threshold is calibrated on held-out human traffic
and changes when the detector or the corpus changes.

The conservative PPO row is a HISTORICAL artifact. It was recorded with the
superseded WGAN-GP critic (whose transformer had collapsed to a constant
output) and with a scheduler that did not yet clip the RL delays, so its
scores are on a different scale from the primary run's and its 44.07 s
mean latency is not reproducible with the current code. It is reported
against the 0.50 threshold that was in force when it was recorded, and it
is not a second operating point of the current pipeline.

Outputs:
    storage/benchmarks/component_3b_warden_score_summary.json
    storage/benchmarks/plots/fig_3b_warden_scores.png
    storage/benchmarks/tables/table3b_warden_score_quantiles.tex

Usage:
    .venv/bin/python scripts/analysis/benchmark_3b_warden_score_distributions.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
from datetime import datetime, timezone
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    recorded_commit,
    scale_table_to_textwidth,
    write_manifest,
)

# The flag threshold in force when the historical baseline run was recorded.
# The primary run's threshold is calibrated per run and is read from the
# Component 3 artifact.
FLAG_THRESHOLD = 0.50

COLORS = {
    "Static Cadence": "#7F8C8D",
    "Stochastic Poisson": "#E67E22",
    "DCASS Temporal WGAN": "#27AE60",
    "DCASS Hybrid Engine": "#16A085",
    "DCASS PPO RL Agent": "#8E44AD",
}
CONSERVATIVE_LABEL = "PPO RL Agent, conservative config"


@dataclass
class ScoreSummary:
    config: str
    scheduler: str
    num_trials: int
    minimum: float
    p10: float
    p25: float
    median: float
    mean: float
    p75: float
    p90: float
    maximum: float
    flagged: int
    interception_rate_pct: float
    mean_latency_s: float
    threshold: float


def summarize(
    config: str,
    scheduler: str,
    scores: List[float],
    latency: float,
    threshold: float = FLAG_THRESHOLD,
) -> ScoreSummary:
    """Summarize a stored score distribution.

    `threshold` is the flag threshold the scores were judged against. It is
    a parameter because the soft branch's threshold is calibrated per run
    and because the historical baseline was judged against the 0.50
    threshold that was in force when it was recorded.
    """
    arr = np.array(scores, dtype=np.float64)
    return ScoreSummary(
        config=config,
        scheduler=scheduler,
        num_trials=len(arr),
        minimum=float(arr.min()),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        mean=float(arr.mean()),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        maximum=float(arr.max()),
        flagged=int((arr >= threshold).sum()),
        interception_rate_pct=100.0 * float((arr >= threshold).sum()) / len(arr),
        mean_latency_s=float(latency),
        threshold=float(threshold),
    )


def collect_provenance() -> dict:
    """Record the code state that produced this summary.

    The table caption reads the commit back out of the artifact, so a table
    regenerated from a recorded artifact carries the commit that made the
    measurement rather than the commit that happens to be checked out.
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except OSError:
            return ""

    status = git("status", "--porcelain")
    return {
        "retrospective": False,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script": "scripts/analysis/benchmark_3b_warden_score_distributions.py",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    }


def primary_flag_threshold(primary: dict) -> float:
    """The flag threshold the primary run was judged against.

    The soft branch's threshold is calibrated on held-out human traffic, so
    it is recorded in the artifact rather than assumed. An artifact written
    before the soft branch was calibrated falls back to the historical 0.50.
    """
    calibration = primary.get("soft_branch_calibration") or {}
    threshold = calibration.get("threshold")
    if threshold is None:
        threshold = (primary.get("detector") or {}).get("flag_threshold")
    if threshold is None:
        return FLAG_THRESHOLD
    return float(threshold)


def load_summaries() -> List[ScoreSummary]:
    primary = json.loads((BENCHMARK_DIR / "component_3_results.json").read_text())
    baseline = json.loads((BENCHMARK_DIR / "component_3_results_baseline.json").read_text())

    threshold = primary_flag_threshold(primary)
    summaries: List[ScoreSummary] = []
    for s in primary["schedulers"]:
        summaries.append(
            summarize(
                "primary",
                s["scheduler"],
                s["warden_scores"],
                s.get("mean_latency", 0.0),
                threshold=threshold,
            )
        )
    for s in baseline["schedulers"]:
        if s["scheduler"] == "DCASS PPO RL Agent":
            # The baseline predates the calibrated soft branch, so it is
            # judged against the 0.50 threshold that was in force when it
            # was recorded. Its scores come from the superseded warden and
            # are on a different scale from the primary run's.
            summaries.append(
                summarize(
                    "historical",
                    CONSERVATIVE_LABEL,
                    s["warden_scores"],
                    s.get("mean_latency", 0.0),
                    threshold=FLAG_THRESHOLD,
                )
            )
    return summaries


def create_plot(summaries: List[ScoreSummary], output_path: Path):
    primary = [s for s in summaries if s.config == "primary"]
    conservative = [s for s in summaries if s.config == "historical"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    labels = [s.scheduler for s in primary] + [s.scheduler for s in conservative]
    positions = list(range(len(labels)))
    for pos, s in zip(positions, primary + conservative):
        color = COLORS.get(s.scheduler, "#8E44AD" if s.config == "historical" else "#34495E")
        # Reproduce the score cloud from quantiles is not possible; draw a box
        # from the stored quantiles instead: whiskers at min and max.
        ax1.plot(
            [pos, pos], [s.minimum, s.maximum], color=color, linewidth=1.6, alpha=0.85
        )
        ax1.plot(
            [pos - 0.18, pos + 0.18], [s.p25, s.p25], color=color, linewidth=2.4
        )
        ax1.plot(
            [pos - 0.18, pos + 0.18], [s.p75, s.p75], color=color, linewidth=2.4
        )
        ax1.plot(
            [pos - 0.22, pos + 0.22], [s.median, s.median], color=color, linewidth=3.2
        )
        ax1.scatter([pos], [s.mean], color=color, marker="D", s=42, zorder=5)
    for threshold, color, style, label in (
        (primary[0].threshold, "#E85D5D", "--", f"Flag threshold {primary[0].threshold:.3f}"),
        (conservative[0].threshold, "#E67E22", ":", f"Historical threshold {conservative[0].threshold:.2f}"),
    ):
        ax1.axhline(
            threshold, color=color, linestyle=style, linewidth=1.8, label=label,
        )
    ax1.set_xticks(positions)
    ax1.set_xticklabels(
        ["Static\nCadence", "Stochastic\nPoisson", "Temporal\nWGAN", "Hybrid\nEngine",
         "PPO\nfast", "PPO\nhistorical"],
        fontsize=9.5,
    )
    ax1.set_ylabel("Composite warden score", fontsize=11)
    ax1.set_title("Stored Score Quantiles per Scheduler", fontsize=13, fontweight="medium")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: zoomed comparison of the two PPO configurations' score bands
    fast = [s for s in primary if s.scheduler == "DCASS PPO RL Agent"][0]
    slow = conservative[0]
    for s, label, color in [
        (fast, f"PPO fast, {fast.mean_latency_s:.1f} s mean, {fast.interception_rate_pct:.0f} percent flagged", "#8E44AD"),
        (slow, f"PPO historical, {slow.mean_latency_s:.1f} s mean, {slow.interception_rate_pct:.0f} percent flagged", "#2E86AB"),
    ]:
        ax2.plot([0, 1], [s.p10, s.p90], color=color, linewidth=6, alpha=0.35)
        ax2.plot([0.02, 0.98], [s.median, s.median], color=color, linewidth=2.4)
        ax2.scatter([0.5], [s.mean], color=color, marker="D", s=42, zorder=5)
        ax2.errorbar(
            [0.5], [s.mean],
            yerr=[[abs(s.mean - s.p10)], [abs(s.p90 - s.mean)]],
            color=color, linewidth=1.4, capsize=6,
        )
        ax2.annotate(label, (0.5, s.p90 + 0.03), ha="center", fontsize=9.5, color=color)
    ax2.axhline(fast.threshold, color="#E85D5D", linestyle="--", linewidth=1.8)
    ax2.axhline(slow.threshold, color="#E67E22", linestyle=":", linewidth=1.8)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([])
    ax2.set_ylabel("Composite warden score", fontsize=11)
    ax2.set_title(
        "PPO Operating Points: Current and Historical",
        fontsize=13, fontweight="medium",
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved warden score plot to {output_path}")


def generate_latex_table(
    summaries: List[ScoreSummary], output_path: Path, commit: str | None = None
):
    provenance = caption_provenance(
        n=f"{len(summaries)} configurations x 50 launches",
        scope=(
            "stored per-trial composite warden scores; the historical row was "
            "scored by the superseded warden, see C3_WARDEN_DIAGNOSTIC.md"
        ),
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_3b_warden_score_summary.json"),
    )
    with open(output_path, "w") as f:
        f.write("% Component 3b: Warden score quantiles per scheduler\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(
            "\\caption{Stored composite warden score quantiles over 50 launches per scheduler. "
            "Interception is the share of scores at or above the row's flag threshold; "
            "the historical row was scored by the superseded warden and is on a "
            "different scale.%s}\n" % provenance
        )
        f.write("\\label{tab:warden_score_quantiles}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Scheduler & Min & P25 & Median & P75 & Max & Flagged \\\\\n\\midrule\n")
        for s in summaries:
            f.write(
                f"{s.scheduler} & {s.minimum:.3f} & {s.p25:.3f} & {s.median:.3f} & "
                f"{s.p75:.3f} & {s.maximum:.3f} & {s.flagged}/{s.num_trials} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        f.flush()
        scale_table_to_textwidth(output_path)
    print(f"Saved LaTeX table to {output_path}")


def main():
    summaries = load_summaries()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'Config':13s} {'Scheduler':38s} {'Min':>6s} {'Med':>6s} {'Max':>6s} {'Flagged':>9s}")
    for s in summaries:
        print(
            f"{s.config:13s} {s.scheduler:38s} {s.minimum:6.3f} {s.median:6.3f} "
            f"{s.maximum:6.3f} {s.flagged:5d}/{s.num_trials}"
        )

    plot_path = PLOTS_DIR / "fig_3b_warden_scores.png"
    create_plot(summaries, plot_path)

    table_path = TABLES_DIR / "table3b_warden_score_quantiles.tex"
    # These scores come from the Component 3 run, so its manifest records
    # the commit that produced the measurement.
    primary_manifest = BENCHMARK_DIR / "component_3_results_manifest.json"
    commit = None
    if primary_manifest.exists():
        commit = json.loads(primary_manifest.read_text()).get("git", {}).get("commit")
    generate_latex_table(summaries, table_path, commit=commit)

    output_json = BENCHMARK_DIR / "component_3b_warden_score_summary.json"
    with open(output_json, "w") as f:
        json.dump(
            {
                "provenance": collect_provenance(),
                "primary_flag_threshold": primary_flag_threshold(
                    json.loads((BENCHMARK_DIR / "component_3_results.json").read_text())
                ),
                "historical_flag_threshold": FLAG_THRESHOLD,
                "summaries": [asdict(s) for s in summaries],
                "plot_path": str(plot_path),
                "table_path": str(table_path),
            },
            f,
            indent=2,
        )
    print(f"Saved JSON summary to {output_json}")

    manifest = build_manifest(
        component="component_3b_warden_score_distributions",
        outputs=[output_json, table_path, plot_path],
        inputs=[
            BENCHMARK_DIR / "component_3_results.json",
            BENCHMARK_DIR / "component_3_results_baseline.json",
        ],
        seeds={
            "primary_flag_threshold": summaries[0].threshold if summaries else FLAG_THRESHOLD,
            "historical_flag_threshold": FLAG_THRESHOLD,
        },
        extra={
            "note": "Summarizes the stored per-trial composite warden scores from "
            "the primary Component 3 run and from the historical baseline run. "
            "The historical row was scored by the superseded WGAN-GP critic and "
            "by a scheduler that did not yet clip the RL delays, so it is on a "
            "different scale and is not a second operating point of the current "
            "pipeline. See docs/benchmarking/C3_WARDEN_DIAGNOSTIC.md.",
        },
    )
    manifest_path = write_manifest(
        manifest, BENCHMARK_DIR / "component_3b_warden_score_summary_manifest.json"
    )
    print(f"Saved run manifest to {manifest_path}")


if __name__ == "__main__":
    main()

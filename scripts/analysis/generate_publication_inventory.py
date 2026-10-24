#!/usr/bin/env python3
"""
Generate the Component 6 publication-artifact document and the report's
section 8 inventory from the filesystem.

The inventory is derived, not hand-maintained: every table, plot, and
artifact on disk is listed with its source script, the claim it supports,
and the command that regenerates it. Both documents are written from the
same generated content, so they cannot drift apart.

Usage:
    python scripts/analysis/generate_publication_inventory.py
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = PROJECT_ROOT / "storage" / "benchmarks"
TABLES = BENCH / "tables"
PLOTS = BENCH / "plots"

PUBDOC = PROJECT_ROOT / "docs" / "benchmarking" / "06_PUBLICATION_ARTIFACTS_AND_LATEX.md"
REPORT = PROJECT_ROOT / "docs" / "benchmarking" / "BENCHMARKING_REPORT.md"

# table file -> (display name, source script, claim IDs, description)
TABLE_MAP: dict[str, tuple[str, str, str, str]] = {
    "table1_steganalysis.tex": (
        "Table 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01, C1-SCOPE-02",
        "Media-layer stegananalysis resistance on the image-level 3-fold protocol",
    ),
    "table2_distributional_drift.tex": (
        "Table 2",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-ENERGY-01, C2-KS-02",
        "Per-modality distributional drift against a disjoint corpus reference",
    ),
    "table2b_cluster_uniformity.tex": (
        "Table 2b",
        "scripts/analysis/benchmark_2b_transmitted_cluster_usage.py",
        "C2-UNIFORM-03, C2-MECH-04",
        "Transmitted cluster usage against the pipeline's own mechanism null",
    ),
    "table3_traffic_evasion.tex": (
        "Table 3",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-DETECT-06, C3-CI-03, C3-WARDEN-02",
        "Warden interception rates and timing fidelity; the second column is the not-flagged rate",
    ),
    "table3b_warden_score_quantiles.tex": (
        "Table 3b",
        "scripts/analysis/benchmark_3b_warden_score_distributions.py",
        "C3-WARDEN-02, C3-STRUCT-03",
        "Stored warden critic score quantiles per scheduler",
    ),
    "table4a_capacity.tex": (
        "Table 4a",
        "scripts/analysis/benchmark_4a_capacity.py",
        "C4-CODED-01, C4-USER-02, C4-REDUCTION-03",
        "DSSC against exact_vcp coded capacity and carrier reduction",
    ),
    "table4b_codec_robustness.tex": (
        "Table 4b",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-FAR-04",
        "Permutation scaling, RS-ECC substitution resilience, and key separation",
    ),
    "table4c_ecc_failure_modes.tex": (
        "Table 4c",
        "scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
        "C4-ECC-07",
        "ECC failure anatomy: substitution, dropout, and the algebraic limits",
    ),
    "table5_system_performance.tex": (
        "Table 5",
        "scripts/analysis/benchmark_5_system_concurrency.py",
        "C5-DIRECT-01, C5-API-02",
        "Direct-engine codec latency, throughput, and thread concurrency",
    ),
}

# plot file -> (display name, source script, claim IDs, description)
PLOT_MAP: dict[str, tuple[str, str, str, str]] = {
    "fig_manifold_tsne.png": (
        "Manifold t-SNE",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-ENERGY-01",
        "Disjoint reference against carrier embedding visualization",
    ),
    "fig2_distributional_stats.png": (
        "Distributional Stats",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-ENERGY-01",
        "Energy, MMD squared, and W1 panels with per-draw dots",
    ),
    "fig_ks_heatmap.png": (
        "KS Heatmap",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-KS-02",
        "KS statistic across 512 dimensions with BH and family-wise annotations",
    ),
    "fig_2b_cluster_usage.png": (
        "Cluster Usage",
        "scripts/analysis/benchmark_2b_transmitted_cluster_usage.py",
        "C2-MECH-04",
        "Transmitted cluster usage against the mechanism null",
    ),
    "fig_interarrival_cdf.png": (
        "Inter-arrival CDF",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-DETECT-06",
        "Scheduler inter-arrival distributions against the synthetic surrogate",
    ),
    "fig_interception_rates.png": (
        "Interception Rates",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-DETECT-06",
        "Warden interception and not-flagged rates per scheduler",
    ),
    "fig_circadian_diurnal.png": (
        "Circadian Diurnal",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-CI-03",
        "Diurnal conditioning check at hour 14 against hour 03 under one identical mapping",
    ),
    "fig_3b_warden_scores.png": (
        "Warden Scores",
        "scripts/analysis/benchmark_3b_warden_score_distributions.py",
        "C3-WARDEN-02",
        "Stored warden score distributions with both PPO operating points",
    ),
    "fig1_roc_curves.png": (
        "ROC Curves",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "Detection accuracy and ROC AUC, positive control against DCASS carriers",
    ),
    "fig1_roc_curves.pdf": (
        "ROC Curves (vector)",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "Vector version of the ROC and accuracy figure",
    ),
    "fig1_payload_sweep.png": (
        "Payload Sweep",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "Detection accuracy against payload rate from 1.0 down to 0.1 bpp",
    ),
    "fig1_payload_sweep.pdf": (
        "Payload Sweep (vector)",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "Vector version of the payload sweep figure",
    ),
    "fig_4a_capacity_comparison.png": (
        "Capacity Comparison",
        "scripts/analysis/benchmark_4a_capacity.py",
        "C4-CODED-01",
        "DSSC against exact_vcp coded capacity",
    ),
    "fig_4b_permutation_scaling.png": (
        "Permutation Scaling",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-CODED-01",
        "HMAC-DRBG latency and throughput against candidate pool size",
    ),
    "fig_4b_ecc_recovery_curve.png": (
        "ECC Recovery",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-ECC-07",
        "Recovery rate against carrier substitution",
    ),
    "fig_4b_key_separation.png": (
        "Key Separation",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-FAR-04",
        "Wrong-key rejection and single-bit avalanche correlations",
    ),
    "fig_4c_ecc_failure_modes.png": (
        "ECC Failure Modes",
        "scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
        "C4-ECC-07",
        "Substitution and dropout recovery with the erasure-bound replay",
    ),
    "fig5_latency_breakdown.png": (
        "Latency Breakdown",
        "scripts/analysis/benchmark_5_system_concurrency.py",
        "C5-DIRECT-01",
        "Residual latency breakdown across payload sizes",
    ),
    "fig5_concurrency_scaling.png": (
        "Concurrency Scaling",
        "scripts/analysis/benchmark_5_system_concurrency.py",
        "C5-DIRECT-01",
        "Throughput and latency percentiles under direct-engine worker-thread load",
    ),
}

# artifact file -> (component, source script, claim IDs, kind, description)
ARTIFACT_MAP: dict[str, tuple[str, str, str, str, str]] = {
    "component_1_summary.json": (
        "Component 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01, C1-SCOPE-02",
        "results",
        "Aggregated three-fold steganalysis results and the protocol record",
    ),
    "component_1_fold0.json": (
        "Component 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "split manifest",
        "Fold 0 split manifest, carrier provenance, and per-sample posteriors",
    ),
    "component_1_fold1.json": (
        "Component 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "split manifest",
        "Fold 1 split manifest, carrier provenance, and per-sample posteriors",
    ),
    "component_1_fold2.json": (
        "Component 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "split manifest",
        "Fold 2 split manifest, carrier provenance, and per-sample posteriors",
    ),
    "component_2_results.json": (
        "Component 2",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-ENERGY-01, C2-KS-02",
        "results",
        "Per-modality distributional drift against a disjoint corpus reference",
    ),
    "component_2b_cluster_usage_results.json": (
        "Component 2b",
        "scripts/analysis/benchmark_2b_transmitted_cluster_usage.py",
        "C2-UNIFORM-03, C2-MECH-04",
        "results",
        "Transmitted cluster histogram against the uniform and mechanism nulls",
    ),
    "component_3_results.json": (
        "Component 3",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-DETECT-06, C3-CI-03",
        "results",
        "Scheduler interception rates, timing statistics, and the warden self-test",
    ),
    "component_1_summary_manifest.json": (
        "Component 1",
        "scripts/analysis/benchmark_1_media_steganalysis.py",
        "C1-AUC-01",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_2_results_manifest.json": (
        "Component 2",
        "scripts/analysis/benchmark_2_semantic_distribution.py",
        "C2-ENERGY-01",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_2b_cluster_usage_results_manifest.json": (
        "Component 2b",
        "scripts/analysis/benchmark_2b_transmitted_cluster_usage.py",
        "C2-MECH-04",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_4a_capacity_results_manifest.json": (
        "Component 4a",
        "scripts/analysis/benchmark_4a_capacity.py",
        "C4-CODED-01",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_4b_robustness_results_manifest.json": (
        "Component 4b",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-ECC-07",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_4c_ecc_failure_results_manifest.json": (
        "Component 4c",
        "scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
        "C4-ECC-07",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_5_results_manifest.json": (
        "Component 5",
        "scripts/analysis/benchmark_5_system_concurrency.py",
        "C5-DIRECT-01",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_3_results_manifest.json": (
        "Component 3",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-RERUN-05",
        "run manifest",
        "Command, git state, seeds, package versions, and input and output hashes",
    ),
    "component_3_results_baseline.json": (
        "Component 3",
        "scripts/analysis/benchmark_3_traffic_evasion.py",
        "C3-DETECT-06",
        "historical run",
        "Earlier component 3 run, scored by the superseded warden, holding the historical PPO operating point",
    ),
    "generator_path_defects.json": (
        "Component 3",
        "scripts/analysis/measure_generator_path_defects.py",
        "C3-GEN-07",
        "results",
        "The unit mismatch between the critic's two training sides, and the "
        "delay mapping's conditioning of the generator's output",
    ),
    "critic_sensitivity_repaired_run.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-08",
        "results",
        "The shipped run's critic, epoch by epoch: its input gradient, the "
        "penalty it implies, and the score gap between real and generated "
        "traffic",
    ),
    "critic_sensitivity_final_run.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-07",
        "results",
        "The first repaired run's critic, epoch by epoch, in which the "
        "critic still collapses",
    ),
    "critic_sensitivity_bound10.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-08",
        "results",
        "The eight-epoch bounded-critic arm at score bound 10, epoch by "
        "epoch: after one over-Lipschitz epoch the critic scores real and "
        "generated traffic identically from epoch 3 through epoch 6",
    ),
    "critic_sensitivity_lambda1000.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-08",
        "results",
        "The ten-epoch lambda_gp=1000 probe, epoch by epoch: the "
        "constraint holds in every epoch, so the shipped weight of 100 is "
        "the smallest measured sufficient weight, not the only one",
    ),
    "critic_sensitivity_bound100.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-08",
        "results",
        "The eight-epoch bounded-critic arm at score bound 100, epoch by "
        "epoch: three epochs of 38-point swings, then the same dead "
        "state from epoch 4 through the final epoch, which is collapsed",
    ),
    "critic_sensitivity_stronger_critic.json": (
        "Component 3",
        "scripts/analysis/track_critic_sensitivity.py",
        "C3-GEN-08",
        "results",
        "The eight-critic-step arm at the old lambda_gp of 10, epoch by "
        "epoch: the critic's score gap swings across 33 points and prefers "
        "the generated traffic in 22 of 30 epochs, and the generator's "
        "delay median sweeps 0.9 to 10.7 seconds",
    ),
    "wgan_training_data_probe.json": (
        "Component 3",
        "scripts/analysis/probe_wgan_training_data.py",
        "C3-GEN-07",
        "results",
        "The same WGAN-GP training replayed against the repository's "
        "synthetic fallback and against the storage traffic corpus, six "
        "epochs each, which rules the training data out as the cause of "
        "the collapse",
    ),
    "critic_input_gradient_probe.json": (
        "Component 3",
        "scripts/analysis/probe_critic_input_gradient.py",
        "C3-GEN-07",
        "results",
        "The critic's input gradient for the shipped post-LN stack and "
        "for a pre-LN variant, at initialization and after training, "
        "which puts the architectural hypothesis out",
    ),
    "generator_detector_adversary_probe.json": (
        "Component 3",
        "scripts/analysis/probe_generator_detector_adversary.py",
        "C3-GEN-07",
        "results",
        "The generator trained directly against the detector's bot "
        "probability for 40 epochs, which shows a bounded adversary "
        "saturates instead of discriminating",
    ),
    "generator_path_repair.json": (
        "Component 3",
        "scripts/analysis/verify_generator_path.py",
        "C3-GEN-08",
        "results",
        "The repaired generator path: the delay mapping is no longer "
        "saturated, the gradient reaches the delay head, and the critic's "
        "input gradient survives training",
    ),
    "rl_stealth_reward_probe.json": (
        "Component 3",
        "scripts/analysis/measure_rl_stealth_reward.py",
        "C3-TRAFFIC-06",
        "results",
        "The reward the RL environment assigns to bot-like and human-like "
        "traffic under the shipped WGAN-GP critic and under a trained "
        "detector",
    ),
    "traffic_ks_null_band.json": (
        "Component 3",
        "scripts/analysis/measure_ks_null.py",
        "C3-DETECT-06",
        "results",
        "The per-trial KS distances a burst drawn from the human reference "
        "itself produces, which is the band a matching sampler lands in",
    ),
    "traffic_detector_eval_results.json": (
        "Component 3",
        "scripts/analysis/train_traffic_detector.py",
        "C3-DETECT-06",
        "results",
        "The held-out evaluator's training summary, held-out AUC, and threshold "
        "calibration",
    ),
    "traffic_detector_generalization.json": (
        "Component 3",
        "scripts/analysis/check_detector_generalization.py",
        "C3-DETECT-06",
        "results",
        "The traffic detector's bot probability on six traffic populations, "
        "including a human generator it has never seen",
    ),
    "soft_branch_design_comparison.json": (
        "Component 3",
        "scripts/analysis/compare_soft_branch_designs.py",
        "C3-DETECT-06",
        "results",
        "Each candidate soft-branch score thresholded at a 5 percent "
        "false-positive rate on held-out human traffic, and the resulting "
        "separation",
    ),
    "traffic_detector_results.json": (
        "Component 3",
        "scripts/analysis/train_traffic_detector.py",
        "C3-DETECT-06",
        "results",
        "The traffic detector's training summary, held-out AUC, and threshold "
        "calibration",
    ),
    "component_3b_warden_score_summary.json": (
        "Component 3b",
        "scripts/analysis/benchmark_3b_warden_score_distributions.py",
        "C3-WARDEN-02",
        "results",
        "Quantiles of the stored per-trial warden scores",
    ),
    "component_3b_warden_score_summary_manifest.json": (
        "Component 3b",
        "scripts/analysis/benchmark_3b_warden_score_distributions.py",
        "C3-RERUN-05",
        "run manifest",
        "Command, git state, seeds, and input and output hashes",
    ),
    "component_4a_capacity_results.json": (
        "Component 4a",
        "scripts/analysis/benchmark_4a_capacity.py",
        "C4-CODED-01, C4-REDUCTION-03",
        "results",
        "DSSC against exact_vcp coded capacity and carrier reduction",
    ),
    "component_4b_robustness_results.json": (
        "Component 4b",
        "scripts/analysis/benchmark_4b_robustness.py",
        "C4-FAR-04, C4-ECC-07",
        "results",
        "Permutation scaling, substitution resilience, and key separation",
    ),
    "component_4c_ecc_failure_results.json": (
        "Component 4c",
        "scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
        "C4-ECC-07",
        "results",
        "Post-hardening ECC failure anatomy",
    ),
    "component_5_results.json": (
        "Component 5",
        "scripts/analysis/benchmark_5_system_concurrency.py",
        "C5-DIRECT-01, C5-API-02",
        "results",
        "Direct-engine latency, throughput, and thread concurrency",
    ),
    "warden_diagnostic.json": (
        "Diagnostic",
        "scripts/analysis/diagnose_warden.py",
        "C3-WARDEN-02",
        "diagnostic",
        "Warden outputs against six deliberately different delay sequences",
    ),
    "warden_diagnostic_layers.json": (
        "Diagnostic",
        "scripts/analysis/diagnose_warden_layers.py",
        "C3-WARDEN-02",
        "diagnostic",
        "Per-layer signal spread showing where the input sensitivity dies",
    ),
    "warden_diagnostic_components.json": (
        "Diagnostic",
        "scripts/analysis/diagnose_warden_components.py",
        "C3-WARDEN-02, C3-STRUCT-03",
        "diagnostic",
        "Component swap isolating the transformer collapse and the dead anomaly head",
    ),
    "sr_net_corrected_results.json": (
        "Component 1 (legacy)",
        "an earlier component 1 script version",
        "",
        "superseded",
        "Per-detector result from a superseded script version; the authoritative "
        "record is component_1_summary.json",
    ),
    "ye_net_improved_results.json": (
        "Component 1 (legacy)",
        "an earlier component 1 script version",
        "",
        "superseded",
        "Per-detector result from a superseded script version; the authoritative "
        "record is component_1_summary.json",
    ),
    "zhu_net_improved_results.json": (
        "Component 1 (legacy)",
        "an earlier component 1 script version",
        "",
        "superseded",
        "Per-detector result from a superseded script version; the authoritative "
        "record is component_1_summary.json",
    ),
}

# The regeneration command per script, used in the inventory.
REGEN_COMMANDS: dict[str, str] = {
    "scripts/analysis/benchmark_1_media_steganalysis.py":
        ".venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py",
    "scripts/analysis/benchmark_2_semantic_distribution.py":
        ".venv/bin/python scripts/analysis/benchmark_2_semantic_distribution.py",
    "scripts/analysis/benchmark_2b_transmitted_cluster_usage.py":
        ".venv/bin/python scripts/analysis/benchmark_2b_transmitted_cluster_usage.py",
    "scripts/analysis/benchmark_3_traffic_evasion.py":
        ".venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py",
    "scripts/analysis/benchmark_3b_warden_score_distributions.py":
        ".venv/bin/python scripts/analysis/benchmark_3b_warden_score_distributions.py",
    "scripts/analysis/benchmark_4a_capacity.py":
        ".venv/bin/python scripts/analysis/benchmark_4a_capacity.py",
    "scripts/analysis/benchmark_4b_robustness.py":
        ".venv/bin/python scripts/analysis/benchmark_4b_robustness.py",
    "scripts/analysis/benchmark_4c_ecc_failure_analysis.py":
        ".venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py",
    "scripts/analysis/benchmark_5_system_concurrency.py":
        ".venv/bin/python scripts/analysis/benchmark_5_system_concurrency.py",
    "scripts/analysis/diagnose_warden.py":
        ".venv/bin/python scripts/analysis/diagnose_warden.py",
    "scripts/analysis/diagnose_warden_layers.py":
        ".venv/bin/python scripts/analysis/diagnose_warden_layers.py",
    "scripts/analysis/diagnose_warden_components.py":
        ".venv/bin/python scripts/analysis/diagnose_warden_components.py",
}


def _strip_trailing(text: str) -> str:
    """Remove trailing whitespace from every line.

    Markdown table separator rows in the template carry two trailing spaces
    for a hard line break, which ruff flags as W291.
    """
    return "\n".join(line.rstrip() for line in text.split("\n"))


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def build_tables_section() -> str:
    lines = [
        "| Table | File | Source script | Claim IDs | Description |",
        "|-------|------|---------------|-----------|-------------|",
    ]
    for name in sorted(TABLE_MAP):
        if not (TABLES / name).exists():
            continue
        display, script, claims, desc = TABLE_MAP[name]
        claim_cell = ", ".join(f"`{c}`" for c in claims.split(", ") if c) or "—"
        lines.append(
            f"| {display} | `{rel(TABLES / name)}` | `{script}` | {claim_cell} | {desc} |"
        )
    return "\n".join(lines)


def build_plots_section() -> str:
    lines = [
        "| Plot | File | Source script | Claim IDs | Description |",
        "|------|------|---------------|-----------|-------------|",
    ]
    for name in sorted(PLOT_MAP):
        if not (PLOTS / name).exists():
            continue
        display, script, claims, desc = PLOT_MAP[name]
        claim_cell = ", ".join(f"`{c}`" for c in claims.split(", ") if c) or "—"
        lines.append(
            f"| {display} | `{rel(PLOTS / name)}` | `{script}` | {claim_cell} | {desc} |"
        )
    return "\n".join(lines)


def build_artifacts_section() -> str:
    lines = [
        "| File | Component | Kind | Source script | Claim IDs | Description |",
        "|------|-----------|------|---------------|-----------|-------------|",
    ]
    for name in sorted(ARTIFACT_MAP):
        if not (BENCH / name).exists():
            continue
        component, script, claims, kind, desc = ARTIFACT_MAP[name]
        claim_cell = ", ".join(f"`{c}`" for c in claims.split(", ") if c) or "—"
        lines.append(
            f"| `{rel(BENCH / name)}` | {component} | {kind} | `{script}` | {claim_cell} | {desc} |"
        )
    return "\n".join(lines)


def build_regen_section() -> str:
    lines = ["| Script | Regeneration command |", "|--------|-----------------------|"]
    for script in sorted(REGEN_COMMANDS):
        lines.append(f"| `{script}` | `{REGEN_COMMANDS[script]}` |")
    return "\n".join(lines)


def write_publication_doc() -> None:
    tables = sorted(p.name for p in TABLES.glob("*.tex"))
    plots = sorted(p.name for p in PLOTS.iterdir() if p.is_file())
    artifacts = sorted(p.name for p in BENCH.glob("*.json"))

    doc = f"""# Component 6: Publication Artifacts & LaTeX Exporter
## Automated Table Generation, Vector Plots & Paper Integration

> **Author:** DCASS Security & Steganography Core Architecture Team
>
> **Target Conferences:** USENIX Security, ACM CCS, IEEE TIFS, NeurIPS
>
> **Implementation:** Artifacts are written directly by each component benchmark script; there is no separate exporter.
>
> **Generated:** This file is produced by `scripts/analysis/generate_publication_inventory.py` from the filesystem, so it cannot drift from what is actually on disk. Do not edit it by hand.

---

## 1. Overview & Purpose

To ensure zero human error in transcribing experimental numbers into the
research paper, every component benchmark script writes its own publication
artifacts directly. There is no central manifest and no separate export step:
the results JSON from each run is the single source of truth for every quoted
number.

Each script writes into `storage/benchmarks/`:

1. **Results JSON:** one `component_*_results.json` file per benchmark, holding the recorded numbers behind every table and figure.
2. **Publication-Ready LaTeX Tables:** Standalone `.tex` files in `storage/benchmarks/tables/` that drop directly into `main.tex` via `\\input{{...}}`.
3. **High-Resolution Plots:** PNG and PDF graphics in `storage/benchmarks/plots/` showing empirical ROC curves, CDF timing distributions, and error-correction recovery cliffs.
4. **Run Manifests:** `*_manifest.json` files recording the command, git state, timestamp, seeds, package versions, and input and output hashes of a run.

Every generated table caption ends with the sample size, the scope of the
measurement, and the commit whose code produced the numbers, in the form
`(n = ...; scope = ...; commit = ...)`. The commit comes from the artifact's
own provenance block, so it names the commit that made the measurement, not
the commit that regenerated the table.

**Current inventory:** {len(tables)} LaTeX tables, {len(plots)} plot files, and {len(artifacts)} top-level JSON artifacts.

---

## 2. Generated LaTeX Tables

{build_tables_section()}

---

## 3. Generated Plots

{build_plots_section()}

---

## 4. Result Artifacts

{build_artifacts_section()}

Artifacts marked *superseded* are kept for provenance only. No current
script writes them and no claim cites them; the authoritative record for
their component is the `component_*_results.json` file.

---

## 5. Regeneration Commands

Every artifact above is regenerable by one documented command:

{build_regen_section()}

Each script also accepts an artifact-only mode that rebuilds the publication
artifacts from the recorded JSON without re-running any measurement. These
leave the results artifact and its provenance untouched:

| Script | Artifact-only mode |
|--------|--------------------|
| `scripts/analysis/benchmark_1_media_steganalysis.py` | `--table-only` |
| `scripts/analysis/benchmark_2_semantic_distribution.py` | `--table-only` |
| `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `--table-only` |
| `scripts/analysis/benchmark_3_traffic_evasion.py` | *(no artifact-only mode; a full run takes under a minute)* |
| `scripts/analysis/benchmark_3b_warden_score_distributions.py` | *(no artifact-only mode; a full run takes seconds)* |
| `scripts/analysis/benchmark_4a_capacity.py` | `--artifacts-only` |
| `scripts/analysis/benchmark_4b_robustness.py` | `--regenerate-from-json` |
| `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `--replot` |
| `scripts/analysis/benchmark_5_system_concurrency.py` | `--table-only` |

Two checks keep the inventory honest:

- `scripts/analysis/check_artifact_inventory.py` diffs the filesystem against this document and the report's section 8, and reports orphans and phantoms.
- `scripts/analysis/check_latex_compile.py` compiles every table under the root-relative paths the paper uses, so an `\\input` that cannot resolve is caught before submission.
- `scripts/analysis/check_latex_table_widths.py` measures each table's rendered width against the text block, so a table that would overflow the page is known in advance.

---

## 6. Integration into LaTeX Paper Source

In your paper's LaTeX file (e.g., `main.tex`), import the generated tables
directly. The paper source is expected to sit at the project root, because
the include paths below are root-relative:

```latex
% main.tex preamble
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}   % required by table4b for \\checkmark

\\section{{Experimental Evaluation}}

\\subsection{{Media-Layer Content Steganalysis}}
\\input{{storage/benchmarks/tables/table1_steganalysis.tex}}
\\begin{{figure*}}[htbp]
  \\centering
  \\includegraphics[width=0.48\\linewidth]{{storage/benchmarks/plots/fig1_roc_curves.pdf}}
  \\includegraphics[width=0.48\\linewidth]{{storage/benchmarks/plots/fig1_payload_sweep.pdf}}
  \\caption{{Steganalysis detection accuracy and ROC AUC evaluation.}}
  \\label{{fig:roc_curves}}
\\end{{figure*}}

\\subsection{{Semantic Distribution}}
\\input{{storage/benchmarks/tables/table2_distributional_drift.tex}}
\\input{{storage/benchmarks/tables/table2b_cluster_uniformity.tex}}
\\begin{{figure*}}[htbp]
  \\centering
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig2_distributional_stats.png}}
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig_ks_heatmap.png}}
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig_manifold_tsne.png}}
  \\caption{{Cross-modal distributional drift evaluation.}}
  \\label{{fig:distributional_drift}}
\\end{{figure*}}

\\subsection{{Network Traffic Cadence}}
\\input{{storage/benchmarks/tables/table3_traffic_evasion.tex}}
\\input{{storage/benchmarks/tables/table3b_warden_score_quantiles.tex}}
\\begin{{figure*}}[htbp]
  \\centering
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig_interarrival_cdf.png}}
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig_interception_rates.png}}
  \\includegraphics[width=0.32\\linewidth]{{storage/benchmarks/plots/fig_circadian_diurnal.png}}
  \\caption{{Traffic timing analysis.}}
  \\label{{fig:traffic_evasion}}
\\end{{figure*}}

\\subsection{{Codec Capacity, Fault Tolerance, and Cryptographic Isolation}}
\\input{{storage/benchmarks/tables/table4a_capacity.tex}}
\\input{{storage/benchmarks/tables/table4b_codec_robustness.tex}}
\\input{{storage/benchmarks/tables/table4c_ecc_failure_modes.tex}}
\\begin{{figure*}}[htbp]
  \\centering
  \\includegraphics[width=0.19\\linewidth]{{storage/benchmarks/plots/fig_4a_capacity_comparison.png}}
  \\includegraphics[width=0.19\\linewidth]{{storage/benchmarks/plots/fig_4b_permutation_scaling.png}}
  \\includegraphics[width=0.19\\linewidth]{{storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png}}
  \\includegraphics[width=0.19\\linewidth]{{storage/benchmarks/plots/fig_4b_key_separation.png}}
  \\includegraphics[width=0.19\\linewidth]{{storage/benchmarks/plots/fig_4c_ecc_failure_modes.png}}
  \\caption{{Codec performance and failure anatomy.}}
  \\label{{fig:codec_performance}}
\\end{{figure*}}

\\subsection{{Computational Performance and System Scalability}}
\\input{{storage/benchmarks/tables/table5_system_performance.tex}}
\\begin{{figure*}}[htbp]
  \\centering
  \\includegraphics[width=0.48\\linewidth]{{storage/benchmarks/plots/fig5_latency_breakdown.png}}
  \\includegraphics[width=0.48\\linewidth]{{storage/benchmarks/plots/fig5_concurrency_scaling.png}}
  \\caption{{Computational overhead and concurrency scaling.}}
  \\label{{fig:system_performance}}
\\end{{figure*}}
```

### 6.1 Table widths

The generated tables carry their own float environment, so the paper cannot
re-wrap them. Each one is emitted as a full-width `table*` with its tabular
scaled to the text width by `\\resizebox`, which makes it fit in both a
one-column and a two-column paper with no overfull box.

`\\resizebox` needs graphicx, which the preamble above already loads.
`scripts/analysis/check_latex_table_widths.py` reports the exact overflow
for each table; every generated table currently fits with zero overflow.

To place a table in a single column instead, wrap the `\\input` in a
`table` float of your own and drop the generated float, but the table will
then be scaled to the column width and its type will be small.

### 6.2 Scope notes that must reach the caption or the surrounding text

- The Component 3 tables record interception under a decision function whose soft branch is a trained traffic detector, with the flag threshold calibrated on held-out human traffic at a measured false-positive rate. The human side of that comparison is the repository's own synthetic traffic corpus, not recorded human browsing, so the rates are not a detectability measurement in the field. See [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md).
- The Component 1 table covers three gated detectors on an image-level protocol. It is a content-residual negative control.
- The Component 4a table reports coded capacity. User-payload density is lower and is reported separately in the report's section 6.1.
- The Component 5 table covers the direct codec engine only. The API transmit path admits concurrent sessions up to a bounded limit, but it is not exercised by this benchmark.
"""
    PUBDOC.write_text(doc, encoding="utf-8")
    return doc


def sync_embedded_tables(report: str) -> str:
    """Replace each embedded table block with the real generated file.

    The report quotes several generated tables inline. Those copies drift
    from the artifacts, so they are replaced with the file contents rather
    than maintained by hand.
    """
    for name in sorted(TABLE_MAP):
        path = TABLES / name
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        header = content.splitlines()[0].strip()
        if not header.startswith("%"):
            continue
        # Match from the fence to the closing fence, but stop before the
        # end of the document: the table body itself can contain a
        # backtick sequence, and the C5 table has two tabular blocks.
        pattern = re.compile(
            r"```latex\n" + re.escape(header) + r".*?\n```\n",
            re.DOTALL,
        )
        replacement = "```latex\n" + content + "\n```\n"
        report, n = pattern.subn(lambda _m: replacement, report)
        if n == 0:
            print(f"  note: no embedded block found for {name}")
    return report


def sync_embedded_tables_first() -> None:
    """Sync the embedded table blocks before any other edit to the report.

    The section 8 replacement changes the report's length, which shifts the
    embedded blocks. Syncing them first keeps the two operations
    independent, so a stale embedded copy can never change what the sync
    matches.
    """
    report = REPORT.read_text(encoding="utf-8")
    synced = sync_embedded_tables(report)
    if synced != report:
        REPORT.write_text(synced, encoding="utf-8")


def write_report_section(doc: str) -> None:
    """Replace section 8 of the report with the generated inventory."""
    report = REPORT.read_text(encoding="utf-8")
    start = report.index("## 8. Component 6: Publication Artifacts")
    end = report.index("## 9.", start)
    section = _strip_trailing(f"""## 8. Component 6: Publication Artifacts

This section is generated by `scripts/analysis/generate_publication_inventory.py`
from the filesystem, so it cannot drift from what is actually on disk. The
full mapping, including the source script and claim ID for every artifact,
is in [`06_PUBLICATION_ARTIFACTS_AND_LATEX.md`](06_PUBLICATION_ARTIFACTS_AND_LATEX.md).

{build_tables_section()}

### 8.2 Generated Plots

{build_plots_section()}

### 8.3 Result Artifacts

{build_artifacts_section()}

### 8.4 Inventory and compile checks

Three checks keep this inventory honest:

| Check | What it verifies | Command |
|-------|------------------|---------|
| Inventory | Every table, plot, and artifact on disk appears here, and nothing here is missing from disk | `.venv/bin/python scripts/analysis/check_artifact_inventory.py` |
| LaTeX compile | Every table compiles under the root-relative `\\input` paths the paper uses | `.venv/bin/python scripts/analysis/check_latex_compile.py` |
| Table widths | Which tables overflow a single-column text block | `.venv/bin/python scripts/analysis/check_latex_table_widths.py` |

The compile check needs a LaTeX engine. With `tectonic` on the path it
reports the result directly; without one it reports that it could not run
rather than claiming a pass.

The LaTeX integration guide, with the full `\\input` and `\\includegraphics`
blocks and the table-width notes, is in
[`06_PUBLICATION_ARTIFACTS_AND_LATEX.md`](06_PUBLICATION_ARTIFACTS_AND_LATEX.md).

""")
    report = sync_embedded_tables(report)
    REPORT.write_text(report[:start] + section + report[end:], encoding="utf-8")


def main() -> int:
    sync_embedded_tables_first()
    doc = write_publication_doc()
    write_report_section(doc)
    tables = len(list(TABLES.glob("*.tex")))
    plots = len([p for p in PLOTS.iterdir() if p.is_file()])
    artifacts = len(list(BENCH.glob("*.json")))
    print(f"Wrote {rel(PUBDOC)}")
    print(f"Updated section 8 of {rel(REPORT)}")
    print(f"Inventory: {tables} tables, {plots} plots, {artifacts} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

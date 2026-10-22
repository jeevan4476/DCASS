# Component 6: Publication Artifacts & LaTeX Exporter
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
2. **Publication-Ready LaTeX Tables:** Standalone `.tex` files in `storage/benchmarks/tables/` that drop directly into `main.tex` via `\input{...}`.
3. **High-Resolution Plots:** PNG and PDF graphics in `storage/benchmarks/plots/` showing empirical ROC curves, CDF timing distributions, and error-correction recovery cliffs.
4. **Run Manifests:** `*_manifest.json` files recording the command, git state, timestamp, seeds, package versions, and input and output hashes of a run.

Every generated table caption ends with the sample size, the scope of the
measurement, and the commit whose code produced the numbers, in the form
`(n = ...; scope = ...; commit = ...)`. The commit comes from the artifact's
own provenance block, so it names the commit that made the measurement, not
the commit that regenerated the table.

**Current inventory:** 9 LaTeX tables, 19 plot files, and 45 top-level JSON artifacts.

---

## 2. Generated LaTeX Tables

| Table | File | Source script | Claim IDs | Description |
|-------|------|---------------|-----------|-------------|
| Table 1 | `storage/benchmarks/tables/table1_steganalysis.tex` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01`, `C1-SCOPE-02` | Media-layer stegananalysis resistance on the image-level 3-fold protocol |
| Table 2 | `storage/benchmarks/tables/table2_distributional_drift.tex` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01`, `C2-KS-02` | Per-modality distributional drift against a disjoint corpus reference |
| Table 2b | `storage/benchmarks/tables/table2b_cluster_uniformity.tex` | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-UNIFORM-03`, `C2-MECH-04` | Transmitted cluster usage against the pipeline's own mechanism null |
| Table 3 | `storage/benchmarks/tables/table3_traffic_evasion.tex` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06`, `C3-CI-03`, `C3-WARDEN-02` | Warden interception rates and timing fidelity; the second column is the not-flagged rate |
| Table 3b | `storage/benchmarks/tables/table3b_warden_score_quantiles.tex` | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02`, `C3-STRUCT-03` | Stored warden critic score quantiles per scheduler |
| Table 4a | `storage/benchmarks/tables/table4a_capacity.tex` | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01`, `C4-USER-02`, `C4-REDUCTION-03` | DSSC against exact_vcp coded capacity and carrier reduction |
| Table 4b | `storage/benchmarks/tables/table4b_codec_robustness.tex` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04` | Permutation scaling, RS-ECC substitution resilience, and key separation |
| Table 4c | `storage/benchmarks/tables/table4c_ecc_failure_modes.tex` | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | ECC failure anatomy: substitution, dropout, and the algebraic limits |
| Table 5 | `storage/benchmarks/tables/table5_system_performance.tex` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01`, `C5-API-02` | Direct-engine codec latency, throughput, and thread concurrency |

---

## 3. Generated Plots

| Plot | File | Source script | Claim IDs | Description |
|------|------|---------------|-----------|-------------|
| Payload Sweep (vector) | `storage/benchmarks/plots/fig1_payload_sweep.pdf` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Vector version of the payload sweep figure |
| Payload Sweep | `storage/benchmarks/plots/fig1_payload_sweep.png` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Detection accuracy against payload rate from 1.0 down to 0.1 bpp |
| ROC Curves (vector) | `storage/benchmarks/plots/fig1_roc_curves.pdf` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Vector version of the ROC and accuracy figure |
| ROC Curves | `storage/benchmarks/plots/fig1_roc_curves.png` | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Detection accuracy and ROC AUC, positive control against DCASS carriers |
| Distributional Stats | `storage/benchmarks/plots/fig2_distributional_stats.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Energy, MMD squared, and W1 panels with per-draw dots |
| Concurrency Scaling | `storage/benchmarks/plots/fig5_concurrency_scaling.png` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Throughput and latency percentiles under direct-engine worker-thread load |
| Latency Breakdown | `storage/benchmarks/plots/fig5_latency_breakdown.png` | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Residual latency breakdown across payload sizes |
| Cluster Usage | `storage/benchmarks/plots/fig_2b_cluster_usage.png` | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-MECH-04` | Transmitted cluster usage against the mechanism null |
| Warden Scores | `storage/benchmarks/plots/fig_3b_warden_scores.png` | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02` | Stored warden score distributions with both PPO operating points |
| Capacity Comparison | `storage/benchmarks/plots/fig_4a_capacity_comparison.png` | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01` | DSSC against exact_vcp coded capacity |
| ECC Recovery | `storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-ECC-07` | Recovery rate against carrier substitution |
| Key Separation | `storage/benchmarks/plots/fig_4b_key_separation.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04` | Wrong-key rejection and single-bit avalanche correlations |
| Permutation Scaling | `storage/benchmarks/plots/fig_4b_permutation_scaling.png` | `scripts/analysis/benchmark_4b_robustness.py` | `C4-CODED-01` | HMAC-DRBG latency and throughput against candidate pool size |
| ECC Failure Modes | `storage/benchmarks/plots/fig_4c_ecc_failure_modes.png` | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Substitution and dropout recovery with the erasure-bound replay |
| Circadian Diurnal | `storage/benchmarks/plots/fig_circadian_diurnal.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-CI-03` | Diurnal conditioning check at hour 14 against hour 03 under one identical mapping |
| Inter-arrival CDF | `storage/benchmarks/plots/fig_interarrival_cdf.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Scheduler inter-arrival distributions against the synthetic surrogate |
| Interception Rates | `storage/benchmarks/plots/fig_interception_rates.png` | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Warden interception and not-flagged rates per scheduler |
| KS Heatmap | `storage/benchmarks/plots/fig_ks_heatmap.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-KS-02` | KS statistic across 512 dimensions with BH and family-wise annotations |
| Manifold t-SNE | `storage/benchmarks/plots/fig_manifold_tsne.png` | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Disjoint reference against carrier embedding visualization |

---

## 4. Result Artifacts

| File | Component | Kind | Source script | Claim IDs | Description |
|------|-----------|------|---------------|-----------|-------------|
| `storage/benchmarks/component_1_fold0.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 0 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_fold1.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 1 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_fold2.json` | Component 1 | split manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Fold 2 split manifest, carrier provenance, and per-sample posteriors |
| `storage/benchmarks/component_1_summary.json` | Component 1 | results | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01`, `C1-SCOPE-02` | Aggregated three-fold steganalysis results and the protocol record |
| `storage/benchmarks/component_1_summary_manifest.json` | Component 1 | run manifest | `scripts/analysis/benchmark_1_media_steganalysis.py` | `C1-AUC-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_2_results.json` | Component 2 | results | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01`, `C2-KS-02` | Per-modality distributional drift against a disjoint corpus reference |
| `storage/benchmarks/component_2_results_manifest.json` | Component 2 | run manifest | `scripts/analysis/benchmark_2_semantic_distribution.py` | `C2-ENERGY-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_2b_cluster_usage_results.json` | Component 2b | results | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-UNIFORM-03`, `C2-MECH-04` | Transmitted cluster histogram against the uniform and mechanism nulls |
| `storage/benchmarks/component_2b_cluster_usage_results_manifest.json` | Component 2b | run manifest | `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `C2-MECH-04` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_3_results.json` | Component 3 | results | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06`, `C3-CI-03` | Scheduler interception rates, timing statistics, and the warden self-test |
| `storage/benchmarks/component_3_results_baseline.json` | Component 3 | historical run | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-DETECT-06` | Earlier component 3 run, scored by the superseded warden, holding the historical PPO operating point |
| `storage/benchmarks/component_3_results_manifest.json` | Component 3 | run manifest | `scripts/analysis/benchmark_3_traffic_evasion.py` | `C3-RERUN-05` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_3b_warden_score_summary.json` | Component 3b | results | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-WARDEN-02` | Quantiles of the stored per-trial warden scores |
| `storage/benchmarks/component_3b_warden_score_summary_manifest.json` | Component 3b | run manifest | `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `C3-RERUN-05` | Command, git state, seeds, and input and output hashes |
| `storage/benchmarks/component_4a_capacity_results.json` | Component 4a | results | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01`, `C4-REDUCTION-03` | DSSC against exact_vcp coded capacity and carrier reduction |
| `storage/benchmarks/component_4a_capacity_results_manifest.json` | Component 4a | run manifest | `scripts/analysis/benchmark_4a_capacity.py` | `C4-CODED-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_4b_robustness_results.json` | Component 4b | results | `scripts/analysis/benchmark_4b_robustness.py` | `C4-FAR-04`, `C4-ECC-07` | Permutation scaling, substitution resilience, and key separation |
| `storage/benchmarks/component_4b_robustness_results_manifest.json` | Component 4b | run manifest | `scripts/analysis/benchmark_4b_robustness.py` | `C4-ECC-07` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_4c_ecc_failure_results.json` | Component 4c | results | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Post-hardening ECC failure anatomy |
| `storage/benchmarks/component_4c_ecc_failure_results_manifest.json` | Component 4c | run manifest | `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `C4-ECC-07` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/component_5_results.json` | Component 5 | results | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01`, `C5-API-02` | Direct-engine latency, throughput, and thread concurrency |
| `storage/benchmarks/component_5_results_manifest.json` | Component 5 | run manifest | `scripts/analysis/benchmark_5_system_concurrency.py` | `C5-DIRECT-01` | Command, git state, seeds, package versions, and input and output hashes |
| `storage/benchmarks/critic_input_gradient_probe.json` | Component 3 | results | `scripts/analysis/probe_critic_input_gradient.py` | `C3-GEN-07` | The critic's input gradient for the shipped post-LN stack and for a pre-LN variant, at initialization and after training, which puts the architectural hypothesis out |
| `storage/benchmarks/critic_sensitivity_bound10.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-epoch bounded-critic arm at score bound 10, epoch by epoch: after one over-Lipschitz epoch the critic scores real and generated traffic identically from epoch 3 through epoch 6 |
| `storage/benchmarks/critic_sensitivity_bound100.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-epoch bounded-critic arm at score bound 100, epoch by epoch: three epochs of 38-point swings, then the same dead state from epoch 4 through the final epoch, which is collapsed |
| `storage/benchmarks/critic_sensitivity_final_run.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-07` | The first repaired run's critic, epoch by epoch, in which the critic still collapses |
| `storage/benchmarks/critic_sensitivity_lambda1000.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The ten-epoch lambda_gp=1000 probe, epoch by epoch: the constraint holds in every epoch, so the shipped weight of 100 is the smallest measured sufficient weight, not the only one |
| `storage/benchmarks/critic_sensitivity_repaired_run.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The shipped run's critic, epoch by epoch: its input gradient, the penalty it implies, and the score gap between real and generated traffic |
| `storage/benchmarks/critic_sensitivity_stronger_critic.json` | Component 3 | results | `scripts/analysis/track_critic_sensitivity.py` | `C3-GEN-08` | The eight-critic-step arm at the old lambda_gp of 10, epoch by epoch: the critic's score gap swings across 33 points and prefers the generated traffic in 22 of 30 epochs, and the generator's delay median sweeps 0.9 to 10.7 seconds |
| `storage/benchmarks/generator_detector_adversary_probe.json` | Component 3 | results | `scripts/analysis/probe_generator_detector_adversary.py` | `C3-GEN-07` | The generator trained directly against the detector's bot probability for 40 epochs, which shows a bounded adversary saturates instead of discriminating |
| `storage/benchmarks/generator_path_defects.json` | Component 3 | results | `scripts/analysis/measure_generator_path_defects.py` | `C3-GEN-07` | The unit mismatch between the critic's two training sides, and the delay mapping's conditioning of the generator's output |
| `storage/benchmarks/generator_path_repair.json` | Component 3 | results | `scripts/analysis/verify_generator_path.py` | `C3-GEN-08` | The repaired generator path: the delay mapping is no longer saturated, the gradient reaches the delay head, and the critic's input gradient survives training |
| `storage/benchmarks/rl_stealth_reward_probe.json` | Component 3 | results | `scripts/analysis/measure_rl_stealth_reward.py` | `C3-TRAFFIC-06` | The reward the RL environment assigns to bot-like and human-like traffic under the shipped WGAN-GP critic and under a trained detector |
| `storage/benchmarks/soft_branch_design_comparison.json` | Component 3 | results | `scripts/analysis/compare_soft_branch_designs.py` | `C3-DETECT-06` | Each candidate soft-branch score thresholded at a 5 percent false-positive rate on held-out human traffic, and the resulting separation |
| `storage/benchmarks/sr_net_corrected_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |
| `storage/benchmarks/traffic_detector_eval_results.json` | Component 3 | results | `scripts/analysis/train_traffic_detector.py` | `C3-DETECT-06` | The held-out evaluator's training summary, held-out AUC, and threshold calibration |
| `storage/benchmarks/traffic_detector_generalization.json` | Component 3 | results | `scripts/analysis/check_detector_generalization.py` | `C3-DETECT-06` | The traffic detector's bot probability on six traffic populations, including a human generator it has never seen; the corpus "held-out" population is the checkpoint's own saved split |
| `storage/benchmarks/traffic_detector_results.json` | Component 3 | results | `scripts/analysis/train_traffic_detector.py` | `C3-DETECT-06` | The traffic detector's training summary, held-out AUC, and threshold calibration |
| `storage/benchmarks/traffic_ks_null_band.json` | Component 3 | results | `scripts/analysis/measure_ks_null.py` | `C3-DETECT-06` | Per-reference KS null bands: the per-trial distances a burst drawn from each reference's own pool produces (synthetic surrogate and corpus), which is the band a matching sampler lands in |
| `storage/benchmarks/warden_diagnostic.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden.py` | `C3-WARDEN-02` | Warden outputs against six deliberately different delay sequences |
| `storage/benchmarks/warden_diagnostic_components.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden_components.py` | `C3-WARDEN-02`, `C3-STRUCT-03` | Component swap isolating the transformer collapse and the dead anomaly head |
| `storage/benchmarks/warden_diagnostic_layers.json` | Diagnostic | diagnostic | `scripts/analysis/diagnose_warden_layers.py` | `C3-WARDEN-02` | Per-layer signal spread showing where the input sensitivity dies |
| `storage/benchmarks/wgan_training_data_probe.json` | Component 3 | results | `scripts/analysis/probe_wgan_training_data.py` | `C3-GEN-07` | The same WGAN-GP training replayed against the repository's synthetic fallback and against the storage traffic corpus, six epochs each, which rules the training data out as the cause of the collapse |
| `storage/benchmarks/ye_net_improved_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |
| `storage/benchmarks/zhu_net_improved_results.json` | Component 1 (legacy) | superseded | `an earlier component 1 script version` | — | Per-detector result from a superseded script version; the authoritative record is component_1_summary.json |

Artifacts marked *superseded* are kept for provenance only. No current
script writes them and no claim cites them; the authoritative record for
their component is the `component_*_results.json` file.

---

## 5. Regeneration Commands

Every artifact above is regenerable by one documented command:

| Script | Regeneration command |
|--------|-----------------------|
| `scripts/analysis/benchmark_1_media_steganalysis.py` | `.venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py` |
| `scripts/analysis/benchmark_2_semantic_distribution.py` | `.venv/bin/python scripts/analysis/benchmark_2_semantic_distribution.py` |
| `scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` | `.venv/bin/python scripts/analysis/benchmark_2b_transmitted_cluster_usage.py` |
| `scripts/analysis/benchmark_3_traffic_evasion.py` | `.venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py` |
| `scripts/analysis/benchmark_3b_warden_score_distributions.py` | `.venv/bin/python scripts/analysis/benchmark_3b_warden_score_distributions.py` |
| `scripts/analysis/benchmark_4a_capacity.py` | `.venv/bin/python scripts/analysis/benchmark_4a_capacity.py` |
| `scripts/analysis/benchmark_4b_robustness.py` | `.venv/bin/python scripts/analysis/benchmark_4b_robustness.py` |
| `scripts/analysis/benchmark_4c_ecc_failure_analysis.py` | `.venv/bin/python scripts/analysis/benchmark_4c_ecc_failure_analysis.py` |
| `scripts/analysis/benchmark_5_system_concurrency.py` | `.venv/bin/python scripts/analysis/benchmark_5_system_concurrency.py` |
| `scripts/analysis/diagnose_warden.py` | `.venv/bin/python scripts/analysis/diagnose_warden.py` |
| `scripts/analysis/diagnose_warden_components.py` | `.venv/bin/python scripts/analysis/diagnose_warden_components.py` |
| `scripts/analysis/diagnose_warden_layers.py` | `.venv/bin/python scripts/analysis/diagnose_warden_layers.py` |

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
- `scripts/analysis/check_latex_compile.py` compiles every table under the root-relative paths the paper uses, so an `\input` that cannot resolve is caught before submission.
- `scripts/analysis/check_latex_table_widths.py` measures each table's rendered width against the text block, so a table that would overflow the page is known in advance.

---

## 6. Integration into LaTeX Paper Source

In your paper's LaTeX file (e.g., `main.tex`), import the generated tables
directly. The paper source is expected to sit at the project root, because
the include paths below are root-relative:

```latex
% main.tex preamble
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}   % required by table4b for \checkmark

\section{Experimental Evaluation}

\subsection{Media-Layer Content Steganalysis}
\input{storage/benchmarks/tables/table1_steganalysis.tex}
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.48\linewidth]{storage/benchmarks/plots/fig1_roc_curves.pdf}
  \includegraphics[width=0.48\linewidth]{storage/benchmarks/plots/fig1_payload_sweep.pdf}
  \caption{Steganalysis detection accuracy and ROC AUC evaluation.}
  \label{fig:roc_curves}
\end{figure*}

\subsection{Semantic Distribution}
\input{storage/benchmarks/tables/table2_distributional_drift.tex}
\input{storage/benchmarks/tables/table2b_cluster_uniformity.tex}
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig2_distributional_stats.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_ks_heatmap.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_manifold_tsne.png}
  \caption{Cross-modal distributional drift evaluation.}
  \label{fig:distributional_drift}
\end{figure*}

\subsection{Network Traffic Cadence}
\input{storage/benchmarks/tables/table3_traffic_evasion.tex}
\input{storage/benchmarks/tables/table3b_warden_score_quantiles.tex}
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_interarrival_cdf.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_interception_rates.png}
  \includegraphics[width=0.32\linewidth]{storage/benchmarks/plots/fig_circadian_diurnal.png}
  \caption{Traffic timing analysis.}
  \label{fig:traffic_evasion}
\end{figure*}

\subsection{Codec Capacity, Fault Tolerance, and Cryptographic Isolation}
\input{storage/benchmarks/tables/table4a_capacity.tex}
\input{storage/benchmarks/tables/table4b_codec_robustness.tex}
\input{storage/benchmarks/tables/table4c_ecc_failure_modes.tex}
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4a_capacity_comparison.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_permutation_scaling.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_ecc_recovery_curve.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4b_key_separation.png}
  \includegraphics[width=0.19\linewidth]{storage/benchmarks/plots/fig_4c_ecc_failure_modes.png}
  \caption{Codec performance and failure anatomy.}
  \label{fig:codec_performance}
\end{figure*}

\subsection{Computational Performance and System Scalability}
\input{storage/benchmarks/tables/table5_system_performance.tex}
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.48\linewidth]{storage/benchmarks/plots/fig5_latency_breakdown.png}
  \includegraphics[width=0.48\linewidth]{storage/benchmarks/plots/fig5_concurrency_scaling.png}
  \caption{Computational overhead and concurrency scaling.}
  \label{fig:system_performance}
\end{figure*}
```

### 6.1 Table widths

The generated tables carry their own float environment, so the paper cannot
re-wrap them. Each one is emitted as a full-width `table*` with its tabular
scaled to the text width by `\resizebox`, which makes it fit in both a
one-column and a two-column paper with no overfull box.

`\resizebox` needs graphicx, which the preamble above already loads.
`scripts/analysis/check_latex_table_widths.py` reports the exact overflow
for each table; every generated table currently fits with zero overflow.

To place a table in a single column instead, wrap the `\input` in a
`table` float of your own and drop the generated float, but the table will
then be scaled to the column width and its type will be small.

### 6.2 Scope notes that must reach the caption or the surrounding text

- The Component 3 tables record interception under a decision function whose soft branch is a trained traffic detector, with the flag threshold calibrated on held-out human traffic at a measured false-positive rate. The human side of that comparison is the repository's own synthetic traffic corpus, not recorded human browsing, so the rates are not a detectability measurement in the field. See [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md).
- The Component 1 table covers three gated detectors on an image-level protocol. It is a content-residual negative control.
- The Component 4a table reports coded capacity. User-payload density is lower and is reported separately in the report's section 6.1.
- The Component 5 table covers the direct codec engine only. The API transmit path admits concurrent sessions up to a bounded limit, but it is not exercised by this benchmark.

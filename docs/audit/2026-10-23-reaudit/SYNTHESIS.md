# DCASS Post-Remediation Audit — Synthesis (re-run)

**Date:** 2026-10-23 · **Git:** clean tree, HEAD `0afd9a9` (six unpushed B2 commits on top of `28ff7d3`)
**Reports:** `semantic-c2-research.md`, `api-codec-infra-research.md`, `ganrl-c3-research.md`, `ganrl-c3-technical.md`
**Method:** Both analysts re-derived everything against the current tree (B1 security + B2 doc-truth batches already landed). Verdicts S/MF/Q/UB, priorities P0/P1/P2, and for code: CONFIRMED/RISK/DESIGN. No repo file was modified; frozen artifacts untouched; no number accepted from a doc without recomputation from its artifact ("inputs not stored" recorded as unverifiable rather than guessed).

## Tallies

| Report | Rows | S | MF/U B | UB | Q | P0 | P1 | P2 |
|---|---|---|---|---|---|---|---|---|
| semantic-c2-research | 17 | 14 | 2 | 0 | 1 | 0 | 0 | 3 |
| api-codec-infra-research | 28 | 15 | 3 | 7 | 0 | 3 | 3 | 8 |
| ganrl-c3-research | 44 | 30 | 3 | 1 | 10 | 0 | 1 | 13 |
| ganrl-c3-technical | 6 candidates + 7 findings | all 6 CONFIRMED | — | — | — | 0 | 1 | 4+3 latent/design notes |

## The one P1 code defect (was my #1, upgraded)

**RL export tail-pad is cooldown-blind and reachable today.** `stealth_scheduler.py:498-508` pads
timeout-truncated RL schedules with 0.5 s channel-0 delays (channel-0 cooldown 6 s). Reachability
proven: `/api/transmit` accepts arbitrary-length `media_ids` with `mode=rl`
(`server.py:939-945, 1133-1137`) and transmits the padded schedule verbatim (`server.py:1011-1063`);
`run_sender.py:436-442` builds arbitrary-length sequences; the pad fires on RL timeout (~350+
carriers at ~5.2 s/item), and a phantom un-backoffed delay from the timeout step is also appended
(`scheduler:492`). The pad block's own comment claims the tail "stays inside the contract"
(`:504-505`) — directly contradicting "guarantee zero rate-limit violations" (`:119, :340`).
**FIX-A:** on timeout, (a) drop the phantom step's delay, (b) finish remaining items through the
cooldown-aware router (reuse hybrid wait-fill `:373-386`), (c) repair the three claim surfaces.

## All six standing B3 candidates CONFIRMED

| # | Candidate | Fix |
|---|---|---|
| 1 (P1) | RL tail-pad cooldown-blind + phantom delay | FIX-A |
| 2 (P2) | Float32 violation counting (`benchmark_3:397-421`); 6.0→5.9999990 counted as violation; recorded Hybrid=13/RL=15 consistent with boundary erosion (`git show 9b02563` proves casts pre-existed at record time) | FIX-B: float64 scoring path, docs note that the recorded run measured in float32 |
| 3 (P2) | Defaults point at missing `data/human_traffic.json` (train_gan:44, evaluate_stealth:88-89, eval_warden:78-80); silent synthetic training, silent skipped human baseline; README:185 + QUICKSTART:213-215 advertise the suite that drops its baseline | FIX-C: repoint at `storage/data/traffic/` corpus + print "HUMAN BASELINE SKIPPED" |
| 4 (P2) | `evaluate_warden` scores zero-padded maskless batches (evaluate_stealth:40-49 vs 64+mask everywhere else) | FIX-D |
| 5 (P2) | Bounded-critic propagation: CLI final checkpoint omits keys (train_gan:162-171) + re-instantiators never call `apply_critic_config` (+ `verify_generator_path:193-194`, `track_critic_sensitivity:117-118` = N3 latent) | FIX-E (+ update tests/test_stealth/test_integration.py:49-60) |
| 6 (P2) | C1 gate computed, recorded, exit 0 on failure (no doc claims enforcement — rider confirmed) | FIX-F: `sys.exit(0 if all_gates_passed else 1)` |

## New technical findings (this re-run)

- **N1 (P2) probe labels lie:** "fixed 30 s" probe actually scores `[20.0]*20` post-clamp
  (`measure_rl_stealth_reward.py:34` vs `pacing.py:41-55`); the "3.0 s metronome" row is distorted by
  i%3 back-offs; ranking conclusion survives, labels don't ("guarantee zero" docstring folded in as
  N5). FIX-G + doc-line sync at `C3_TRAFFIC_DETECTOR_EVALUATION.md:32`.
- **N2 (P2):** dead `rate_limit_violation` retry branch (`scheduler:474-480`) + stale justification
  comments. FIX-H.
- **N4 (P2, latent):** `TimingSchedule.delays` docstring says "seconds", carries raw band values;
  `generate_stream` test-only today. FIX-I.
- **N7 (RISK):** all-masked fallback can pull the timeout path earlier than estimated — covered by
  FIX-A.
- **Notes (designs, no action except doc sync):** env mean-padding (P1-6), last-delay not slept,
  env timeout −50, select_threshold conservatism, baselines' cooldown counts recorded-not-scored.

## Doc-truth residuals (B2.5 candidates — docs only, no artifacts rewritten)

- **P1:** `C3_GENERATOR_ADVERSARY.md` §8 first-repaired-run table: 4 of 6 epoch rows disagree with
  `critic_sensitivity_final_run.json` (ep5 grad 3.0519→2.19, penalty 42.10→14.23; ep11 0.1113→2.04,
  7.90→10.90; ep17/21 cells; runs-table range "1e-4–3.05" → 4.5e-05–2.39). 12-of-30 collapse +
  end-collapsed conclusions are artifact-true — regenerate the cells only.
- **P1 (three P0-class doc command defects):** QUICKSTART `--speed` flag does not exist (actual
  `--speed-multiplier`); SCRIPTS.md `train_gan`/`train_rl` option sets wrong (three wrong/absent
  flags per trainer, wrong paths), and the "run without a pre-trained warden" recipe cannot work
  (`--warden-checkpoint` is required); the "generate_traffic_data" recipe names a script that isn't
  the real one (actual `scripts/stealth/generate_traffic_dataset.py`, different output schema).
- **P1:** QUICKSTART JWT guidance wrong on mechanism — dev fallback is a **fixed constant**
  `dcass-dev-secret-DO-NOT-USE-IN-PRODUCTION` (`src/api/auth.py`), not "random at startup".
- **P1:** `REAL_TRAFFIC_CAPTURE.md` corpus table mixes two files: χ² p=0.33 is the real-file-only
  value (pooled p=0.159654→0.16), median 5.5 is human-only (pooled 4.18), while floor 4.7% and max
  120.0 are pooled — no single artifact backs the table as written.
- **MF/P2:** C2 §4 names nonexistent symbol `MODALITY_SEQUENCE` (mechanism is the round-robin at
  `dssc_encoder.py:201`); `BENCHMARK_DEEP_EXPLANATION.md:346` analytic null center 0.0155 → 0.01526
  (5% critical value 0.0238 correct); §-level "projection KS" must say single pooled PC1.
- **MF/P2 rider:** ledger C3-PPO-06 cites `schedulers[3]` three times; PPO row is `schedulers[4]`
  (`schedulers[3]` = Hybrid). Values already correct; change the pointer only.
- **Q rows (leave recorded-open):** four self-labeled narrative runs, W §9.2's 214/300 + 0.16/0.83,
  warden.pt epoch-11/20 provenance, §4 stage-trace tables — no archived backing; no fix proposed
  under the no-fabrication rule.

## C2 bucket — CLOSED

The lost "C2 P1 follow-ups" are re-derived: **none exist** at P0/P1. All C2 headline/mechanism
numbers match (`component_2_results.json`, `component_2b_*`). Remaining C2 items are two MF doc
fixes and one Q wording (above). One code note flagged for the record: `dssc_encoder.py:228-229`
third fallback `all_ids[:256]` weakens topic-family purity when pools run thin (decodability
unaffected) — worth a limitation sentence in docs, not a behavior change.

## Batch mapping for the proposal

1. **B3 (code, needs approval):** FIX-A … FIX-F (one commit each) + FIX-G/H/I. Behavior deltas:
   exported schedules cooldown-safe; benchmark measures float64; defaults + loud skip; masked eval;
   critic-config propagation; C1 gate exit code; probe labels. Frozen artifacts stay frozen; doc
   numbers change only where the technical fix requires the wording to track it (§5 "guarantee zero",
   `:32` labels, float32 note).
2. **B2.5 (docs, B2-style):** §8 regeneration from artifact; QUICKSTART/SCRIPTS command cluster;
   JWT mechanism; REAL_TRAFFIC_CAPTURE table; C2 MFs; ledger `schedulers[4]` pointer.
3. **B4 (cosmetics):** prior ~61 P2s + 24 new (13 ganrl + 8 api-codec + 3 c2) P2 doc rows.
4. **Still open / by design (recorded):** real-traffic rerun (blocked on capture), four
   self-labeled narrative runs, warden.pt provenance, deployment-env severity, uv.lock intent,
   repaired-run λ_gp label misprint (artifact frozen).

## Standing rule surfaces that governed this audit

Every number above was recomputed from an artifact this session; unverifiable claims are recorded
as unverifiable, not corrected; audits document and propose — only explicit user approval turns
beans (B3/B2.5/B4) into commits.

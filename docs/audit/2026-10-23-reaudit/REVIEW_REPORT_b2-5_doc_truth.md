# Review Report — B2.5 Doc-Truth Repair Batch

**Branch:** `feature/gan-rl-pipeline` · **Merge-base:** `2e0f97b` · **Head:** `9919b32` · **Six commits:** 6653cd1, 656b752, d758880, d581e65, 835a76a, 9919b32  
**Reviewer role:** senior reviewer, read-only. Nothing under `storage/` was written; no tests run; `--help` probes and source reads only.

## Verdict: **Fix-Required**

One Important finding (F-1) remains: an internal inconsistency in a displayed formula constant left by a partial fix in Task 5. The batch is **not mergeable as-is**; F-1 is actionable in a single one-line fix pass. No Critical findings. No Minors identified.

---

## Method / evidence checked

For each of the seven touched docs I re-derived every number the batch wrote from exactly one artifact or verified source:

| Task | Commit | Doc(s) | Backing evidence checked |
|---|---|---|---|
| 1 | 6653cd1 | `docs/benchmarking/C3_GENERATOR_ADVERSARY.md` | `storage/benchmarks/critic_sensitivity_final_run.json` (30-epoch history, `collapsed_epochs`, `final_epoch_collapsed`); `critic_sensitivity_repaired_run.json`; `critic_sensitivity_lambda1000.json`; `critic_input_gradient_probe.json` |
| 2 | 656b752 | `docs/QUICKSTART.md`, `docs/guides/SCRIPTS.md` | Live argparse via `--help` (venv) for `train_gan.py`, `train_rl.py`, `eval_warden.py`, `generate_traffic_dataset.py`, `run_pipeline.py`, `run_sender.py`, `run_receiver.py`, `evaluate_stealth.py` (with `PYTHONPATH=.`), `docker_orchestrate.py`; source reads for flag defaults; `docker-compose.yml` services/profiles/env; `Makefile` targets; `src/api/server.py` `speed_multiplier` field |
| 3 | d758880 | `docs/QUICKSTART.md` | `src/api/auth.py` (:31 `sha256_crypt`, :50–55 production fail-fast, :55 fixed dev constant) |
| 4 | d581e65 | `docs/benchmarking/REAL_TRAFFIC_CAPTURE.md` | `storage/data/traffic/real_human_traffic.json` (10,000 sessions), `storage/data/traffic/human_traffic.json` (2,000 sessions); recomputed chi-square (24 bins, df=23), floor, max, reuse-gap |
| 5 | 835a76a | `docs/benchmarking/02_SEMANTIC_SELECTION_BIAS.md`, `docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md` | `storage/benchmarks/component_2_results.json` (`headline.*.n_carriers_slots` 9250/9500/9100, reference 5000); `src/engine/dssc_encoder.py` (:173, :201) |
| 6 | 9919b32 | `docs/benchmarking/CLAIM_LEDGER.md` | `storage/benchmarks/component_3_results.json` `schedulers[]` index 4 = PPO |

All six commits touch only their designated docs; commit style matches history (`fix(docs): …`). Suite held 412 passed / 39 skipped on every task (per the SDD ledger); docs-only, no ruff changes.

---

## Findings

### Critical — none.

### Important — F-1

**File:** `docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md:343`  
**Commit:** `835a76a` (Task 5)  
**Focus mode broken:** #3 — internal inconsistency after a partial fix (corrected prose whose displayed formula still returns a different value). Also violates the builder-check amendment: "formula constants corrected alongside prose"; "the 5% critical 1.3581 constant ... stay[s]".

**What the doc shows:**
```
E[D_KS] ≈ sqrt(pi/2)·ln 2 · sqrt((n+m)/(nm)) ≈ 0.8687 sqrt((n+m/(nm)),
D_KS^0.05 ≈ 1.36 sqrt((n + m)/(nm))
```
and :346 quotes per-coordinate critical values `0.0238 / 0.0237 / 0.0239` (image/text/audio; n = 9250/9500/9100, m = 5000).

**Recomputed from the artifact n/m:**
- Constant `1.36` → image 0.023872 → **0.0239**, text 0.023762 → **0.0238**, audio 0.023941 → **0.0239**. This contradicts the quoted prose `0.0238 / 0.0237 / 0.0239` for image and text.
- Constant `1.3581` → image **0.0238**, text **0.0237**, audio **0.0239** — matches the prose exactly (and matches the audit's own computation: 1.3581 × √((n+m)/(nm)) = 0.023862 for n=9200/m=5000).
- Center constant `0.8687` and prose centers `0.0152 / 0.0152 / 0.0153` are correct and were verified (√(π/2)·ln2 = 0.868731…; per-modality centers 0.015249/0.015178/0.015293).

**Required fix (one line, one fix pass):** replace `1.36` with `1.3581` at `BENCHMARK_DEEP_EXPLANATION.md:343` so the displayed formula reproduces the quoted critical values. Prose values stay untouched (they are artifact-true).

### Minor — none identified.

---

### Declined-to-judge lines — rulings

**DJ-1 — Pooled same-channel reuse-gap median `4.18` (old `REAL_TRAFFIC_CAPTURE.md` cell).**  
**Ruling: recorded UNVERIFIABLE; omitted from the corrected table; no fix possible in this batch.** The Task 4 escalation trigger fired: the pinned recipe (delay of the second of each chronologically adjacent same-channel event pair) recomputes ≈4.2–4.3 pooled (human-alone ≈5.5 → matches the doc's 5.5 s cell), and no principled variant matches 4.18. The 4.18 comes from the audit session's unarchived computation; no recipe is recorded anywhere in the repo.  
**Evidence that would decide it:** the audit's original computation script/recipe, or a human ruling pinning a reproducible recipe (bounded: the value must lie between the per-file medians). Until then it stays out of the docs (recorded for B4 reverse-derivation). This is the Task 4 mid-task escalation already ruled by the human.

**DJ-2 — C3 runs-table `lambda_gp = 100` label for "the shipped run" row (`C3_GENERATOR_ADVERSARY.md:259`).**  
**Ruling: the behavioral values are artifact-true and stay; the `lambda_gp = 100` label is accepted as inherited from the run path `/home/img1/gan_runs/l100`; the artifact's own embedded `lambda_gp` field (10.0) is a known frozen misprint.** Measured ranges verify: repaired run (`critic_sensitivity_repaired_run.json`) 0 collapsed epochs, grad 0.108–0.782, gap 0.327–19.398 → doc's `0.11 to 0.78 / 0.33 to 19.40` ✓; lambda1000 probe grad 0.421–0.915, gap −3.0471–2.1333 → doc's `0.42 to 0.92 / −3.0 to 2.1` ✓. The label misprint is recorded-open by design (design §Non-goals: "repaired-run λ_gp label misprint with frozen artifact"); the artifact is frozen, so no doc change is required or possible.

**DJ-3 — Docker Compose dynamic execution (SCRIPTS.md / QUICKSTART.md §11).**  
**Ruling: verified statically, not executed.** Service names (`dcass-sender`, `dcass-receiver`, `dcass-gen-traffic`, `dcass-train-gan`, `dcass-train-rl`), profiles (`training`, `web`), env defaults (`DCASS_MODE=static`, `DCASS_SEQ_LENGTH=20`, `GAN_EPOCHS=50`, `RL_EPISODES=1000`), and all `make docker-*` targets are present in `docker-compose.yml` and `Makefile` exactly as documented. Actual `docker compose` execution was out of scope (no daemon; executor recorded the same limitation). Not a docs-truth defect.

**DJ-4 — `evaluate_stealth.py` requires `PYTHONPATH=.` (QUICKSTART.md §9, SCRIPTS.md §5).**  
**Ruling: documented and verified.** The doc's heads-up and the `PYTHONPATH=.` prefix are correct; `--help` with `PYTHONPATH=.` confirms `--gan-checkpoint`/`--rl-checkpoint` exist, and the executor's P3 probe ran exit 0. Not a defect.

---

## Numbers I could not re-derive (explicitly recorded)

Exactly one number in the changed-docs corpus fails the re-derivation rule: **the pooled same-channel reuse-gap median `4.18`** previously shown in `REAL_TRAFFIC_CAPTURE.md` (Task 4). It exists in no artifact and no recorded recipe reproduces it from `storage/data/traffic/{real_human_traffic,human_traffic}.json`; the corrected table omits it (human-alone cell `5.5 s` kept, backed by the pinned recipe) and records it as unverifiable for B4 (DJ-1).

Every other number in the six commits' changed docs was re-derived from its artifact/source and verified: all 13 per-epoch C3 table cells + both runs-table ranges (Task 1); every command flag/path/default in QUICKSTART/SCRIPTS (Task 2); the JWT dev constant, production fail-fast, and `sha256_crypt` (Task 3); corpus-table cells 0.33/0.16/120.0/60.0/4.7%/5.5 and session counts 10,000/2,000 (Task 4); KS centers `0.0152/0.0152/0.0153`, criticals `0.0238/0.0237/0.0239`, center constant `0.8687` (Task 5); and ledger pointer `schedulers[4]` → PPO row with interception_rate 0.0, warden_scores 0.0040–0.0047, cooldown_violations 15, threshold 0.0181, n_calibration 457, measured_fpr 4.60% (Task 6).

Also confirmed absent from the changed docs (per the design's "not transcribed" / "dropped" rulings): the `(text-first with text ctx offset)` shorthand, the `MODALITY_SEQUENCE` symbol, `0.01713`, `1.2533`, `--speed`, `generate_traffic_data.py`, `demo_dcass.py`, and any `bcrypt` claim; no example writes into `storage/data/traffic/`.

---

## Conclusion for the executor

Apply one fix: `BENCHMARK_DEEP_EXPLANATION.md:343` `1.36` → `1.3581`. Then the batch satisfies the Review Focus on every changed cell/sentence and is mergeable. The push/B4 decision (including DJ-1's `4.18` reverse-derivation and the sibling claim instances recorded for B4: `BENCHMARKING_REPORT.md:276–279`, `BENCHMARK_PAPER_WRITEUP_PLAN.md:116`, `BENCHMARK_PAPER_WRITEUP.md:83`) belongs to the human.

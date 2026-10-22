# DCASS Remediation Plan

**Status:** All three phases complete, plus a follow-on phase that repaired the RL stealth pipeline and re-measured C3 against a held-out detector. Phase 0 (docs-only truth pass), Phase 1 (B1, B2, B1-b, B4, B6, B7, P1-10, and the final verification), and Phase 2 (Component 6 inventory and LaTeX reconciliation, and the final claim sweep). The frozen state is tagged `benchmark-freeze-2026-10-21`. The follow-on work is recorded in `C3_TRAFFIC_DETECTOR_EVALUATION.md`. One item remains open and is a data problem, not a code problem: a real human traffic baseline for C3. See section 11.

**Branch:** `feature/gan-rl-pipeline`  
**Baseline commit:** `0c24ca8`  
**Plan date:** 2026-10-21  
**Target completion:** 2026-10-28  
**Venue:** Not yet selected  
**Component 6 scope:** Inventory and LaTeX reconciliation only. Camera-ready paper work is deferred until paper writing begins.

---

## 1. Locked decisions

1. **Docs-only truth pass first.** No behavior-changing code, benchmark methodology, or result artifacts are modified until the Phase 0 review.
2. **C3 is a negative/limited result by default.** The `0%` evasion figure this plan started from was structurally guaranteed by the soft-branch cap and could not be presented as evasion. A timeboxed warden diagnostic may improve the result, but no positive evasion claim is made without a functional warden and a defensible baseline.
3. **C1 is restricted to three detectors:** SRNet (the implemented deep residual detector), ZhuNet, and YeNet. StegExpose, Xu-Net, and classical SRM are not part of the recorded C1 claim. The user confirmed that "SRM" means SRNet.
4. **Credentials are test-only.** The database may be reset. Credential cleanup is low priority and is not a release blocker.
5. **Reruns are allowed.** Existing artifacts are archived before any overwrite. Every new run records provenance.
6. **Component 6 is inventory/LaTeX reconciliation only.** A full camera-ready artifact pass happens later, during paper writing.
7. **One-week deadline.** A full C3 rebuild with a real human baseline is out of scope unless the required data already exists and the work fits the schedule.

---

## 2. Definition of done

The week is successful when:

- Every headline number in `README.md`, the benchmarking report, the paper writeup, module docs, and the UI is traceable to a recorded artifact, or is explicitly labeled as theory/design intent.
- C3 is presented as a negative/limited result with confidence intervals and a clear explanation of the soft cap, collapsed warden, and synthetic baseline — or as a repaired result if the timeboxed diagnostic succeeds.
- C1 reports only SRNet, ZhuNet, and YeNet, with actual measured AUCs and scope limits.
- C4 distinguishes coded capacity from user-payload capacity.
- C5 is scoped to the direct codec engine, with the API's bounded concurrency stated.
- C2 reports the actual multiple-testing outcome and does not claim uniform indistinguishability.
- Component 6 inventory matches the filesystem and report section 8.
- `ruff check src tests` passes and the non-integration test suite passes.
- All changed work is committed per task; nothing is pushed.
- A final claim sweep finds no unqualified "perfect," "0%," "100%," "all detectors," or "zero synthetic" claims.

---

## 3. Phase 0 — Documentation truth pass

**Goal:** Remove or qualify every claim the current artifacts do not support. No reruns in this phase.

### 3.1 Claim ledger

Create `docs/benchmarking/CLAIM_LEDGER.md` as the single source of truth for numerical and security claims.

Required columns:

| Column | Meaning |
|---|---|
| `id` | Stable claim ID, e.g. `C3-EVASION-01` |
| `claim` | Exact claim text or short paraphrase |
| `location` | File and section |
| `artifact` | JSON/table/plot path |
| `field` | JSON pointer or table row |
| `command` | Reproduction command |
| `commit` | Artifact-producing commit |
| `n` | Sample size |
| `scope` | What the result does and does not cover |
| `status` | `supported`, `qualified`, `retracted`, `theory`, `pending-rerun` |
| `notes` | Caveats and follow-up |

Every Phase 0 edit references a ledger ID. No new number enters a document without a ledger row.

### 3.2 Task list

| ID | Task | Primary files | Acceptance |
|---|---|---|---|
| P0-1 | Build claim ledger skeleton and populate all headline claims | `docs/benchmarking/CLAIM_LEDGER.md` | Every README/report/paper headline has a row and a status |
| P0-2 | Rewrite README empirical and setup claims | `README.md` | Benchmark table matches artifacts; "50 client threads at 100% request success" removed or scoped; credentials corrected; MIT claim reconciled with repository state |
| P0-3 | Fix theory and security framing | `docs/benchmarking/README.md` | Remove "perfectly secure," achieved `D_KL = 0`, Nash/game-theoretic overclaim, and "zero synthetic mocking"; add selection/timing limits |
| P0-4 | Reconcile report executive summary with body | `docs/benchmarking/BENCHMARKING_REPORT.md` | Executive table no longer conflicts with technical sections; C2/C3/C4/C5 scope boxes added |
| P0-5 | Reconcile paper writeup | `docs/benchmarking/BENCHMARK_PAPER_WRITEUP.md` | Abstract, results, and limitations match artifacts; audio provenance corrected to `libretta` (13,496 clips); C1 detector list corrected; C3 CIs added |
| P0-6 | Rewrite stale WGAN module doc | `docs/modules/05_WGAN_GP_TRAFFIC_MIMICRY.md` | Remove unsupported 10k-real/17-epoch/0.4999/0.0214 claims; fix or remove `benchmark_gan_timing.py` reference |
| P0-7 | Add as-run headers and limitations to C1–C5 docs | `docs/benchmarking/0{1..5}_*.md` | Each states command, commit, n, artifact, date, and limitations; C3 soft cap is prominent; C1 audio probe labeled exploratory; C4 coded/user labels; C5 API limitation |
| P0-8 | Correct UI copy | `frontend/src/app/login/page.tsx`, `frontend/src/app/send/page.tsx`, showcase components | Demo credentials match the chosen canonical set; Auto mode is not labeled "best available" while hardcoding `exact_vcp`; showcase claims are scoped |
| P0-9 | Resolve low-risk repository mismatches | `README.md`, `LICENSE`, Docker files | MIT license added or README claim corrected; missing Docker training script either referenced correctly or the service disabled/documented |
| P0-10 | Run the Phase 0 claim sweep | All docs and UI | No unqualified banned phrases; ledger has no unexplained `pending-rerun` rows for headline claims |

### 3.3 Phase 0 exit criteria

- Claim ledger complete for all externally visible claims.
- README, benchmarking overview, report executive summary, paper writeup, module doc, C1–C5 docs, and UI are mutually consistent.
- Credentials documented correctly for a fresh test database.
- No file in `storage/benchmarks/` has been overwritten.
- Working tree committed; no push.

### 3.4 Phase 0 completion record (2026-10-21)

Phase 0 was completed as a docs-and-copy-only pass. No file under `storage/benchmarks/` was touched, and no file under `src/`, `tests/`, or `scripts/` was modified.

| Task | Result | Notes |
|---|---|---|
| P0-1 | Done | `docs/benchmarking/CLAIM_LEDGER.md` created with 26 claim rows, each carrying artifact, field, command, status, and scope |
| P0-2 | Done | README table now reports recorded values with scope; "50 client threads at 100% request success" replaced with the direct-engine result; credentials corrected; the MIT license claim now matches the added `LICENSE` file |
| P0-3 | Done | "Perfectly secure", achieved `D_{KL}=0`, Nash equilibrium, and "zero synthetic mocking" removed; selection and timing limits added |
| P0-4 | Done | Executive summary now separates coded from user payload, marks traffic evasion NOT DEMONSTRATED, and scopes the concurrency row to the direct engine |
| P0-5 | Done | Audio provenance corrected to `libretta` (13,496 clips); C1 detector list reduced to SRNet/ZhuNet/YeNet; C3 confidence intervals and cap disclosure added; `Evasion Rate` columns renamed to not-flagged |
| P0-6 | Done | Module 05 section 5 rewritten as an as-run record; the 10k/17-epoch/0.4999/0.0214 claims and the missing `benchmark_gan_timing.py` reference are withdrawn |
| P0-7 | Done | As-run headers with scope added to C1–C5; C3 soft cap, C4 coded/user split, and the C5 API limitation are stated in the headers |
| P0-8 | Done | Demo credentials corrected in the login page and QUICKSTART; "best available model" removed; the hardcoded `exact_vcp` codec is now disclosed in the UI; showcase claims scoped |
| P0-9 | Done | The MIT `LICENSE` file was added on 2026-10-21 on explicit user confirmation, and the README points to it. The Docker training service is documented as not runnable instead of removed, so no deployment behavior changed |
| P0-10 | Done | Sweep run across `README.md`, `docs/`, and `frontend/src/`. Remaining "perfect" and "0%" occurrences are scoped, definitional, or inside explicit retraction notes |

Additional findings discovered during the sweep and folded into the same pass:

- `docs/modules/07_PPO_REINFORCEMENT_LEARNING_SCHEDULER.md` carried an unsupported "Production Verified" status with 49.39% warden bot probability, 98.88% delivery, 1.57/1.58-bit channel entropy, and a 2,406-hop result. No artifact supports any of it; the module now records only the artifact-backed C3 PPO numbers.
- `docs/modules/01_REED_SOLOMON_ECC_MODULE.md` and 11 research, guide, and architecture files asserted an unconditional 0% BER and 100% recovery. All are now bounded to the clean channel and the `t=4` correction budget, with dropout disclosed as unrecoverable.
- `docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md` and the C3/C4 report tables used the "Evasion Rate" column label. The documents say "not flagged", and Component 6 regenerated the table with the corrected **Not Flagged Rate** label.

Verification for this phase: `ruff check src tests` passes, the frontend production build succeeds with 14 routes, and `git diff --check` is clean.

---

## 4. Approval gate — behavior-changing items

Phase 1 does not start until these are discussed and explicitly approved.

| ID | Proposed change | Why it matters | Risk | Default recommendation |
|---|---|---|---|---|
| B1 | Unify GAN delay mapping between `_schedule_gan` and `_schedule_hybrid` | Deployed GAN mode returns raw ~0.501 s delays; the benchmark applies `(raw - 0.5) * 1500 + 1.5`. The reported C3 row is not the deployed path | Changes deployed timing; affects C3 results | Approve. Add a regression test and rerun C3 |
| B2 | Diagnose and repair the collapsed warden | `bot_probability = 0.0` on 50 real and 50 random sessions; anomaly is constant at 0.62493 | Changes detection outcomes; needs recalibration | Approve a timeboxed diagnostic (maximum 1 day) |
| B3 | Remove the soft-branch 0.45 cap in a separate ablation | The cap makes `0%` interception structurally guaranteed | Changes the C3 headline; only meaningful after B2 | Approve only if B2 succeeds; label as diagnostic, not evasion |
| B4 | Rerun C1 with only SRNet, ZhuNet, YeNet | Current claims mention five detectors, but only three are implemented/gated | New AUCs may differ; compute time | Approve. Report actual results, no cherry-picking |
| B5 | Reset test database and standardize credentials | Docs and UI disagree; tests only | Existing test users lost | Approve if convenient; otherwise update docs only |
| B6 | Add API multi-session/queue support | `/api/transmit` admitted one active transmission and returned HTTP 409 for a second session | New feature; changes concurrency semantics | **Approved and done.** Several sessions run at once, bounded by `MAX_CONCURRENT_TRANSMISSIONS` (default 4, `DCASS_MAX_CONCURRENT_TRANSMISSIONS`); over the bound the endpoint answers HTTP 429. The engine's encode and decode are serialized because the encoder and decoder hold mutable scratch state. |
| B7 | Repair Docker training service | References missing `scripts/stealth/generate_traffic_dataset.py` | Build/runtime breakage | **Approved and done.** The generator script is implemented and the training profile was run end to end. |

---

## 5. Phase 1 — Code, methodology, and reruns

**Goal:** Fix the mechanisms that generated misleading results, then regenerate the affected evidence. Start only after the approval gate.

### 5.1 C3 direction

**Default: negative/limited result.**

The C3 result at the time this plan was written was:

- `0/50` interceptions, 95% Clopper-Pearson CI **[0%, 7.1%]**, one-sided 95% upper bound **5.8%**.
- The soft branch was clipped to **0.45** by construction, below the **0.50** flag threshold.
- The warden returned `bot_probability = 0.0` on 50 real and 50 random sessions.
- The anomaly score was constant at **0.62493**.
- The "human" baseline was synthetic.
- WGAN KS was **0.3917**, worse than Poisson's **0.2211**. (That figure belongs to the pre-repair generator; the retrained generator records **0.5595**.)

This could not support an evasion claim, and the `0%` was an artifact of the cap and the collapsed detector rather than a measurement. **That was the state before the Phase 1 remediation.** The soft branch is now a trained traffic detector with a threshold calibrated on held-out human traffic, and the recorded result is 100% interception for the WGAN and Hybrid rows and 0% for the retrained PPO arm, at a measured 4.60% false-positive rate. The conclusion is unchanged in direction: the benchmark does not demonstrate evasion, and now it also does not leave the question open, because the traffic is detected. See the Phase 1 progress record in section 5.4.

**Timeboxed diagnostic (maximum 1 day):**

1. Trace why `bot_probability` is exactly zero and the anomaly score is constant.
2. Check feature extraction, model loading, normalization, and threshold logic.
3. If the cause is a small, verifiable bug, repair it, calibrate the threshold on held-out data, and rerun C3 with the cap removed as a diagnostic.
4. If the cause is not tractable within the timebox, freeze C3 as a negative/limited result and publish the current numbers with full disclosure.

**Upgrade condition for a positive claim:** a functional, calibrated warden plus a real human/background traffic baseline, with a pre-registered analysis. If that data is not available by Day 4, the negative result stands for this deadline.

**Never do:** report `0%` as evasion, use the benchmark-only GAN mapping without disclosure, or hide the soft cap.

#### 5.1.1 Diagnostic outcome (2026-10-21)

The diagnostic ran to completion. Outcome: **branch 4 — freeze C3 as a negative/limited result.** The full write-up is [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md).

Root cause, in order of proximity:

1. **The trained transformer collapsed to a constant-output function.** The shipped checkpoint attenuates the difference between two maximally different delay sequences by roughly a factor of ten at every transformer layer, about `1e-4` in total, while a fresh network passes the same difference through unchanged. The per-layer statistics show the feed-forward branch learning a large input-independent constant that dominates the residual stream, with the final LayerNorm scaling the input-dependent component away. Swapping a fresh transformer in restores full input sensitivity to the shipped anomaly head.
2. **The anomaly head is a dead branch.** It receives no gradient at all: `compute_warden_loss` and `compute_gradient_penalty` both differentiate only `raw_critic_score`, and the anomaly head's output never reaches the loss.
3. **`bot_probability = 0.0` is a float32 underflow.** The raw critic score is a constant `-90.2509`, and `sigmoid(-90.25) = 6.38e-40` is below the float32 normal range.
4. **The training run had already diverged** when the checkpoint was written. The saved history shows the generator loss growing from 2.3 to 127.9 and the gradient penalty pinned at exactly `lambda_gp`, which is the value a constant critic produces.

Checked and cleared: checkpoint loading and key coverage, input normalization and feature extraction, and the benchmark's threshold logic. All are correct; the fault is in the trained weights and the training objective.

Two consequences beyond C3's own numbers:

- The RL environment takes its stealth penalty from `verdict.bot_probability`, so the PPO agent was trained with **no stealth signal at all**. The PPO arm is a throughput-and-diversity policy.
- The available traffic corpus, `storage/data/traffic/real_human_traffic.json`, is a synthetic surrogate despite its filename: its time-of-day distribution is uniform (chi-square p = 0.33), it has no delays beyond 120 s, and 4.7% of values sit exactly on the 0.5 s floor. Training a detector against it and then reporting detection rates against it would be circular.

Because the cause is a diverged training run plus a synthetic corpus, and not a small verifiable bug, the upgrade condition for a positive claim is not met within this deadline. **C3 is frozen as a negative/limited result.** The soft-cap ablation (B3) was not approved and is not required: the `0/50` is cap-determined, so it is invariant to the warden and to the traffic being measured.

### 5.2 Task list

| ID | Task | Primary files | Acceptance |
|---|---|---|---|
| P1-1 | Archive current benchmark artifacts | `storage/benchmarks/archive/<date>-pre-remediation/` | Old JSON, tables, and plots remain available; nothing overwritten in place |
| P1-2 | Unify scheduler delay mapping and add regression test | `src/stealth/stealth_scheduler.py`, tests | Deployed and benchmark paths produce identical delays for the same seed/checkpoint, or the contract is explicit and tested |
| P1-3 | Run C3 warden diagnostic | Warden/scoring code, diagnostic note | Root cause identified or timebox expired with a written conclusion |
| P1-4 | Rerun or reframe C3 | `scripts/analysis/benchmark_3_traffic_evasion.py`, `component_3_results.json`, docs | Either a repaired diagnostic with CIs or a negative-result writeup with structural-cap disclosure |
| P1-5 | Add missing GAN timing benchmark or remove reference | `scripts/analysis/benchmark_gan_timing.py`, module doc | The referenced script exists and runs against the production path, or the reference is removed |
| P1-6 | Rerun C1 with three detectors only | `scripts/analysis/benchmark_1_media_steganalysis.py`, `component_1_*.json` | Only SRNet, ZhuNet, YeNet are reported; actual AUCs and CIs recorded; audio probe marked exploratory |
| P1-7 | Correct C2 statistical framing | `component_2_results.json`, C2 doc, report, paper | `0/1536` BH discoveries reported; image/text small shifts and audio `p ≈ 0.57` stated; mechanism null `p = 0.208`; no uniform-indistinguishability claim |
| P1-8 | Separate coded and user capacity | `component_4a_capacity_results.json`, tables, report, paper | Coded bpc and user-payload bpc are distinct columns; `exact_vcp` labeled coded; 14 B user payload is 7.00 bpc |
| P1-9 | Scope C5 to the direct engine | `component_5_results.json`, C5 doc, report, paper | Direct-engine results stated; the API's admission limit stated; no API concurrency claim |
| P1-10 | Add run manifests | Shared benchmark utility and all benchmark scripts | Each run writes git SHA, dirty flag, timestamp, command, seeds, package versions, input counts/hashes, output hashes |
| P1-11 | Rerun affected benchmarks | `scripts/analysis/*` | Actual commands and durations recorded; artifacts archived and manifested |
| P1-12 | Sync docs to rerun numbers | All benchmark docs | Claim ledger updated; no hand-copied numbers |
| P1-13 | Final verification | Tests, ruff, integration if behavior changed | Exact pass/fail counts reported; no known claim/artifact mismatch |

### 5.5 Run manifest format

`scripts/analysis/benchmark_manifest.py` provides the shared helper. Each
manifest records:

- the exact command and `argv`;
- the git commit, branch, and dirty flag with the dirty paths;
- the UTC timestamp, Python version, and platform;
- the versions of torch, numpy, scipy, scikit-learn, faiss, and matplotlib;
- the seeds the run used;
- the SHA-256 and size of every input read;
- the SHA-256 and size of every output written.

Component 3 and component 3b write manifests today. The helper is
provenance only: it never changes what a benchmark measures. The remaining
components get manifests when they are next rerun, so that a manifest never
describes a run that did not happen.

### 5.6 Rerun verification (2026-10-21)

Component 3 was rerun after the B1 fix, with the pre-remediation artifacts
archived first in `storage/benchmarks/archive/2026-10-21-pre-remediation/`.

| Comparison | Result |
|---|---|
| Max scalar difference across every scheduler field | **0.0** |
| Rule attribution per scheduler | Identical |
| Diurnal breakdown | Identical |
| Warden positive-control self-test | Identical |
| Component 3b LaTeX table | Byte identical |
| Component 3b JSON summary | One float32 mean differs in the last digit |

The numbers are unchanged because the benchmark was already applying the
canonical mapping. The defect B1 fixed was only in the deployed GAN path,
which returned the raw generator output. The rerun therefore confirms
rather than alters the result, and it establishes that the recorded row and
the shipped row are now the same code path.

### 5.3 Phase 1 exit criteria

- Scheduler mapping is consistent or explicitly contracted and tested.
- C3 has a defensible conclusion with uncertainty and full disclosure.
- C1 reports exactly three detectors.
- C2, C4, and C5 claims match their artifacts and stated scope.
- Every regenerated artifact has a manifest and an archived predecessor.
- Tests and lint pass.
- Working tree committed per task; no push.

### 5.4 Phase 1 progress record (2026-10-21)

Approved items: **B1** (unify the GAN delay mapping), **B2** (timeboxed warden diagnostic), **B1-b** (fix the soft branch's weighting and make the warden a real detector), **B4** (rerun C1 with three detectors), **B6** (API multi-session support), **B7** (repair the Docker training service), and **P1-10** (run manifests for the other components). B3 and B5 are not approved and are not needed: B3 is a mathematical no-op, because the soft branch's maximum was 0.45 against a 0.50 threshold, so removing the cap could not change any verdict; and B5 is already satisfied, since the documented seed passwords verify against the existing database.

| ID | Task | Outcome |
|---|---|---|
| P1-1 | Archive current benchmark artifacts | **Done.** The pre-remediation C3 JSON, tables, and plots are in `storage/benchmarks/archive/2026-10-21-pre-remediation/`. |
| P1-2 | Unify scheduler delay mapping and add regression test | **Done.** `canonical_delay_mapping` now lives in `stealth_scheduler.py` and is shared by the deployed GAN path, the deployed hybrid path, and the benchmark. The deployed GAN path applies it, `schedule()` accepts a pinned `time_of_day` so the benchmark drives the deployed path, and `tests/test_stealth/test_scheduler_delay_mapping.py` pins the contract. |
| P1-3 | Run C3 warden diagnostic | **Done.** Root cause identified and written up in `C3_WARDEN_DIAGNOSTIC.md`, with three reproduction scripts and three stored JSON artifacts. |
| P1-4 | Rerun or reframe C3 | **Done, and reframed (B1-b).** The diagnostic's "the branch cannot flag" reading turned out to be stronger than "the result is negative": the soft branch could not flag *any* traffic, for *any* input, so the recorded 0/50 was arithmetic rather than a measurement. The branch is now a trained traffic detector's bot probability against a threshold calibrated on held-out human traffic, and C3 records 100% interception for the WGAN and Hybrid rows and 0% for the retrained PPO arm, at a measured 4.60% false-positive rate. The conclusion is unchanged in direction: C3 remains a negative/limited result and no positive evasion claim is made. |
| P1-5 | Add missing GAN timing benchmark or remove reference | **Done.** The reference was removed in Phase 0; `scripts/analysis/benchmark_3_traffic_evasion.py` is the timing benchmark, and it now drives the deployed scheduler path. |
| P1-6 | Rerun C1 with three detectors only | **Done (B4).** The rerun with `--detectors sr_net,zhu_net,ye_net` completed. The three fold artifacts and the summary now carry exactly the same three detectors, and the measured AUCs reproduce the recorded values exactly: 0.4973, 0.4993, and 0.4993, each with a 95% interval covering 0.5000, with all three positive-control gates passing. The two foil detectors are no longer in the fold artifacts, so the report points at the archived five-detector campaign for their numbers. |
| P1-7 | Correct C2 statistical framing | **Done in Phase 0.** |
| P1-8 | Separate coded and user capacity | **Done in Phase 0** for the documents; the report table now carries both columns. Regenerating the C4a artifact to store a user-payload field is not approved and not required for the claim. |
| P1-9 | Scope C5 to the direct engine | **Done in Phase 0.** |
| P1-10 | Add run manifests | **Done.** The shared helper `scripts/analysis/benchmark_manifest.py` exists, and components 1, 2, 2b, 3, 3b, 4a, 4b, 4c, and 5 all write a manifest recording the git SHA, the dirty flag, the command line, the seeds, the package versions, and the input and output hashes. Each manifest is written after the tables and plots it lists, so the hashes are of the final files. Producing the manifests meant rerunning every component, which surfaced three provenance-ordering defects that are now fixed: a table caption that named the previous run's commit, two components that recorded no provenance at all, and a test that asserted the component 5 artifact was a retrospective reconstruction when it is now a fresh run. |
| P1-11 | Rerun affected benchmarks | **Done for every component.** Components 3 and 3b were rerun because B1 and B1-b could affect them; the rest were rerun to produce their manifests, and components 4a and 5 were rerun again on an idle GPU because their first runs shared the device with the component 1 training. The capacity and success rates are unchanged; the latency percentiles are hardware measurements and shifted, so the documents that quote them were updated. |
| P1-12 | Sync docs to rerun numbers | **Done.** The ledger rows `C3-WARDEN-02`, `C3-CAP-01`, `C3-STRUCT-03`, `C3-PPO-05`, `C3-MAP-04`, and `C3-RERUN-05` carry the root cause, `C3-CAP-01` is retracted, and the new `C3-DETECT-06` records the replacement detector, the calibrated threshold, the measured false-positive rate, and the scope limit. The C3 document, the report, the paper writeup, the deep explanation, the benchmarking README, the module docs, and the implementation summary all carry the new numbers. |
| P1-13 | Final verification | **Done.** 319 tests pass and 39 are skipped under `--no-integration`; `ruff check src tests` is clean; the four LaTeX and inventory checks (`check_artifact_inventory`, `check_latex_compile`, `check_latex_table_widths`, `check_paper_latex_block`) are green and repeatable; the frontend builds. The working tree is committed per task and nothing is pushed. |

---

## 6. Phase 2 — Component 6 inventory and LaTeX reconciliation

**Scope:** inventory and table/figure reconciliation only. Camera-ready formatting is deferred until paper writing.

| ID | Task | Output |
|---|---|---|
| P2-1 | Inventory diff | Compare 9 tables, 19 plots, and 34 top-level JSON artifacts against report section 8 and `06_PUBLICATION_ARTIFACTS_AND_LATEX.md`; list orphans and missing entries |
| P2-2 | Regenerate `06_PUBLICATION_ARTIFACTS_AND_LATEX.md` | Every artifact mapped to source script, claim ID, and component |
| P2-3 | Integrate table5 | Add the missing table and any new tables/plots from Phase 1 |
| P2-4 | Reconcile report section 8 | Report inventory matches the filesystem |
| P2-5 | LaTeX compile check | Tables/figures compile; captions include n, scope, and commit |
| P2-6 | Final sweep | README, report, paper, docs, and UI consistent; ledger green; archive/tag the frozen state |

**Exit criteria:** every artifact is regenerable by one documented command; the inventory matches the filesystem; the LaTeX build succeeds; no orphan or stale artifact remains.

### 6.1 Phase 2 completion record (2026-10-21)

| ID | Task | Outcome |
|---|---|---|
| P2-1 | Inventory diff | **Done.** `scripts/analysis/check_artifact_inventory.py` diffs the filesystem against the report and the publication document. The report had been missing 6 plots and 9 JSON artifacts; the publication document was missing table5 entirely. |
| P2-2 | Regenerate `06_PUBLICATION_ARTIFACTS_AND_LATEX.md` | **Done.** Both documents are now generated by `scripts/analysis/generate_publication_inventory.py` from the filesystem, so they cannot drift. Every artifact carries its source script, claim IDs, and regeneration command. |
| P2-3 | Integrate table5 | **Done.** Table 5 and the six missing plots and nine missing artifacts are now in the inventory. The three superseded per-detector C1 JSON files are listed as superseded, with the authoritative record named. |
| P2-4 | Reconcile report section 8 | **Done.** Section 8 is generated. The report's five inline copies of generated tables are now synced from the files rather than maintained by hand; they had drifted from the artifacts. |
| P2-5 | LaTeX compile check | **Done.** Every table compiles under the root-relative `\input` paths the paper uses, with `XeLaTeX` via `tectonic`. Every caption carries the sample size, the scope, and the commit that produced the measurement. The check found two real defects: table4b needs `amssymb` for `\checkmark`, and six of the nine tables overflowed the text block. |
| P2-6 | Final sweep | **Done.** The inventory, the compile check, the width check, and the paper-block check are all green and repeatable. The final claim sweep ran as three passes over the whole tree, each comparing every benchmark number against the artifacts: the first found the withdrawn C3 figures, the second found the numbers that the reruns had moved, and the third found the remaining twelve. The last three unqualified claims were a mermaid diagram reading "UNDETECTABLE!", a Cachin's-theorem sentence claiming "perfect information-theoretic security", and "completely undetectable" in the implementation handout; all three now say what the artifacts support. The only remaining hits for the sweep's vocabulary are a definition of epsilon-security in module 06, a "do not say this" instruction in module 08, the plan's own text, and the retracted rows in the claim ledger, all of which are correct as written. The frozen state is tagged `benchmark-freeze-2026-10-21`. |

### 6.2 What the LaTeX checks found

Two defects that would have surfaced only at submission:

1. **`\checkmark` needs `amssymb`.** Table 4b uses it, and the harness did not load the package, so the table failed to compile.
2. **Six of nine tables overflowed the text block.** The generated tables carry their own float environment, so the paper cannot re-wrap them. They are now emitted as a full-width `table*` with the tabular scaled by `\resizebox`, which fits in both a one-column and a two-column paper. All nine measure zero overfull.

An earlier attempt at the second fix was invalid: it nested a `table` inside a `table`, so the compile aborted and the measurement looked clean. The width check only became meaningful once the probe compiled.

### 6.3 Table regeneration contract

Every generated table is rebuildable from its recorded JSON without re-running a measurement:

| Script | Artifact-only mode |
|--------|--------------------|
| `benchmark_1_media_steganalysis.py` | `--table-only` |
| `benchmark_2_semantic_distribution.py` | `--table-only` |
| `benchmark_2b_transmitted_cluster_usage.py` | `--table-only` |
| `benchmark_3_traffic_evasion.py` | full run only (under a minute) |
| `benchmark_3b_warden_score_distributions.py` | full run only (seconds) |
| `benchmark_4a_capacity.py` | `--artifacts-only` |
| `benchmark_4b_robustness.py` | `--regenerate-from-json` |
| `benchmark_4c_ecc_failure_analysis.py` | `--replot` |
| `benchmark_5_system_concurrency.py` | `--table-only` |

A regenerated table is byte-identical to the committed one, because the caption's commit is read from the artifact's own provenance block rather than from the current HEAD.

### 6.4 Verification commands

```bash
# inventory matches the filesystem
.venv/bin/python scripts/analysis/check_artifact_inventory.py

# every table compiles under the paths the paper uses
.venv/bin/python scripts/analysis/check_latex_compile.py

# every table fits the text block
.venv/bin/python scripts/analysis/check_latex_table_widths.py

# the paper writeup's own LaTeX block compiles
.venv/bin/python scripts/analysis/check_paper_latex_block.py

# regenerate the inventory documents
.venv/bin/python scripts/analysis/generate_publication_inventory.py
```

The three LaTeX checks need a `tectonic` binary. Without one they report
that they could not run rather than claiming a pass.

---

## 6.5 Phase 3 — Follow-on engineering (2026-10-21)

Phases 0 to 2 closed out the documentation and the recorded measurements.
Phase 3 is the follow-on engineering that the C3 reframe showed was
needed, and it was approved as it went. Three pieces of it.

### 6.5.1 The RL arm's stealth path (B8)

The PPO arm's 0% interception was recorded against a warden whose
`bot_probability` was identically zero, so its stealth penalty was zero
and the agent optimised throughput and channel diversity only. Four
defects in the RL path were fixed and the agent was retrained against a
working detector, with a **held-out** detector at a different seed doing
the measuring, and a threshold calibrated on human bursts at a measured
4.60 percent false-positive rate. See
[`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md).

### 6.5.2 The generator's adversarial path (B9)

The generator's adversarial signal was a WGAN-GP critic whose training
diverged, so the generator had never optimised against anything. Four
defects were measured and repaired: the critic was given the generator's
raw output while the real traffic was in seconds, the delay mapping left
a fresh generator with no gradient at all, the critic's score had no
anchor, and the gradient penalty was too weak to keep the critic
reading its input at the penalty's default weight. The repair is
recorded in
[`C3_GENERATOR_ADVERSARY.md`](C3_GENERATOR_ADVERSARY.md), and the ledger
rows are `C3-GEN-07` (retracted) and `C3-GEN-08` (supported).

### 6.5.3 The real human traffic path (B10)

The human side of every Component 3 claim is a synthetic corpus. The
corpus directory is now resolved from the environment, an ingester
converts a capture of real per-request timestamps into the corpus format
and reports on it before it is used, and the re-run sequence is
documented. The measurement machinery did not need to change; only the
corpus does. See
[`REAL_TRAFFIC_CAPTURE.md`](REAL_TRAFFIC_CAPTURE.md). The capture itself
is still missing, and it is a data problem rather than a code one.

### 6.5.4 What did not change

The conclusion. C3 is a negative and limited result: the WGAN and Hybrid
schedules are flagged at 100 percent, the retrained PPO arm at 0 percent,
and no positive evasion claim is made. The scope limit is unchanged and
is the operative one: the human side of the comparison is the
repository's own synthetic corpus, and the detector is a narrow timing
model. The scope limit travels with every number.

---

## 7. Seven-day schedule

Dates are targets, not promises. The C3 diagnostic is the main schedule risk.

| Day | Work | Gate |
|---|---|---|
| **Day 1 (Oct 21)** | Claim ledger; README; benchmarking overview theory section; report executive summary | Phase 0 checkpoint |
| **Day 2 (Oct 22)** | Paper writeup; module 05; C1–C5 as-run headers; UI copy; license/Docker references | Phase 0 review |
| **Day 3 (Oct 23)** | **Approval discussion.** Archive artifacts; begin C3 warden diagnostic; scheduler mapping fix if approved | Behavior-change gate |
| **Day 4 (Oct 24)** | Finish C3 diagnostic; C3 rerun or reframe; C1 three-detector rerun | C3 direction frozen |
| **Day 5 (Oct 25)** | C2/C4/C5 doc/schema sync; run manifests; full tests and ruff | Verification gate |
| **Day 6 (Oct 26)** | Component 6 inventory; table5; report section 8; LaTeX compile | Component 6 checkpoint |
| **Day 7 (Oct 27)** | Final claim sweep; archive/tag; buffer for rerun overruns | Freeze |

If the C3 rebuild is chosen instead of the negative-result reframe, the one-week deadline is at risk. Either reduce scope or extend the deadline.

---

## 8. Finding coverage matrix

| Priority | Finding | Resolution |
|---|---|---|
| P0 | C3 `0%` evasion is structurally guaranteed by the 0.45 soft cap | Resolved in Phase 1 (B1-b): the soft branch could not flag any traffic for any input, so the 0/50 was arithmetic. The branch is now a trained detector with a calibrated threshold, and C3 records 100% interception for the WGAN and Hybrid rows and 0% for the retrained PPO arm. `C3-CAP-01` is retracted |
| P0 | Warden collapsed: `bot_probability = 0.0`, anomaly constant `0.62493` | Resolved in Phase 1 (B1-b): the root cause is in `C3_WARDEN_DIAGNOSTIC.md`, and a purpose-built detector now supplies the neural term. The shipped checkpoint is still the RL environment's stealth penalty |
| P0 | Benchmark GAN row is not the deployed `_schedule_gan` path | Phase 1 scheduler mapping fix; Phase 0 disclosure |
| P0 | README benchmark table and "50 client threads at 100% request success" | Phase 0 README rewrite |
| P0 | Perfect-security, `D_KL = 0`, Nash claims | Phase 0 benchmarking overview and paper rewrite |
| P0 | Demo credential mismatch | Phase 0 UI/docs; optional test DB reset |
| P1 | C4 coded vs user capacity conflation | Phase 0 labels; Phase 1 schema/table check |
| P1 | C5 direct-engine scope and API HTTP 409 | Resolved: the scope box is in place from Phase 0, and the API now admits concurrent sessions up to a bound and answers HTTP 429 beyond it |
| P1 | C2 `0/1536` BH discoveries vs uniform-null claim | Phase 0 rewrite; Phase 1 statistical framing |
| P1 | Audio provenance (`libretta`, 13,496 clips) and C1 detector scope | Phase 0 correction; Phase 1 C1 rerun |
| P1 | C3 confidence intervals absent from headline claims | Phase 0/1 rewrite |
| P1 | Stale module 05 WGAN claims and missing GAN benchmark script | Phase 0 rewrite; Phase 1 script fix |
| P1 | No run manifests/provenance | Resolved in Phase 1 (P1-10): all nine benchmark components write a run manifest |
| P2 | Component 6 inventory stale; table5 missing | Phase 2 |
| P2 | No `LICENSE` despite the former MIT claim | Resolved in Phase 1: MIT `LICENSE` file added 2026-10-21 |
| P2 | Docker training service references missing script | Resolved in Phase 1 (B7): `scripts/stealth/generate_traffic_dataset.py` implemented and the training profile run end to end |
| P2 | Frontend Send page hardcodes `exact_vcp`; showcase claims unqualified | Phase 0 UI copy |
| P2 | Camera-ready paper artifacts | Deferred until paper writing |

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Warden root cause is not fixable in one day | C3 remains negative/limited | Timebox strictly; publish the negative result with CIs |
| C1 rerun changes AUCs | Headline numbers move | Report actual values; archive old artifacts; no cherry-picking |
| Reruns exceed the schedule | Phase 2 compressed | Prioritize C3 and C1; defer manifest work for unaffected components if necessary |
| Docs edits reveal more inconsistencies | Scope creep | Log every new finding in the claim ledger; fix only headline claims this week |
| C3 rebuild is chosen late | Deadline missed | Decide by Day 3; otherwise default to negative result |
| Auth reset breaks saved test state | Test friction | Test-only environment; document the reset |
| GPU/model downloads unavailable | C1/C3 reruns blocked | Check environment on Day 3; use cached models; record any fallback |

---

## 10. Out of scope this week

- A positive C3 evasion claim requiring a real human baseline and a calibrated warden.
- New API multi-session/queue support.
- Camera-ready paper formatting and submission.
- Full legal/license review.
- Repairing the Docker training service unless it is trivial.
- New detector implementations (StegExpose, Xu-Net, classical SRM) beyond the three agreed C1 models.
- Component 6 camera-ready regeneration beyond inventory and LaTeX reconciliation.

---

## 11. Open items

1. **C1 naming:** resolved. "SRM" means **SRNet**; the recorded C1 scope is SRNet, ZhuNet, and YeNet.
2. **C3 data:** the negative/limited result is final for this deadline, and it is now a measurement rather than an artifact. The retrained RL arm is not flagged by a held-out detector, and it respects the platform's cooldowns, but that is a statement about the repository's own synthetic traffic corpus. If a positive evasion result is wanted in the real world, the missing input is data, not code: a real human or background traffic capture plus a calibration set. Everything the analysis needs is built — the detector trainer, the session-disjoint held-out protocol, the held-out evaluator, the threshold calibration with a measured false-positive rate, and the benchmark. Drop a real capture in place of the corpus and rerun. Until then the scope limit travels with every number.
3. **License:** resolved. The project is MIT-licensed. A `LICENSE` file was added on 2026-10-21 on explicit user confirmation, and the README points to it.
4. **Credentials:** resolved for now. The environment is test-only and the database may be reset; the documented accounts are `alice/alice12345`, `bob/bob12345`, and `charlie/charlie12345`.
5. **Component 6:** resolved. Inventory and LaTeX reconciliation only this week; camera-ready work is deferred until paper writing.

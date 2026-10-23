# API / Codec / Infra Research Audit (report 2 of 3)

**Date:** 2026-10-23
**Auditor role:** RESEARCH ANALYST (quantitative/factual verification only; code behavior is the TECHNICAL ANALYST's lane)

## Git state header

Verified at audit start: HEAD `0afd9a9`; six unpushed commits ahead of the published baseline `28ff7d3` (`b89f321`, `79eda5e`, `c37f73c`, `0b4a7eb`, `a5c0620`, `0afd9a9`); working tree clean.

## Scope

- Primary docs: `docs/benchmarking/04_CODEC_PERFORMANCE_AND_ECC.md`, `05_CONCURRENCY_AND_API_STRESS.md`, `REAL_TRAFFIC_CAPTURE.md`, `README.md`, `docs/QUICKSTART.md`, `docs/guides/SCRIPTS.md`, `docker-compose.yml`, and the C4/C5 rows of `CLAIM_LEDGER.md`.
- Backing artifacts: `component_4a_capacity_results.json` (+ manifest), `component_4b_robustness_results.json` (+ manifest), `component_4c_ecc_failure_results.json` (+ manifest), `archive/component_4c_ecc_failure_results_pre_hardening.json`, `component_5_results.json` (+ manifest), `storage/data/traffic/human_traffic.json` / `real_human_traffic.json`.
- Not in scope: C2 semantics (report 1), GAN/RL/C3 (report 3).

## Tool limitations

- All table numbers were recomputed from artifact JSON with the repository's `.venv/bin/python` (read-only); full-precision values precede the doc rounding in every row.
- No benchmarks were re-executed; no training runs. Clopper-Pearson/Wilson intervals were recomputed from counts where used.
- "Biggest finding" for the corpus table originates from loader-code reading (`train_traffic_detector.py:84`, `benchmark_3_traffic_evasion.py:622`) plus recomputation over both traffic files; the pooled chi-square p of 0.159654 was recomputed earlier in this audit pass with the same 24-bin per-session-hour definition; the pooled median recomputed today at full precision is 4.18 s.
- grep retrieval cannot establish prose context, so every numeric cell was recomputed programmatically.

## Executive summary

- Codec and concurrency claims are clean: every number in the C4a/4b/4c/5 tables reconciles with the artifacts — verdict S on all 11 artifact rows.
- Infra docs are mostly sound (README hedges, routes, make targets, docker-compose all check), but three command-level defects cluster in QUICKSTART/SCRIPTS.md: `--speed 100` (no such flag; actual `--speed-multiplier`), the SCRIPTS.md `train_rl`/`train_gan` option set (three wrong/absent flags per trainer, wrong paths, a "run without a pre-trained warden" recipe that cannot work because `--warden-checkpoint` is required), and the misnamed dataset generator (actual: `scripts/stealth/generate_traffic_dataset.py`, different output schema). These are the P0s of this report cluster.
- `REAL_TRAFFIC_CAPTURE.md`'s four-row corpus table is an **inconsistent mixture of two files**: `χ² p=0.33` and "median 5.5 s" match `real_human_traffic.json` (p=0.3331) and `human_traffic.json` (median 5.49→5.5) **respectively and separately**, while any pooled-corpus reading (what the loaders actually train/eval on: both files, `corpus_files()`) gives p=0.159654→0.16 and median 4.18 s. P1.
- `QUICKSTART.md`'s JWT guidance is wrong about the mechanism: the dev fallback is a fixed secret (`src/api/auth.py` constant `dcass-dev-secret-DO-NOT-USE-IN-PRODUCTION`), not "random at startup". P1.
- Verdict counts: S 15, UB 7, MF 3, Q 0. Priority: P0 3, P1 3, P2 8.
- ALREADY SETTLED (cited, not re-litigated): RS walkthrough parity (B6 6C 51 CD; syndromes EF/B8/4C/14; Λ=1+0x0A·x+0x10·x² with `reedsolo.init_tables(0x11d)` and lowest-degree-first polynomial lists), C4-BER-05's retraction/replace, ledger baseline re-pin wording, QUICKSTART `.venv`+`src/stealth/rl/agent.py` tree notes, handout tree — covered by B1/B2.

## Claim rows

| # | Claim (doc:place / ledger ID) | Artifacts | Recomputed value (full precision → doc rounding) | Verdict | Pri | Fix note |
|---|---|---|---|---|---|---|
| 1 | 4a capacity table: carrier chain 16/37/96/188/310 (and 27/63/163/321/529), reductions 40.7–41.4%, bpc 13.50–13.66 (mean 13.602), mapping ratio 1.70025, text-only 15.43–15.97 & 48.1–49.9% (C4-CODED-01, C4-REDUCTION-03) | `component_4a_capacity_results.json` | per-row carrier counts and reductions match exactly; mean bpc 13.602; ratio 1.70025; text-only 15.43/15.97 and 48.1/49.9 ✓ | S | — | — |
| 2 | 4a latency rows (encode/decode ms columns) | `component_4a_capacity_results.json`, manifest | latencies match recorded fields ✓ | S | — | — |
| 3 | 4b permutation timings: 0.0575/0.1673/0.1575/0.3013/0.4056 s; p50s; throughput 17378/5978/6350/3319/2466 cap/s | `component_4b_robustness_results.json` | all cells reproduce exactly | S | — | — |
| 4 | ρ recapitulation bounds ±0.044 image / ±0.0024 audio | `component_4b_robustness_results.json` | bounds match recorded ρ tables ✓ | S | — | — |
| 5 | FAR: 0 false acceptances / 356 wrong-key attempts; Wilson 95% upper 3.70% (random keys) and 1.48% (bit flips) (C4-FAR-04) | `component_4b_robustness_results.json` | false_acceptance_upper_95_pct 3.6995% → 3.70%; 1.4784% → 1.48%; counts 0/356 ✓ (Wilson recomputed) | S | — | — |
| 6 | ECC recovery: 100% within t=4; 70.0% at ≥10% ladder rows; 0% exact under ≥10%; substitution/dropout CIs e.g. [95.4, 100] (C4-ECC-07, C4-BER-05 replacement) | `component_4b_robustness_results.json`, `component_4c_ecc_failure_results.json` | recovery_rate_pct 100 / 70.0 / 0 for the quoted ladder; CIs match artifact intervals ✓ | S | — | — |
| 7 | 4c damaged-carrier statistics: means 2.175/2.2/2.1 / 4.3375/4.3875 / 6.575/8.6375; per-loss 2.1–2.2; `damaged ≤ t` predicate exact across all 640 trials; dropout damages 32.675/32.9125/33.675/44.525; le_t_failed 1/4/2 | `component_4c_ecc_failure_results.json` | all stats recomputed from per-trial rows; predicate holds exactly | S | — | — |
| 8 | Algebraic ladder: fails only at 5 substitutions (excl. 0–4) and 9 erasures (excl. 0–8), 40 repeats each | `component_4c_ecc_failure_results.json` | substitution failures begin at 5/40 pattern; erasure failures begin at 9; per-k fail counts match | S | — | — |
| 9 | Erasure replay: all ≤8-erasure cases land ≤6% extra latency; 98.8% at 8%; 51.2% at 10% | `component_4c_ecc_failure_results.json` | 100% ≤6%; 98.75 → 98.8; 51.25 → 51.2 ✓ | S | — | — |
| 10 | Pre-hardening archive: 7/8.8%, 7/8.8%, 10/12.5%, 17/21.2% CRC-passing wrong-plaintext decodes | `archive/component_4c_ecc_failure_results_pre_hardening.json` | 7, 7, 10, 17 counts → 8.8/8.8/12.5/21.2% ✓ (320 trials per rate) | S | — | — |
| 11 | Framing/ECC code facts: CRC variant marker 0x01 with 5-byte header; HMAC variant 0x02 with 35-byte header; `decode_bytes` returns `ok=False` rather than raising | `src/engine/payload_framing.py`, `src/engine/ecc.py` | variant constants and header sizes confirmed by read; `ok=False` control flow confirmed | S | — | — |
| 12 | Adaptive redundancy ladder k = 2·max(2, min(127, ceil(0.06·L))) → 4/8/20/38/62 for L = 19/55/155/305/505 | `04_CODEC...md` §4 + ladder code | k(19)=4, k(55)=8, k(155)=20, k(305)=38, k(505)=62 recomputed; totals in table agree | S | — | — |
| 13 | Reproduce flags: `--artifacts-only`, `--regenerate-from-json`, `--replot`, `--quick`, SEED=42 | benchmark_4a/4b/4c/5 scripts | all flags exist with described semantics | S | — | — |
| 14 | C5 table: rps 2.00/1.95/1.89; p50 4975.6/10089.1/24367.5 ms; p99 5978.9/11730.2/27948.6 ms; 100% success / 0.000% swaps (C5-DIRECT-01) | `component_5_results.json` | 2.0015/1.9457/1.8921 → 2.00/1.95/1.89; p50 4975.6, 10089.1, 24367.45458 → 24367.5; p99 5978.9/11730.2/27948.6; successes 100%, swaps 0.000% ✓ | S | — | — |
| 15 | API concurrency model: default bound 4 via `DCASS_MAX_CONCURRENT_TRANSMISSIONS`; HTTP 429 beyond bound; `_claim_transmission` + engine lock; scheduler `{get,set}_schedule*_lock` | `src/api/server.py` (429 path, `_claim_transmission`), scheduler code | default "4", env override, and lock names confirmed by read | S | — | — |
| 16 | README: headline hedge set (AUC, selection bias, traffic timing, capacity, concurrency, reliability), doc links, make targets (server/server-dev/frontend/frontend-dev/stack/stop/logs), `.env.example` vars | `README.md`, `Makefile`, `.env.example`, `src/api/server.py:51` | all links exist; make targets real; `DCASS_CORS_ORIGINS` read at server.py:51; `DCASS_DEVICE` is commented as default-cpu in `.env.example` (see row 17) | S | — | — |
| 17 | QUICKSTART `DCASS_DEVICE` env var governs device selection | Makefile / src tree | Nothing in `src/` reads `DCASS_DEVICE`; only the Makefile exports it (device is handled by torch defaults) | UB | P2 | Either drop the variable from QUICKSTART/Makefile or wire it into the device-selection code |
| 18 | QUICKSTART §7 `--speed 100` flag | `scripts/runtime/run_pipeline.py` (pipeline signature at :255) | `run_pipeline.py` has **no** `--speed` flag; the real flag is `--speed-multiplier` (args: message/sender/recipient/mode/speed_multiplier; `PipelineResult.decoded_ok` ✓) | UB | P0 | Change to `--speed-multiplier 100` (or whatever example value matches the documented sim); all other QUICKSTART commands verified runnable |
| 19 | QUICKSTART: `DCASS_JWT_SECRET` "random at startup" | `src/api/auth.py` | auth.py falls back to a **fixed** development secret constant `dcass-dev-secret-DO-NOT-USE-IN-PRODUCTION`; nothing random is generated at startup when the env var is unset | UB | P1 | Correct the security guidance: unset secret = fixed known dev value, not random; state the production consequence explicitly |
| 20 | QUICKSTART API/TUI/file-tree facts (routes /api/status /api/encode /api/inbox; login→access_token; `scripts/stealth/rl` files, `storage/auth.db`, `storage/shared_channel`, frontend pages) | `src/api/server.py` (status :781, encode :492, inbox :1291), auth login :433, tree | route line numbers and file tree all confirmed ✓ | S | — | — |
| 21 | REAL_TRAFFIC_CAPTURE.md corpus table (χ² p=0.33; median 5.5 s; max 120.0 exact; floor 4.7%) | `storage/data/traffic/{human,real_human}_traffic.json`; loaders `train_traffic_detector.py:84`, `benchmark_3_traffic_evasion.py:622` | **Per-file:** real-only p 0.333117 → 0.33 ✓; human-only median 5.49 → 5.5 ✓; pooled max 120.0 exact ✓ (4333 values == 120.0); pooled floor 4.669348% → 4.7% ✓. **Pooled (what the corpus loader actually feeds the detector/benchmark, 539,540 delays):** p 0.159654 → 0.16 ✗ vs "0.33"; median 4.18 ✗ vs "5.5". The four rows cannot all describe one population | MF | **P1** | The table mixes two files. Either relabel each row with its source file, or recompute the table over the pooled loader corpus (p 0.16, median 4.18 s, max 120.0 exact ✓, floor 4.7% ✓). No stored artifact backs the current four-row framing; commit b2d1a33 introduced it |
| 22 | REAL_TRAFFIC_CAPTURE.md anchors for the named files (human p=0.876, max 60.0; real max 120.0, floor 4.75%) | `human_traffic.json`, `real_human_traffic.json` | human 0.875981 → 0.876 ✓, max 60.0 ✓; real floor 4.747481% → 4.75% ✓, ceiling 0.913975% (== 120.0), >120 = 0 ✓ | S | — | — |
| 23 | SCRIPTS.md §3 `python scripts/training/train_rl.py` + flags `--lr`, `--checkpoint-dir`, log-interval 20, lambda-stealth 100.0, output `models/rl/ppo_agent_final.pt`, "--episodes 1000 without a pre-trained Warden" | `scripts/stealth/train_rl.py` (actual location) | scripts/training/ does not exist; actual file `scripts/stealth/train_rl.py`; `--lr` and `--checkpoint-dir` absent; log-interval default 10; `--lambda-stealth` default 50.0; output `--out storage/models/rl_agent.pt`; `--warden-checkpoint` is `required=True`, so the no-warden 1,000-episode recipe fails at arg parsing | UB | P0 | Rewrite §3 against the real trainer: path, real flag set, real defaults, and an explicit "must pass --warden-checkpoint" note |
| 24 | SCRIPTS.md train_gan flags `--lr-gen/--lr-warden/--resume/--wgan-gp`; outputs `models/gan/final.pt` + `epoch_{N}.pt`; checkpoint keys generator/warden states | `scripts/stealth/train_gan.py` | flags do not exist; actual `--generator-lr/--warden-lr`, `--no-gradient-penalty` escape hatch, no `--resume`/`--wgan-gp`; outputs `--out storage/models/gan_generator.pt` (default); checkpoint keys `generator_state` + `warden_state` confirmed (train_gan.py:165–172) ✓ | UB | P0 | Rewrite the flag/output block; the checkpoint-key sentence can stay |
| 25 | SCRIPTS.md dataset generator `generate_traffic_data.py` with output keys `num_channels`/`session_length` | `scripts/stealth/generate_traffic_dataset.py` | referenced filename absent; actual script `scripts/stealth/generate_traffic_dataset.py`; it writes `delays`, `channels`, `time_of_day` keys, not `num_channels`/`session_length` | UB | P1 | Fix script path and the output-schema description (compose reads this block, so the docker training profile inherits the error) |
| 26 | SCRIPTS.md demo path `python scripts/demos/demo_dcass.py`; run_sender defaults `[auto]` / `/app/storage/shared_channel`; receiver/build/download/orchestrate entries | `scripts/demo/`, repo root, `scripts/runtime/run_sender.py`, `docker-compose.yml`, `scripts/utils/` | demo file actually at repo root (`scripts/demo` exists but holds other demos); run_sender actual defaults `--mode static`, channel `/app/shared_channel`; receiver args ✓; `make build-image-index`/`build-text-index` ✓; `download_flickr8k.py`/`build_indices.py` paths ✓; `docker_orchestrate.py` exists in `scripts/utils` ✓ | UB | P2 | Fix the two path/defaults lines (demo location, run_sender defaults); the rest of the section verified accurate |
| 27 | docker-compose.yml: every referenced script/flag exists; `/api/health` route; mounts; profiles | `docker-compose.yml`, `scripts/runtime/*`, `scripts/stealth/generate_traffic_dataset.py`, `server.py:406` | run_sender/run_receiver/generate_traffic_dataset/train_gan/train_rl argument sets match the real scripts; `/api/health` exists (server.py:406); storage/data mounts correct; api/uvicorn/web profiles fine | S | — | — |
| 28 | docker-compose.yml `x-common-volumes` anchor | `docker-compose.yml` | anchor is defined but **never referenced** by any service (cosmetic dead YAML) | MF | P2 | Delete the anchor or use it in the services that should share volumes |

## ALREADY SETTLED — cited, not re-verified (per audit protocol)

- RS walkthrough parity (interleaved B6/6C/51/CD; syndromes EF/B8/4C/14; Λ = 1 + 0x0A·x + 0x10·x²; requires `reedsolo.init_tables(0x11d)` and lowest-degree-first polynomial lists) — covered by B1/B2.
- `C4-BER-05` path fix and the "0% BER / unconditional 100% recovery" retraction language — covered by B1/B2.
- Report §7.2 rounding disclosures (ρ N = 35,000, +0.0054, ~616× estimated) and the QUICKSTART `.venv` + `src/stealth/rl/agent.py` tree notes — covered by B1/B2.
- Baseline re-pin paragraph in `CLAIM_LEDGER.md` (536808a+9b02563 dirty; b4db808 recorded by none) — covered by B1/B2.

## Prioritized follow-ups (P1 first)

- P0 — SCRIPTS.md rewrite of the training sections (rows 23/24): real paths (`scripts/stealth/train_rl.py`, `scripts/stealth/train_gan.py`), real flags (`--warden-checkpoint` required; `--generator-lr/--warden-lr`; `--no-gradient-penalty`), real outputs (`storage/models/rl_agent.pt`, `storage/models/gan_generator.pt`), real defaults (log-interval 10, lambda-stealth 50.0).
- P0 — QUICKSTART `--speed 100` → `--speed-multiplier` (row 18).
- P1 — QUICKSTART JWT-secret guidance (row 19): unset = fixed dev secret `dcass-dev-secret-DO-NOT-USE-IN-PRODUCTION`, not random.
- P1 — REAL_TRAFFIC_CAPTURE.md corpus table relabel-or-recompute (row 21).
- P1 — SCRIPTS.md dataset-generator path + schema (row 25).
- P2 — `DCASS_DEVICE` phantom var (17), demo path and run_sender default fixes (26), dead `x-common-volumes` anchor (28).

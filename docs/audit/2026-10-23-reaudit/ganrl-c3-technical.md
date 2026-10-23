# ganrl-c3 — Technical Analyst Report (audit re-run)

**Date:** 2026-10-23
**Analyst:** Technical analyst (two-agent audit re-run; reporting out of scope for security semantics)
**Scope:** ganrl-c3 = Part 1 verdicts on the 6 standing B3 candidates + Part 2 hunt for additional
correctness defects in the RL/GAN/scheduler/warden/tooling scope, with a secondary skim of
`src/api/server.py` and `src/engine/dssc_encoder.py` for mask/behave bugs. Out of scope: B1 security
findings, C2 semantic issues, C4 codec/ECC core, C5 concurrency (covered by their own analysts).

**Repo state at audit:** git tree **clean**, HEAD = **`0afd9a9`**, three commits visible
(`0afd9a9`, `a5c0620`, `0b4a7eb`). All line numbers below refer to this HEAD.

**Method / compliance:** Read-only analysis. No repo files modified, no commits, no training runs.
Evidence: direct file reads (file:line cited), artifact JSON reads (`storage/benchmarks/*.json`),
read-only `torch.load` inspection of `storage/models/gan_generator.pt` and `rl_agent.pt`,
`git show 9b02563:<file>` for recorded-artifact provenance, and two throwaway in-memory repro
scripts run with the project python (results quoted inline, not committed). The pytest suite was
not needed to reach any verdict. Recorded benchmark artifacts carry `provenance.git_commit`
`9b02563` (branch feature/gan-rl-pipeline, `git_dirty=True`, 2026‑10‑21), i.e. they predate HEAD
and are treated as recorded evidence of the code at that commit, not as claims about HEAD.

**Verdict vocabulary:** CONFIRMED (defect exists and reproduces/persists in HEAD, or is reachable
by a supported caller path today) · REFUTED · RISK (mechanism real but latent; fires only under
conditions not exercised by recorded runs) · DESIGN (documented-by-design deviation). Severity
P0 (corrupts a headline recorded result) / P1 (real correctness defect, reachable or high impact) /
P2 (correctness defect with bounded blast radius).

---

## Part 1 — Verdicts on the six standing candidates

| # | Candidate | Verdict | Severity | One-line disposition |
|---|-----------|---------|----------|----------------------|
| 1 | RL export tail-pad is cooldown-blind | **CONFIRMED** (reachability upgraded from RISK) | **P1** | Timeout-truncated RL schedules are padded with 0.5 s channel‑0 delays, violating the 6 s channel‑0 cooldown; reachable today via `/api/transmit` and `run_sender` with long sequences. |
| 2 | Float32 violation counting in benchmark 3 | **CONFIRMED** (mechanism + recorded artifact) | **P2** | float32 cast → cumsum → strict `<` counts boundary-exact gaps as violations; recorded Hybrid=13 / RL=15 contradict the "guarantee zero" docstring; likely dominated by float32 boundary erosion, not real breaches. |
| 3 | Default data paths point at missing files | **CONFIRMED** | **P2** | `data/human_traffic.json` does not exist; three scripts silently degrade (synthetic training, skipped human baseline, synthetic warden eval). |
| 4 | evaluate_stealth.py unmasked zero-padded warden calls | **CONFIRMED** | **P2** | The advertised evaluation suite scores zero-padded sequences with `mask=None`; scheduler rows (all length 20) are safe, ragged human rows are not. |
| 5 | Bounded-critic propagation gaps | **CONFIRMED** (latent for the shipped checkpoint) | **P2** | CLI final checkpoint omits the two critic-config keys and 4 of 5 re-instantiators never call `apply_critic_config`; the shipped `gan_generator.pt` is unbounded, so nothing differs today — bites on any retrain with `--critic-score-bound`. |
| 6 | C1 positive-control gate recorded but never enforced | **CONFIRMED** | **P2** | `benchmark_1` computes the gate, prints it, records it — and always exits 0; nothing downstream fails on `all_gates_passed=false`. |

### 1 — RL export tail-pad is cooldown-blind — CONFIRMED, P1

Mechanism (all read this session at HEAD):

- `src/stealth/stealth_scheduler.py:498-508` — "Pad if RL ended early", then
  `delays.append(RL_DELAY_MIN); channels.append(0)` for every unsent item.
- `src/stealth/rl/pacing.py:41-42` — `RL_DELAY_MIN = 0.5` / `RL_DELAY_MAX = 20.0`.
- `src/stealth/stealth_scheduler.py:365-367` — enterprise cooldowns `ch0 = 6.0, ch1 = 12.0, ch2 = 4.0`;
  the RL contract is "sent now, the delay is the cooldown that follows"
  (`src/stealth/rl/environment.py:303-305`). A tail of consecutive ≥2 items at 0.5 s on channel 0
  therefore violates the 6 s cooldown for every pad item after the first — the pad is
  cooldown-blind by construction.
- Doc-code contradiction: the pad's own comment claims "the tail stays inside the contract"
  (`stealth_scheduler.py:504-505`) — true for the *pacing window* [0.5, 20.0], false for the
  per-channel cooldown contract; and the RL docstrings promise "guarantee zero rate-limit
  violations" (`stealth_scheduler.py:119`, `:340`).
- Phantom step: on the timeout step the environment returns `done` **without** executing the
  transmission (`environment.py:290-296` — early return before the queue pop at `:331`), yet the
  scheduler appends a delay for it: `info.get("delay", action["delay"])` falls back to the agent's
  un-backoffed proposal (`stealth_scheduler.py:492`) because `info` has no `delay` key on timeout.
  So the exported tail is: one un-backoffed agent-proposed delay for an item the env never sent,
  followed by the 0.5 s × channel‑0 pad. (Counted once with the pad; same code path.)

When the pad fires: only when the RL loop ends with `len(delays) < len(media_ids)`. With inference
action masking (`stealth_scheduler.py:470-471`, mask vetoed channel picks; env mask
`environment.py:454-462`) and the landing-clock back-off (`environment.py:317-324`), the only
realistic early exits are the episode timeout (`environment.py:293-296`, `:325-328`,
`max_episode_time=3600.0` at `:111`) or the `max_steps` bound (`stealth_scheduler.py:462-465`).
Benchmark trials cannot hit either: ≤50 carriers × ≤20 s + back-offs ≪ 3600 s.

Reachability — **this is why the verdict upgrades beyond a pure latent RISK**: both production
callers accept arbitrarily long RL schedules today.

- `src/api/server.py:939-945` — `TransmitRequest` takes free-form `media_ids` and `mode: rl/auto`;
  `server.py:1132-1137` calls `scheduler.schedule(...)` with them; `server.py:1011-1063` transmits
  the returned (possibly padded, cooldown-violating) schedule verbatim in a background thread,
  sleeping `delay / speed_multiplier` (`:1018`, `:1054-1056`).
- `scripts/runtime/run_sender.py:436-442` — encodes an arbitrary message into an arbitrarily long
  `media_ids` sequence (exact_vcp + RS-ECC) and `:454-463` passes it to `schedule(mode=...)`;
  `:226-243` transmits the schedule verbatim.
- The RL checkpoint exists at the default path (`storage/models/rl_agent.pt`), so mode `rl`
  does not silently fall back. The timeout fires once `projected_time ≥ 3600` — at the RL pace
  (≈5.2 s mean per the recorded artifact) that is roughly ≥ 350 carriers; the API imposes no
  length limit, and a long user message reproduced through the encoder easily clears that.

Severity P1: the exported schedule violates the rate-limit contract the scheduler advertises
(`stealth_scheduler.py:119`, `:340`), producing a machine-flaggable burst tail exactly where the
RL engine was supposed to be stealthiest, on a supported API path. Not P0 because no recorded
artifact or default demo flow triggers it (benchmarks use 50-carrier trials).

### 2 — Float32 violation counting in benchmark 3 — CONFIRMED, P2

Mechanism (HEAD):

- `scripts/analysis/benchmark_3_traffic_evasion.py:397`, `:400`, `:413`, `:417`, `:421` — every
  per-mode delay vector is cast to `np.float32` before scoring (`:421` for RL/scheduler rows, 
  `:484` for the canonical delay mapping too).
- `:284` — `timestamps = np.cumsum(delays)` (float32 in → float32 out).
- `:303` — strict comparison `(t - last_tx[ch]) < CHANNEL_COOLDOWNS.get(ch, 6.0)` against the
  integer-ish cooldowns at `:70` (`{0: 6.0, 1: 12.0, 2: 4.0}`).

The executors deliberately land **boundary-exact** gaps in float64: hybrid routing waits until the
earliest channel is exactly free and adds that wait (`stealth_scheduler.py:379-383`,
`delays[i] += wait`), the env back-off extends the delay to exactly the channel-free time
(`environment.py:319-324`), and `scheduler.py:373` admits `>=` (equality = allowed). A float32
cast erodes such exact boundaries: the repro run this session turned the boundary gaps 6.0 and
12.0 into float32 timestamps of **5.999999046325684** and **11.999999046325684**, both then
counted as violations by `:303`. Over 200 randomized wait-fill scenarios, 30 showed spurious
violations under the float32 pipeline where the float64 accounting recorded zero; in one clean
trial float64 accounting saw 1 violation, the float32 pipeline saw 2.

Recorded-artifact cross-check (`storage/benchmarks/component_3_results.json`, provenance
`git_commit=9b02563`, `git_dirty=True`, 2026‑10‑21T19:00:32Z; flag threshold 0.018127655369997026):

| scheduler | trials | cooldown_violations |
|---|---|---|
| Static Cadence | 50 | 2450 |
| Stochastic Poisson | 50 | 1276 |
| DCASS Temporal WGAN | 50 | 982 |
| DCASS Hybrid Engine | 50 | **13** |
| DCASS PPO RL Agent | 50 | **15** |

`git show 9b02563:...benchmark_3_traffic_evasion.py` confirms the float32 casts (old :412‑:420) and
the hybrid wait-fill (old :380‑:383) already existed at record time, so the recorded Hybrid=13 /
RL=15 are consistent with float32 boundary erosion (tens of boundary-exact gaps per 50 trials × 49
gaps could plausibly round down) rather than true cooldown breaches — but they are recorded as 13
and 15, contradicting the scheduler's own "guarantee zero rate-limit cooldown violations"
(`stealth_scheduler.py:340`; also `:119`). The same numbers are quoted as fact in the C3 docs
(`docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md §5` table). Under HEAD the numbers are not
reproducible with real traffic (masked inference + landing-clock env keep float64 gaps
cooldown-clean); whatever residual is real cannot be separated from the round-off — that is precisely
the defect. P2: it corrupts a quality metric and a doc-guarantee claim, not the deployed traffic.

### 3 — Default data paths point at missing files — CONFIRMED, P2

- `scripts/stealth/train_gan.py:42-45` — `--data` default `PROJECT_ROOT/"data"/"human_traffic.json"`
  (help text at `:45` itself admits "synthetic fallback if missing").
- `src/stealth/gan/trainer.py:123-125` — on missing path prints a warning and
  `self.data = self._generate_synthetic_data(1000)` (`:127+`). The shipped
  `storage/models/gan_generator.pt` config shows the generator was trained exactly on this
  synthetic fallback (C3_GENERATOR_ADVERSARY.md §2 documents it — DESIGN, not re-run here).
- `scripts/testing/evaluate_stealth.py:88-89` — `--human-data` default same missing path;
  `load_human_traffic` returns `None` silently when the file is absent (`:30-35`) and the baseline
  block is guarded by `if human:` with **no else** (`:136-144`), so the human baseline disappears
  from the printed summary without any warning.
- `scripts/stealth/eval_warden.py:76-81` — same default, with `:80` help "Must match the --data
  used at train time" (so an eval rerun against the missing default silently compares against a
  different distribution than enumerated in the printed header); `:102` builds the eval dataset
  from it.
- Facts: `data/` contains only `README.md`; the real corpus lives at
  `storage/data/traffic/human_traffic.json` (+ `real_human_traffic.json`); a third writer
  (`generate_traffic_dataset.py`) targets `storage/data/behavioral/human_traffic.json`.
- Doc tie-in: `README.md:185` advertises bare `python scripts/testing/evaluate_stealth.py` as
  "the full stealth and timing evaluation suite", and `docs/QUICKSTART.md:210-216` likewise shows
  the command with only the two checkpoint flags — neither mentions `--human-data`, so the
  advertised suite silently runs without its human baseline. P2 (silent degradation of three
  documented entry points; the corpus itself is only one flagged flag away).

### 4 — evaluate_stealth.py unmasked zero-padded warden calls — CONFIRMED, P2

- `scripts/testing/evaluate_stealth.py:38-49` — `evaluate_warden` pads each sequence to the batch
  max with `0.0` delays / channel `0` and calls `warden(dt, ct)` with **no mask**.
- Receiver side at HEAD: `src/analysis/adversarial/warden.py:254-255` — `mask=None → ones`, so
  every pad zero is treated as a real transmission of zero seconds; the zero rows then enter the
  stat features (`:259-321`: masked mean/var/min/max/range/MAD/skew/kurtosis + adjacent-pair
  features), the transformer runs with `src_key_padding_mask=None` (`:437-442`), and the dual
  pooling averages pad outputs too (`:452-454`, the masked branch `:455-466` is skipped).
- Asymmetry: scheduler rows are scheduled for exactly 20 items (`evaluate_stealth.py:109`, schedule
  length validated equal to `len(media_ids)`), so their batch hits `max_len=20` with no padding —
  unaffected. Human sessions have variable length, so the human baseline is padded to the longest
  session in the batch — its zero rows suppress the min-delay feature and pollute every aggregate.
  The "human baseline bot prob" of the suite is therefore structurally biased.
- Correct protocol exists in-house: `scripts/analysis/benchmark_3_traffic_evasion.py:311-327`
  pads to 64 **with a mask** and even documents why ("score for a given traffic does not depend on
  how the trial was batched"); `train_traffic_detector.py:185-190` does the same. `evaluate_stealth`
  is the outlier.
- Doc tie-in: same as candidate 3 (`README.md:185`, `docs/QUICKSTART.md:212-215`) — this script is
  positioned as the evaluation suite. P2.

### 5 — Bounded-critic propagation gaps — CONFIRMED (latent for shipped artifact), P2

- (a) Final CLI checkpoint omits the critic-config keys: `scripts/stealth/train_gan.py:162-171`
  saves `epoch, global_step, generator_state, warden_state, generator_optimizer,
  warden_optimizer, config, metrics_history` — no `critic_bounded` / `critic_score_bound`,
  whereas the trainer's canonical saver includes them
  (`src/stealth/gan/trainer.py:528-529`) and the trainer's own loader re-applies them
  (`:543` → `apply_critic_config`, defined `:194-206`).
- (b) Re-instantiators that never call `apply_critic_config`: `eval_warden.py:61-64`,
  `evaluate_stealth.py:69-79`, `train_rl.py:43-76` (no `apply` hit at all),
  scheduler `_load_rl_agent`'s warden load (`stealth_scheduler.py:417-433` — memory from the full
  read this session). The only loader that does it is
  `scripts/analysis/measure_rl_stealth_reward.py:28,44`.
- Shipped-checkpoint fact (read-only `torch.load` this session):
  `storage/models/gan_generator.pt` config has `critic_score_bound=None` and the checkpoint carries
  neither top-level key; the warden default is unbounded
  (`warden.py:119`, `:129`, gate at `:483-485` is state-dependent `self.bounded_critic`). So today
  defaults align and the shipped artifact is correctly interpreted wherever the keys are missing.
- Where the latent bug bites: any CLI retrain with `--critic-score-bound <b>` (the flag exists in
  train_gan) writes a bounded critic into a checkpoint format that neither the CLI path nor the
  four re-instantiators restore; downstream loaders would then score with an unbounded head against
  weights trained bounded. Known aggravators read this session: `verify_generator_path.py:193-194`
  silently defaults `bounded=False` when the key is absent, and `track_critic_sensitivity.py`
  never applies the critic config (`:117-118` area). `tests/test_stealth/test_integration.py:49-60`
  replicates the CLI checkpoint format, so the tests lock the incomplete format in. P2, RISK-flag.

### 6 — C1 positive-control gate recorded but never enforced — CONFIRMED, P2

- `scripts/analysis/benchmark_1_media_steganalysis.py:92` — `POSITIVE_CONTROL_GATE = 0.90`;
  per-fold `gate_pass` (`:808`); summary seeds `"all_gates_passed": True` (`:925`) but correctly
  re-flips it per detector (`:974-976` — `if not all(gates): summary["all_gates_passed"] = False`);
  printed `gate=pass|FAIL` (`:1279`).
- Enforcement: none. `main()` returns the summary (`:1215`, returned at the end) and
  `if __name__ == "__main__": main()` (`:1285-1286`) discards the return value; a grep of the file
  finds no `sys.exit` and no gate-driven raise — the process exits 0 on gate FAIL.
- Recorded artifact: `storage/benchmarks/component_1_summary.json` — gate 0.90 at `:20`, per
  detector control accuracy 0.93675 / 0.9505 / 0.93975 with `gate_pass_all: true`
  (`:29-42`, `:169-182`, `:308-322`), and `"all_gates_passed": true` at `:445`. So the recorded run
  passed; docs describe the gate semantics (`docs/benchmarking/01_MEDIA_LAYER_STEGANALYSIS.md:42`
  — foils "failed the 0.90 positive-control gate, excluded from the official summary") and CLAIM_LEDGER
  contains no claim the gate is exit-enforced — no doc contradiction today, the defect is purely
  structural: a rerun that fails the gate would still exit 0 and regenerate every table/plot while
  CI stays green. P2.

---

## Part 2 — New findings (this audit)

| ID | Severity | Verdict | Finding |
|----|----------|---------|---------|
| N1 | P2 | CONFIRMED | **"fixed 30 s" probe row actually scores 20 s traffic.** `scripts/analysis/measure_rl_stealth_reward.py:34` defines the traffic as `[30.0]*20`, but the env clamps every delay into the canonical window before scoring (`environment.py:287` → `canonical_rl_delay`, `pacing.py:41-55`, `RL_DELAY_MAX=20.0`), so the scored traffic is `[20.0]*20` (+ back-offs). The artifact `storage/benchmarks/rl_stealth_reward_probe.json` stores the label, and `docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md:32` quotes "fixed 30 s" as fact. Related: the "fixed 3.0 s metronome" row (`:32`) is also mislabeled — the step loop rotates channels `i % 3` with cooldowns 6/12/4 s (`:57`, scheduler `:366`), so the env back-off (`environment.py:319-324`) extends several of the 3.0 s delays (e.g. every 3rd item on ch1 needs 12 s), and the executed sequence is not a metronome. The reward-ordering conclusion is unaffected; the labels are wrong. |
| N2 | P2 | CONFIRMED | **Dead retry branch + stale comments in the RL loop.** `stealth_scheduler.py:474-480` retries on `reason == "rate_limit_violation"`, but the env can no longer emit that reason: a cooling channel is now a back-off, not a rejection (`environment.py:307-324`), and the in-loop comment admits it ("should no longer occur here", `:469`). The pad block's justification copy-pastes the old reason ("A rate-limit violation consumes a loop iteration…", `:500-501`) while the *actual* cause of an early exit today is the timeout. Misleads maintainers, and — as shown in Part 1 #1 — the pad it justifies is cooldown-blind. |
| N3 | P2 | CONFIRMED (latent) | **Diagnosis tooling silently misreads a bounded-critic checkpoint.** `scripts/analysis/verify_generator_path.py:193-194` defaults `bounded=False` when the checkpoint keys are absent (true for the CLI save format, Part 1 #5a) and never applies the stored config; `track_critic_sensitivity.py:117-118` instantiates the warden without `apply_critic_config` either. On today's shipped (unbounded) ckpt the outputs are correct; after any bounded retrain the path-verification and sensitivity reports would mischaracterize the model. Related to Part 1 #5. |
| N4 | P2 | CONFIRMED (latent) | **TimingSchedule.delays is documented as seconds but carries raw band values.** `src/stealth/gan/generator.py:28` ("Inter-transmission delays in seconds") vs what `forward()` actually fills (`:248-257`, `raw_delay_band_transform` output in the [0.49933, 0.519] band per `delay_contract.py:12-27`). `generate_stream` (`:273-297`) yields those raw values as `(delay, channel)` — a caller that sleeps them gets a constant ≈0.5 s metronome. Grep shows only tests call `generate_stream` today, hence latent; the scheduler never uses it (it applies `canonical_delay_mapping` itself). Docstring fix + rename or map-to-seconds needed. |
| N5 | P2 | CONFIRMED (numbers already artifact-recorded and doc-quoted) | **"guarantee zero rate-limit cooldown violations" docstring is contradicted by the recorded C3 artifact** (Hybrid=13, RL=15 in `component_3_results.json`, quoted in `C3_TRAFFIC_DETECTOR_EVALUATION.md §5`). Attributable in large part to the float32 counter (Part 1 #2), but the docstring is written as an unconditional guarantee about the export — on HEAD a real (non-round-off) violation is reachable via the Part 1 #1 pad path. Severity folded into #1/#2 fixes; listed separately because the docstring + doc §5 both need edits, not just the counter. |
| N6 | — | DESIGN (noted, not pursued) | Secondary skims clean: `src/api/server.py` transmits the schedule faithfully (`:1011-1063`), keeps noise gaps ("None") as waits (`:1016-1021`), skips the last item's delay as designed (`:1055`, matching `pipeline._transmit`), and its displayed duration matches the thread's behavior (`:1018`, `:1143`) — no mask/behave correctness bug found; concurrency stays with the C5 analyst. `src/engine/dssc_encoder.py`: the candidate-pool fallback chain guarantees ≥8 candidates before the ≥2-bit symbol read (`:217-235`), the bit reader zero-pads deterministically (`:79-95`), and the pool caches are keyed per (modality, family) (`:190-232`) — no mask/correctness defect; note for the C2 analyst: the "ultimate safety fallback" `all_ids[:256]` (`:228-229`) drops the family/cluster constraint when pools run thin, weakening topic-consistency (decodability unaffected). |
| N7 | — | RISK (flagged, low) | **All-masked action-mask fallback can still produce a landing-time mask lie.** `environment.py:459-461` returns all-ones when every channel is cooling; the policy may then pick a cooling channel, the env back-offs it (`:319-324`) so the *landing gap* stays cooldown-legal, but the episode timeout may trigger from the extended delay (`:325-328`) — i.e. the fallback feeds the Part 1 #1 pad path earlier than the ~350-carrier estimate. Recorded artifacts don't exercise it; no separate fix beyond #1 (timeout handling) and the mask could honestly report "none available" to the caller. |

---

## Fix batch (proposed; nothing changed in the repo)

1. **FIX-A (P1 → Part 1 #1, N5):** Make the RL pad cooldown-safe and honest. Concretely: on
   `reason == "timeout"` (a) do **not** append the phantom step's delay — only env-confirmed
   transmissions go on the wire (`stealth_scheduler.py:492`), and (b) finish the remaining items
   with the cooldown-aware router (reuse the hybrid wait-fill logic `stealth_scheduler.py:373-386`)
   instead of `RL_DELAY_MIN` × channel 0 (`:506-508`). Then repair the claim surface:
   `:119`, `:340` ("guarantee zero") and `:500-505` ("inside the contract", "rate-limit
   violation consumes an iteration") must either be true or say exactly what they exclude.
2. **FIX-B (P2 → #2):** Keep timing in float64 end-to-end in `benchmark_3_traffic_evasion.py`
   (drop the `astype(np.float32)` at `:397-421` for the *scoring* path; keep float32 only where
   the detector interface genuinely requires it, `:315-327`), so boundary-exact gaps measure as
   satisfied. Re-run C3; if real residual violations remain, they then measure honestly.
3. **FIX-C (P2 → #3):** Repoint the three defaults to the real corpus
   (`train_gan.py:44`, `evaluate_stealth.py:88-89`, `eval_warden.py:78-80`) or route them through
   one resolver (traffic path) + make `evaluate_stealth` print "HUMAN BASELINE SKIPPED (file
   missing)" instead of silently omitting (`:136-144`).
4. **FIX-D (P2 → #4):** Pass the validity mask in `evaluate_warden`
   (`evaluate_stealth.py:40-49`), mirroring `benchmark_3:311-327`, so padded batches stop injecting
   zero-delays into the warden features.
5. **FIX-E (P2 → #5, N3):** Add `critic_bounded`/`critic_score_bound` to the CLI final checkpoint
   dict (`train_gan.py:162-171`) and call `apply_critic_config` in `evaluate_stealth`,
   `eval_warden`, `train_rl`, the scheduler warden load, `verify_generator_path.py`, and
   `track_critic_sensitivity.py`. Then update `tests/test_stealth/test_integration.py:49-60` to the
   complete format.
6. **FIX-F (P2 → #6):** Enforce the C1 gate at exit: after printing, `sys.exit(0 if
   summary["all_gates_passed"] else 1)` in `benchmark_1_media_steganalysis.py` (keep the artifact
   writing unchanged).
7. **FIX-G (P2 → N1):** Fix the probe labels to the canonical window they actually score
   ("fixed 20.0 s (30 clamped)", and document the back-off distortion on the metronome row), then
   sync `C3_TRAFFIC_DETECTOR_EVALUATION.md:32` (and the artifact note).
8. **FIX-H (P2 → N2):** Delete the dead retry branch or keep it guarded with a comment that states
   the env no longer emits that reason; fix the pad-block justification.
9. **FIX-I (P2 → N4):** Rename/clarify `TimingSchedule.delays` (raw band) or make `forward()`
   return mapped seconds; same for `generate_stream`. Update the docstring at `generator.py:28`.

---

## NOTES — documented-by-design deviations (not counted as defects)

- **env `get_warden_score` mean-padding (P1‑6).** Short histories are padded with the running mean
  delay, not zeros, and the docstring explains why zeros would bias the logged metric
  (`environment.py:464-485`). Documented deviation — acceptable; contrast Part 1 #4, where
  `evaluate_stealth` does the *opposite* (zero, unmasked) with no documentation.
- **Last-item delay not slept.** Server thread and pipeline both skip the delay *before* the next
  packet after the final item (`server.py:1052-1055`; `src/stealth/pipeline.py` `_transmit`,
  read this session) — delay semantics are "the cooldown that follows" the send
  (`environment.py:303-305`), so a final terminal snooze is meaningless. Consistent by design.
- **`generate_stream` raw-band yields.** Only tests consume it (grep this session); the production
  scheduler maps raw → seconds itself. Covered as doc mismatch N4, not a live bug.
- **env timeout costs the step.** On timeout the episode ends with a −50 reward and the item is
  never sent (`environment.py:290-296`, re-verified); the C3 docs describe the agent as
  throughput-preferring rather than stealth-preferring. Design as documented; the *export* side of
  it is the Part 1 #1 defect.
- **Static/Poisson/WGAN cooldown counts in the artifact** (2450/1276/982) are recorded
  measurements of those baselines at `9b02563` and are described as "recorded, not scored" in the
  C3 docs; only the RL/Hybrid counter contradiction is treated as a defect (Part 1 #2/N5).
- **`select_threshold` conservatism** (`src/analysis/detector_scoring.py:38-73`, read this session):
  measured FPR kept ≤ budget − 1/n by construction; consistent between trainer and benchmark. No defect.
- **dssc encoder third fallback** `all_ids[:256]` weakens topic-family purity under
  thin-family conditions (`dssc_encoder.py:228-229`) — flagged for the C2 analyst, decodability
  unaffected (see N6).

---

## Executive summary (≤10 lines)

1. Part 1 candidate 1 upgraded and **CONFIRMED P1**: the RL export tail-pad (`stealth_scheduler.py:506-508`)
   is cooldown-blind (0.5 s × ch0 vs 6 s cooldown), its comment claims the opposite, and it is
   reachable today via `/api/transmit` and `run_sender` with long sequences (phantom timeout-step
   delay included).
2. Candidate 2 **CONFIRMED P2**: float32 cast→cumsum→strict-`<` counts boundary-exact, cooldown-legal
   gaps as violations; recorded Hybrid=13/RL=15 (artifact @ dirty `9b02563`) contradict the
   "guarantee zero" docstring; mostly round-off, not real breaches.
3. Candidates 3, 4, 6 **CONFIRMED P2**: missing default data path (silent degradation of three
   entry points incl. the README/QUICKSTART-advertised suite), unmasked zero-padded warden calls in
   that same suite (asymmetric: human rows distorted, 20-item scheduler rows not), and a C1 gate
   that is recorded/printed but can never fail the process exit code.
4. Candidate 5 **CONFIRMED but latent P2**: CLI checkpoint omits the critic-config keys and 4/5
   re-instantiators never re-apply them; the shipped unbounded `gan_generator.pt` currently aligns.
5. New Part 2 findings: 4 confirmed P2 (mislabeled "fixed 30 s" probe actually scores 20 s; dead
   retry branch + stale pad comments; verify/track tools silently default unbounded;
   TimingSchedule docstring says seconds, carries raw band) + 1 recorded-doc contradiction (N5) +
   1 low RISK (all-masked fallback feeds the pad path) + clean secondary skims (server, dssc encoder).
6. Totals: Part 1 = 4 CONFIRMED, 0 REFUTED, 1 CONFIRMED-latent (#5), 0 DESIGN (delegation to NOTES);
   Part 2 = 4×P2 new, 1×RISK. Most important new finding: **N1** (the flagship reward-probe artifact
   and its doc quote score 20 s traffic under a "fixed 30 s" label).

# Fix & Decision Log

**Scope:** Every fix and design decision made across commits `986fe5d` (pre-audit refactor) through `14abc4e` (implementation-plan completion), spanning three work phases:

| Phase | Commits | Trigger |
|---|---|---|
| A - Audit remediation | `0ebdaee`, `18b5739`, `f920cc9`, `970fd11` | CODEBASE_AUDIT.md (B/H/M/L + research R/P1/P2 findings) |
| B - Namesake feature + cleanup | `bcc68c6` | User decisions on audit round 2 |
| C - Implementation plan execution | `14abc4e` | IMPLEMENTATION_PLAN.md (WP-1..9, Decisions 1-5) |

**Entry format:** Finding · root cause → fix → why this approach (rejected alternatives) → files → guarding test.

---

## 1. Engine correctness

### E-1 · ECC exception swallowing (`ecc.py`)
- **Broken:** `except (reedsolo.ReedSolomonError, Exception)` caught everything; docstring falsely claimed "Guarantees 0% BER / 100% exact recovery".
- **Fix:** Catch only `reedsolo.ReedSolomonError`; docstring states the true guarantee (correction up to t = floor(R/2) byte errors).
- **Why:** Swallowing all exceptions hides genuine bugs (e.g., a TypeError inside decode would masquerade as "uncorrectable corruption"). RS-ECC is bounded-capacity by mathematics; claiming otherwise would not survive paper review.
- **Files:** `src/engine/ecc.py`
- **Guarded by:** `tests/test_engine/test_ecc.py` (existing suite).

### E-2 · Sentinel score silently corrupting carrier ranking (`vcp_payload.py`)
- **Broken:** `_vector_score()` returned constant `1.0` on any exception — a failing candidate ranked TOP of the cluster, deterministically choosing broken carriers.
- **Rejected alternative:** return random scores (non-deterministic, unreproducible encodes). Chosen fix: neutral fallback derived from stored metadata score, plus logged warning.
- **Why:** ranking failures must never *promote* candidates; neutral demotion keeps encoding correct when scoring degrades (e.g., CLAP unavailable).
- **Files:** `src/engine/vcp_payload.py`
- **Guarded by:** engine test suite (all exact_vcp roundtrips exercise scoring paths).

### E-3 · Binary payloads corrupted by UTF-8 round-trip in RS decode (`ecc.py`)
- **Broken:** `decode()` returned `str` via `errors="replace"` — framed payloads contain arbitrary header bytes (version/length/CRC), so re-encoding the string did not reproduce the original bytes.
- **Fix:** Added `decode_bytes()` returning raw bytes; `decode()` now delegates to it.
- **Rejected alternative:** base64-wrapping framed payloads before RS (wastes ~33% carriers).
- **Files:** `src/engine/ecc.py`
- **Guarded by:** `tests/test_engine/test_payload_framing.py::TestPayloadFraming`.

### E-4 · Payload framing absent (Decision 4 / WP-6)
- **Problem:** RS success alone has a small false-accept rate; without an independent integrity check, a wrong epoch/key could decode to *plausible garbage*. Also no explicit length field or version for future format evolution.
- **Fix:** `src/engine/payload_framing.py` — `[version u8 = 0x01 | length BE u16 | CRC-16/CCITT | body]`. Decoder accepts an epoch candidate only when **RS succeeds AND CRC verifies**; legacy unframed payloads still decode via version-byte sniffing.
- **Rejected alternatives:** HMAC over payload (heavier, needs key material in codec layer — CRC suffices for epoch detection); magic-string header (collides with real text).
- **Cost:** exactly 5 carriers per message (documented in module 08).
- **Files:** `src/engine/payload_framing.py` (new), `encoder.py`, `decoder.py`, `ecc.py`
- **Guarded by:** `test_payload_framing.py::TestPayloadFraming` (7 tests incl. corruption detection, truncation rejection, legacy compat).

### E-5 · Plaintext leaked through carrier selection (Decision 5 / WP-6)
- **Broken (research-level finding):** exact_vcp ranked carriers within each cluster by similarity to `semantic_chunks[idx % n]` — text **derived from the secret message**. An adversary suspecting the channel could recover the message's *topic* by clustering transmitted carriers, without breaking the code at all.
- **Fix:** Ranking query is now a cover story: explicit `cover_story` param > deterministic per-epoch decoy topic (HMAC of epoch id into a topic list) > bland neutral constant.
- **Why:** The tension is topical coherence (human feeds are about something) vs plaintext independence (carriers reveal nothing). Cover stories keep coherence while severing correlation with the secret. Small code change, large security payoff — adopted from IMPLEMENTATION_PLAN Decision 5 verbatim.
- **Rejected alternative:** uniform-random carrier choice inside clusters (kills coherence entirely; feed becomes visibly incoherent).
- **Files:** `src/engine/encoder.py`
- **Guarded by:** full engine roundtrip suite (framing change reuses the same selection path).

---

## 2. Corpus / indexing

### C-1 · O(N) media lookup on every decoded item (`unified_index.py`)
- **Broken:** `get_by_id()` linear-scanned every metadata row per call — O(N·M) during decode of M-carrier messages (~153k rows × message length).
- **Fix:** `_id_lookup: dict[str, tuple[Modality, int]]` built once at load time; O(1) lookups.
- **Files:** `src/corpus/index/unified_index.py`
- **Guarded by:** decoder unit tests using mocked indexes with side-effect lookups.

### C-2 · Dishonest file paths returned as carriers (M-16)
- **Broken:** `MediaItem.file_path` fell back to returning `image_metadata.json` / arbitrary `.arrow` shards / `dataset_info.txt` as if they were the carrier files — UI displayed missing corpus items as present.
- **Fix:** Returns `None` when no genuine carrier file exists on disk. Result cached after first resolution (was doing filesystem rglob on every access).
- **Why (strategic):** converts a storage problem into a correctness improvement — the system runs honestly on a 43 GB budget without raw media downloads (IMPLEMENTATION_PLAN constraint).
- **Rejected alternatives:** downloading raw media at runtime (violates disk budget); placeholder images (dishonest in a different way).
- **Files:** `src/corpus/index/unified_index.py`
- **Guarded by:** `tests/test_engine/test_encoder.py::TestEncoderIntegration::test_real_encode_file_paths`.

### C-3 · Hardcoded score calibration constants (M-17)
- **Broken:** z-score+sigmoid calibration used empirical constants measured once (2026-02-17); `update_calibration()` existed but nothing ever fed it.
- **Fix:** `ScoreNormalizer` auto-loads `storage/data/indices/score_calibration.json` if present (falls back to defaults); `scripts/analysis/calibrate_scores.py` measures per-modality raw-score distributions against the live indices (30 probe queries) and writes it. Artifact generated for the shipped corpus.
- **Measured vs old constants:** image 0.295±0.027 (old 0.271±0.028), text 0.906±0.023 (old 0.885±0.053 — std was 2.3x off), audio 0.115±0.018 (old 0.100±0.021).
- **Files:** `src/corpus/index/unified_index.py`, `scripts/analysis/calibrate_scores.py` (new)
- **Guarded by:** normalizer exercised by every search; script run recorded in repo history.

### C-4 · Codebook ↔ index binding unchecked (WP-3 / Decision 2 — highest-value item)
- **Broken:** rebuilding any FAISS index without re-fitting the codebook shifted **every global offset silently** — all bytes wrong, no error anywhere. The count check alone cannot catch same-length rebuilds.
- **Fix:**
  - `scripts/cluster/bless_codebook.py`: writes `voronoi_codebook.meta.json` recording per-modality `ntotal` + cheap content fingerprint `sha256(reconstruct(0) ‖ reconstruct(ntotal-1) ‖ ntotal)`; `--bless` gated behind the Phase-0 count check; `--check` verify mode.
  - `VCPPayloadMapper.load()` **refuses to run** on fingerprint mismatch (RuntimeError naming the remedy). Check skipped only for injected in-memory codebooks (tests).
- **Rejected alternative:** monotonic version integer (nothing increments it — the exact failure mode remains); hashing all 314 MB every load (microsecond-cheap two-vector fingerprints chosen instead).
- **Honesty note:** the first pairing is *asserted* (blessed), not proven — the codebook predates the mechanism. Documented in the bless output itself.
- **Files:** `scripts/cluster/bless_codebook.py` (new), `src/engine/vcp_payload.py`
- **Guarded by:** `tests/test_diagnostics/test_doctor.py::TestFingerprintBinding::test_rebuilt_index_is_refused` — perturbs one vector at identical ntotal and proves refusal.

---

## 3. API / server

### A-1 · Invalid CORS configuration
- **Broken:** `allow_origins=["*"]` combined with `allow_credentials=True` (invalid per CORS spec; effectively reflected-origin with credentials — insecure).
- **Fix:** Explicit origin allowlist from `DCASS_CORS_ORIGINS` env (default localhost dev origins), credentials kept.
- **Files:** `src/api/server.py`

### A-2 · Lazy singleton warm-up race
- **Broken:** concurrent first requests could both see `_encoder is None` and construct/load CLIP+FAISS twice (memory spike, wasted minutes).
- **Fix:** double-checked locking under `_engine_lock`.
- **Files:** `src/api/server.py`

### A-3 · Stop endpoint reported success but did nothing (audit H-4)
- **Broken (twice):** original code set a flag the transmit loop never read; our first fix introduced `_transmission_stop_requested` but **forgot it in the `global` statement**, making it a local variable — the auditor caught our regression. Recorded here deliberately.
- **Fix:** declared global; worker checks the flag between packets and sleeps in ≤250 ms slices so stop takes effect promptly; status reports "stopped".
- **Lesson documented:** every new module-level flag needs its `global` declaration reviewed.
- **Files:** `src/api/server.py`

### A-4 · TOCTOU double-start race on transmissions
- **Broken:** `_transmission_active` checked outside any lock; two simultaneous POSTs both passed.
- **Fix:** atomic claim under `_transmission_lock`; claim released if scheduling fails before worker takeover.
- **Files:** `src/api/server.py`
- **Guarded by:** `tests/test_api_endpoints.py` (stop/status lifecycle endpoints).

### A-5 · `/api/status` re-read all FAISS indexes from disk per call
- **Fix:** counts cached keyed on (path, size, mtime); only re-reads when artifacts actually change.
- **Files:** `src/api/server.py`

### A-6 · DELETE wire packets destroyed sender control files
- **Broken:** DELETE globbed all `*.json` including `_manifest.json`; GET skipped underscore-prefixed files — inconsistent semantics.
- **Fix:** DELETE skips `_`-prefixed files, matching GET.
- **Files:** `src/api/server.py`

### A-7 · Side-channel codeword exposed in responses (P2-12)
- **Broken:** `raw_codeword_hex` carried the payload *outside* the media sequence even in exact_vcp mode — inviting the reviewer question "so is the payload in the media or not?"
- **Fix:** field marked DEBUG ONLY in schema docs; suppressed unless `payload_mode == "semantic_legacy"`.
- **Rejected alternative:** deleting outright (frontend type references; legacy mode still needs it).
- **Files:** `src/api/server.py`

### A-8 · URL convention contradiction (audit H-5)
- **Broken:** `lib/api.ts` treated `NEXT_PUBLIC_API_URL` as base-with-`/api`; wire page appended `/api/...` again (double prefix, guaranteed 404s); docker-compose injected a container-internal hostname (`http://dcass-api:8000`) into browser-baked env.
- **Fix:** one convention — env holds the ORIGIN, client appends `/api`; wire page imports `API_BASE` from lib; compose uses `http://localhost:${API_PORT}` (browser-resolvable).
- **Files:** `frontend/src/lib/api.ts`, `frontend/src/app/wire/page.tsx`, `docker-compose.yml`
- **Guarded by:** `npx tsc --noEmit` clean.

---

## 4. Runtime pipeline (sender / receiver)

### P-1 · Receiver decoded without ECC that the sender applied
- **Broken (critical):** sender encoded with RS parity; receiver called `decoder.decode(media_ids)` defaulting `use_ecc=False` — parity bytes decoded as message content. ECC was decorative end-to-end.
- **Fix:** receiver decodes with `use_ecc=True` for exact_vcp; sender explicitly encodes `payload_mode="exact_vcp", use_ecc=True`; manifest records `payload_mode`; receiver honors it.
- **Files:** `scripts/runtime/run_receiver.py`, `run_sender.py`
- **Guarded by:** `tests/test_engine/test_exact_vcp_recovery.py` (tier-4: corrupt ≤ t carriers through the real VCP path, assert exact recovery; beyond-capacity must fail closed).

### P-2 · Partial-write race permanently lost packets
- **Broken:** receiver marked files processed even when JSON parse failed (sender still writing) — packet gone forever.
- **Fix:** bounded retry (5 attempts) before giving up; successful-parse-only marking.
- **Files:** `scripts/runtime/run_receiver.py`

### P-3 · No duplicate/out-of-order protection in reassembly
- **Fix:** sequence-number dedup set; manifest-driven `expected_total` with silence-threshold fallback (GAN delays can exceed the old fixed threshold, causing premature decode).
- **Files:** `scripts/runtime/run_receiver.py`

### P-4 · Noise controller dropped payload bytes (static schedule path)
- **Broken:** `skip_prob` randomly dropped media items — but in exact_vcp **every ID is one payload byte**. Dropping = corrupting. Designed for the image-sequence demo where drops were harmless.
- **Rejected alternatives:** re-request dropped items (channel has no feedback path); FEC-only mitigation (already have RS). Chosen fix: skips disabled by default behind explicit `drop_packets=True` opt-in for legacy demos; jitter/idle gaps retained (safe).
- **Files:** `src/distribution/noise.py`
- **Guarded by:** distribution test group.

### P-5 · Scheduler delay semantics contradicted NoiseController (H-6)
- **Broken:** Scheduler slept *before* dispatching item i; NoiseController folds idle gaps into `delays[i]` meaning *after* item i — combining them shifted every gap by one position.
- **Fix (per plan recommendation):** unified on "after" (`delays[i]` = pause AFTER item i); Scheduler dispatches then sleeps; documented on both classes; API transmitter already matched.
- **Files:** `src/distribution/scheduler.py`, `src/distribution/noise.py`

### P-6 · Deprecated `datetime.utcnow()`
- **Fix:** timezone-aware `datetime.now(timezone.utc)`; scheduler stamps actual send time, not pre-computed intent.
- **Files:** `src/distribution/scheduler.py`, `base_channel.py`

### P-7 · Encode hung indefinitely on torch < 2.6 (CLAP CVE restriction)
- **Broken (root cause found by instrumentation):** CLAP `from_pretrained` raises under torch<2.6's `torch.load` restriction; the failure wasn't cached, so **every audio-candidate scoring re-attempted a full model download/load** — `/api/encode` appeared frozen.
- **Fix:** fail-once caching (`_clap_failed`), single loud warning, graceful degradation to CLIP embeddings for audio queries (ranking quality reduced, correctness unaffected).
- **Rejected alternative:** hard-failing on CLAP absence (would make the whole channel unusable on this environment).
- **Environment note:** upgrade to torch ≥ 2.6 or safetensors CLAP weights to restore native CLAP ranking.
- **Commit:** `f920cc9`
- **Guarded by:** encode completing in seconds (regression visible as timeout).

---

## 5. Stealth model fixes (code-complete; training pending — see TRAINING_HANDOFF.md)

### S-1 · WGAN-GP constrained the wrong function (R-22)
- **Broken:** gradient penalty computed `grad(bot_probability)` = sigmoid(D). Since σ′ ≤ 0.25, ‖∇σ(D)‖→1 does NOT enforce ‖∇D‖≈1 — the Lipschitz constraint was essentially unsatisfiable. Two of three training scripts ran with this live.
- **Rejected alternative:** removing GP entirely (Wasserstein distance degenerates without it).
- **Fix:** penalty on `feature_importance["raw_critic_score"]`.
- **Files:** `src/analysis/adversarial/warden.py`
- **Tier-3 proof:** `test_penalty_computes_and_backprops`, `test_raw_score_gradient_not_sigmoid_squashed` (CPU, seconds).

### S-2 · Generator's channel head never trained (R-23)
- **Broken:** `sample_channels()` = softmax → argmax. Argmax is non-differentiable ⇒ zero gradient to `channel_head` forever. Any claim of "GAN-learned platform switching" was unsupported.
- **Fix:** straight-through Gumbel-Softmax (`F.gumbel_softmax(hard=True)`) — hard one-hot forward, soft backward. Warden `forward` extended to accept soft channel distributions (weighted embedding sum), since Embedding lookups alone can't pass gradients.
- **Rejected alternative:** REINFORCE-style score-function estimators (high variance for marginal gain here).
- **Files:** `src/stealth/gan/generator.py`, `trainer.py`, `warden.py`
- **Tier-3 proof:** `test_generator_channel_head_gets_nonzero_grad` asserts nonzero `channel_head` gradients end-to-end.

### S-3 · Critic loop backpropped through the generator 5× per step (R-24)
- **Broken:** `fake_schedule.delays` undetached in warden loop — every critic step built a graph through G and discarded it.
- **Fix:** `.detach()` before critic forwards.
- **Tier-3 proof:** source-inspection guard `TestCriticLoopDetachesGenerator`.

### S-4 · Five permanently-zero RL network inputs (P1-5)
- **Broken:** `_compute_state_dim()` returned 21 (claimed "1 time + 10 history") but `_get_state()` wrote 16 slots (sin+cos cyclical = 2; 4 history features). Five network inputs were always zero.
- **Rejected alternative (plan-recommended):** invent six history features (burstiness, peak-hour, autocorrelation) to fill 21 — unjustifiable extra surface in review. Chosen: honest 16.
- **Files:** `src/stealth/rl/environment.py`

### S-5 · Zero-padded warden history biased the logged metric (P1-6)
- **Broken:** early episodes padded delay windows with zeros — an artificial burst pattern the warden reads as bot-like. (Clarified per plan: biases the *logged metric*, not the learning signal.)
- **Fix:** pad with running mean delay.
- **Files:** `src/stealth/rl/environment.py`

---

## 6. Infrastructure, dependencies, dead code

### I-1 · Missing dependencies (B-1/B-2)
- `reedsolo` undeclared yet imported by the engine; `fastapi`/`uvicorn[standard]`/`bert-score` used but undeclared; Dockerfile got them transitively via requirements.txt install stage. All declared; compose healthcheck switched from curl (absent in slim image) to python stdlib.

### I-2 · Dead code removal (L-19, H-11 — user decision)
- **Deleted:** `config/` YAML layer + both competing `Config` classes (zero consumers, contradicted code in 8 places — user chose deletion over the plan's wire-10-keys option), `src/embeddings_legacy/` (superseded pipeline incl. broken-import step8 that CLI referenced), orphaned embedders (`clip_embedder/text_embedder/vector_engine` — superseded by `image_embedder.ImageEmbedder`), duplicate trainers `scripts/training/*` (three different `use_gradient_penalty` settings = untrustworthy results), `scripts/demos/*`, `tools/`, `FE_DCASS/` (superseded Flask demo UI), second chunker in `preprocessors/`, legacy `run_pipeline.py`.
- **Kept deliberately:** `semantic_legacy` payload mode — reframed as the paper's ablation baseline ("what if you encode meaning instead of bytes"), not a transport channel.
- **Repointed:** docker-compose services to canonical `scripts/stealth/*` entry points.

### I-3 · Project tooling absent (L-20)
- **Added:** `pyproject.toml` (pytest/ruff/black config, package metadata) and `.github/workflows/tests.yml` (python tests + frontend typecheck). Repo lint driven to genuinely clean (1315 errors → 0; E402 tolerated in standalone scripts via per-file-ignores since they manipulate sys.path).

### I-4 · Frontend wire view 404
- **Broken:** page lived in `src/archive/wire/` outside Next.js App Router while home linked to `/wire`.
- **Fix:** moved to `app/wire/page.tsx`. Guarded by tsc + route existence.

### I-5 · Stale paths everywhere (H-7/H-8)
- CLI status hardcoded `<repo>/data/indices`; tests/conftest and engine tests likewise; benchmark wrote to `data/benchmarks/results` while the API read `storage/data/benchmarks/results`. All swept onto `resolve_indices_base_path()` / storage layout. Guarded by doctor + existing suites.

---

## 7. Design decisions register

### D1 — `dcass doctor` as the first artifact (WP-2)
**Chosen:** one command validating dependencies, index/metadata agreement, codebook binding, cluster histogram, checkpoints, disk footprint; exits non-zero on hard failure; mirrored at `/api/doctor`.
**Why:** replaces scattered path fixes; is the instrument producing Phase-0 numbers; makes the remote training handoff verifiable ("if doctor isn't READY, don't start").
**Measured outcome (Phase 0):**
- `sum(ntotal)=153,281 == len(cluster_assignments)=153,281` → stop-the-line PASS (the codebook matches these indices)
- Empty clusters K_empty = 0; min density = 48 carriers
- All 256 clusters usable → K′=256
**Consequences:** M-13 (empty-cluster fatalism) and M-14 (repeat-byte exhaustion) retired as non-issues on this corpus — loud error paths retained for future corpora; base-K′ transport (Decision 3's branch) correctly *not* built.

### D2 — Content fingerprints, not version numbers (WP-3)
See C-4. **Rejected:** version integers (nothing increments them); full-content hashes (cost). **Accepted cost:** first pairing asserted via `--bless`, honestly labeled.

### D3 — Keyed bijection over the usable cluster set (WP-6 / H-12)
**Chosen:** interpretation-layer permutation P (byte b → cluster P[b]), derived per epoch. Keys rotate without touching the codebook or corpus.
**Rejected:** refitting/re-distributing the codebook per epoch (massively operationally heavier); base-256→base-K′ conversion (Phase 0 measured K′=256, unnecessary — implemented as a loud validation instead).
**Key derivation:** HMAC-SHA256(secret?, canonical epoch material) → PCG64 seed → permutation of 256. Sources: time bucket (default, offline) + CoinGecko BTC price quantized to $100 (opt-in, coarse, pinned endpoint).
**Security honesty (documented in-module):** without a shared secret this is time-bucket obfuscation defeating casual static decoding; with one, it is keyed steganography. This distinction is stated wherever the feature is claimed.
**Epoch sync risk mitigated by D4:** candidate-window search accepting on RS+CRC, covering clock skew and boundary-crossing sends.

### D4 — Frame the payload; RS as the epoch oracle
See E-4. **Rejected:** relying on RS alone (false-accept rate); transmitting epoch ids out-of-band as a requirement (hint supported but optional).

### D5 — Cover-story carrier ranking
See E-5. Connects to `docs/research/RESEARCH_LLM_NARRATIVE_COHESION_GUARD.md` for the next phase: LLM-generated coherent narratives as programmable cover stories.

### D6 — Fail-closed philosophy throughout
Wrong secret/epoch/garbage decode returns `ecc_success=False`, never plausible-looking text (asserted by `test_wrong_secret_fails_closed`, `test_no_context_manager_fails_on_keyed_traffic`). Fingerprint mismatch refuses loudly. Cluster exhaustion names its cause (empty vs drained).

---

## 8. Deviations from the plans

| Plan said | We did | Why |
|---|---|---|
| H-11: wire ~10 config keys, delete rest | Deleted entire config layer | Explicit user decision; zero-consumer config documenting nonexistent features was judged pure liability |
| WP-4 frontend renders null-path state | Backend honesty done; frontend rendering deferred to Track B roadmap | Scope control |
| WP-8 consolidate trainers to ONE entry point | Deleted duplicates; `train_gan.py` + `train_rl.py` (+`train_gan_extended`) remain in `scripts/stealth/` | Full consolidation folded into Phase R checklist |
| Decision 3 branch table | Measured K′=256 → simpler branch taken, conversion explicitly not built | Plan instructed measure-first |

---

## 9. Verification state at close (commit `14abc4e`)

- **167 tests passing** across tiers:
  - Tier 1 (no artifacts): chunker/ECC/framing/permutation/header-CRC/gradient-flow proofs
  - Tier 2 (indices+codebook): real VCPPayloadMapper offset reconstruction, fingerprint refusal, doctor Phase-0 assertions
  - Tier 3 (CPU seconds): R-22/R-23/R-24 trainer-fix proofs — the teammate's pre-training gate
  - Tier 4 (end-to-end): encode→corrupt≤t→decode recovery through the real VCP path; wrong-key fail-closed
- ruff: clean (from 1,315 findings). TypeScript: clean. `dcass doctor`: VERDICT READY.

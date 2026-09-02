# DCASS Implementation Plan

**Companion to:** [CODEBASE_AUDIT.md](./CODEBASE_AUDIT.md) · **Base commit:** `18b5739` · **Status:** plan only, nothing implemented

## Constraints this plan is built around

| Constraint | Consequence for the plan |
|---|---|
| Indices + metadata now available locally; codebook path incoming | Phase 0 is *measure*, not code. Three design decisions below are unanswerable until we do. |
| **No GAN/RL training on this machine** — teammate runs it | Every model-side fix must be code-complete and provable by a **CPU gradient-flow test in seconds**, never by a training run. The handoff artifact is as important as the fix. |
| **43 GB SSD** | Never download raw media. Runtime footprint target: **< 1 GB**. Make path resolution honest instead of chasing files. |
| Code changes here → teammate runs | One canonical training entry point. Four scripts with three different `use_gradient_penalty` settings is how you get a result nobody can trust. |

---

## Phase 0 — Ground truth (no code, ~30 min)

Everything downstream branches on these numbers, so this comes first.

1. Place files at `storage/data/indices/{image,text,audio}.index` + `*_metadata.json` and `voronoi_codebook.npz`.
2. Record, per modality: `index.ntotal`, `index.d`, `len(metadata)`, file size.
3. Record from the codebook: `num_clusters`, `dim`, `len(cluster_assignments)`, `delta_margin`.
4. **Compute the cluster population histogram** — how many of the 256 clusters have 0, 1–4, 5–20, 20+ carriers.
5. Check `sum(ntotal)` **==** `len(cluster_assignments)`, and `len(metadata) == ntotal` per modality.

### The four questions Phase 0 answers

| Question | Why it gates design |
|---|---|
| **Is `sum(ntotal) == len(cluster_assignments)`?** | If not, the codebook was fitted against a *different* index set and every byte mapping is already wrong. This is a stop-the-line check. It is also exactly the failure the current code cannot detect (audit §7 step 10). |
| **How many clusters are empty (`K_empty`)?** | Decides whether M-13 needs a real fix or is a non-issue. Drives Decision 2. |
| **What is `min_density` across clusters?** | Decides whether `avoid_duplicates` can survive repeated bytes (M-14). A cluster with 3 members cannot carry a byte that appears 4 times. |
| **Was the codebook fitted with all three indices present?** | `fit_voronoi_codebook.py` skips missing indices silently, which shifts every global offset. Verifiable from the count check. |

> **Do not skip to Phase 1.** If check 5 fails, the entire `exact_vcp` channel is producing wrong bytes right now, and everything else on this list is cosmetic by comparison.

---

## Design decisions

Five real forks. **Decisions 1, 2, 4 and 5 are settled; Decision 3 branches on a Phase 0 measurement.** Reasoning kept below so the choices are auditable later.

| # | Decision | Status |
|---|---|---|
| 1 | `dcass doctor` as the first code written | **Settled — yes** |
| 2 | Content fingerprint binding codebook ↔ indices | **Settled — yes, with a one-time `--bless`** |
| 3 | Keyed bijection over usable clusters | **Settled — yes.** Whether base-`K'` conversion is also needed depends on Phase 0 |
| 4 | 5-byte framed payload + RS/CRC epoch search | **Settled — yes** |
| 5 | Cover-story carrier ranking | **Settled — yes, inside WP-6** |
| — | `config/default.yaml` (H-11) | **Settled — wire ~10 keys, delete ~40, one loader** |

### Decision 1 — Build a `dcass doctor` command as the *first* code we write

Instead of scattering path fixes and existence checks across modules, add one command that validates the entire runtime and prints a verdict table: dependencies importable · index counts · metadata/ntotal agreement · codebook present · **codebook↔index fingerprint match** · cluster population histogram · empty-cluster list · checkpoint presence · disk footprint.

**Why this first:** it is the instrument that produces Phase 0's numbers, it is what the teammate runs before touching the GPU, it replaces the broken `dcass status` (H-7), and it collapses B-3, H-7, M-13, M-14 and the version-stamp work into one reviewable deliverable. It is also the only artifact that makes a remote handoff verifiable — right now neither of you can prove the other's environment is sane.

**Recommendation: yes, and make it the definition of "the system is ready".**

### Decision 2 — Bind the codebook to the indices with a content fingerprint, not a version number

The failure mode that will cost the most time is rebuilding an index without refitting: every byte silently shifts, no error, wrong plaintext. A monotonic version integer doesn't catch it because nothing increments it.

**Proposal:** a sidecar `voronoi_codebook.meta.json` (or extra keys in the `.npz`) recording per modality `ntotal`, `d`, and a cheap content fingerprint — e.g. `sha256(reconstruct(0) ‖ reconstruct(ntotal-1) ‖ ntotal)` — plus `dim`, `num_clusters`, fit seed, and git commit. `VCPPayloadMapper.load()` refuses to run on mismatch instead of producing garbage.

Two vectors per index is enough to catch a rebuild and costs microseconds; hashing 314 MB every load does not.

**One-time wrinkle:** the codebook you have was fitted *before* this mechanism exists, so we can't recover its true fit-time fingerprint. Handle it as an explicit, logged **"blessing" step** — a `--bless` flag that computes the fingerprint from the currently-present indices and asserts *"I certify this pairing"*, gated behind the Phase 0 count check passing. Every codebook fitted afterwards gets the sidecar automatically. Being honest that the first pairing is asserted rather than proven is better than pretending otherwise.

**Recommendation: yes. This is the highest-value item in the whole plan.**

### Decision 3 — Solve the empty-cluster problem *and* dynamic context with one mechanism

Right now `byte b → cluster b`. That identity mapping is why an empty cluster is fatal (M-13), and it is why the channel has no key at all (H-12 — the project's namesake).

Replace it with a **keyed bijection over the usable cluster set**:

```
U        = clusters with >= min_carriers members   (measured in Phase 0)
π_key    = keyed permutation over U                (Fisher–Yates seeded from the epoch key)
encode   symbol s → cluster π_key(s) → pick a carrier from that cluster
decode   cluster c → symbol π_key⁻¹(c)
```

Three things fall out of this at once:

- **Empty clusters stop mattering** — they're simply never in the image of π.
- **The mapping becomes keyed and rotates per epoch**, which is the dynamic-context claim, actually implemented.
- **The codebook never has to be refitted or redistributed when the key rotates** — only the interpretation layer changes. That is a significant practical win over the refit-based alternatives.

Branch on `K' = |U|` from Phase 0:

| Phase 0 result | Approach |
|---|---|
| `K' >= 256` (likely — 153k vectors / 256 clusters ≈ 600 mean density) | Keyed permutation over any 256 of the usable clusters. **Nothing else needed.** M-13 solved by construction. |
| `K' < 256` | Add base-256 → base-`K'` conversion of the RS codeword before mapping. Order matters: **RS first** (it needs GF(2⁸) bytes), then base-convert for transport, then invert on receipt. Costs ~`log256/logK'` more carriers. |

**Recommendation: implement the keyed permutation regardless of `K'`; add base conversion only if Phase 0 forces it.** Measure before you build the harder version.

### Decision 4 — Frame the payload, and use RS as the epoch-detection oracle

The context mechanism has one serious operational risk: **if Alice and Bob derive different keys, decode fails completely** — no graceful degradation. Live external sources (crypto price, weather) are *not* reliably identical across two machines: API rounding, request timing across an hour boundary, rate limits, revisions.

Two mitigations, both cheap:

**(a) A 5-byte payload header inside the RS-protected region:**

```
[1 byte  version / flags ]
[2 bytes plaintext length ]
[2 bytes CRC-16 of plaintext ]
[N bytes plaintext UTF-8    ]
   → RS(GF(2^8), R parity) → codeword → keyed symbol map → clusters → carriers
```

Cost: 5 extra carriers. Gains: a version field for future format changes, an explicit length (no reliance on trailing bytes), and — critically — an integrity check independent of RS.

**(b) Epoch-window search on decode.** Try epochs `e, e−1, e+1, e−2, …` and accept the first where **RS succeeds *and* the CRC matches**. RS alone has a small but real false-accept rate; RS + CRC-16 makes wrong-epoch acceptance negligible. This turns a hard sync requirement into a soft one and is the difference between a demo that works and one that fails intermittently in front of a mentor.

**Also recommend:** make the *default* context source deterministic and offline — a coarse time bucket (`floor(unix / 3600)`) plus a pre-shared secret. Keep the network sources (`CryptoSource`, etc.) as opt-in, quantized coarsely (e.g. BTC rounded to the nearest \$100 from one pinned endpoint). A demo should not depend on CoinGecko being up.

**Recommendation: yes to both. The header is 5 bytes and buys the epoch search, which is what makes the feature usable.**

### Decision 5 — Stop ranking carriers by similarity to the *plaintext*

This is a research-level finding, not on either audit, and I think it is the most interesting thing here.

In `exact_vcp`, the query used to rank carriers within a cluster is `semantic_chunks[idx % len(semantic_chunks)]` — **derived from the secret message.** So the chosen carrier sequence is semantically correlated with the plaintext. An adversary who merely *suspects* the channel can cluster the transmitted carriers and recover the message's topic without breaking the code at all. The system leaks meaning through the very mechanism meant to hide it.

The underlying tension is real and worth naming in the paper:

> **Topical coherence** (a real person's feed is about something) **versus plaintext independence** (the carriers must reveal nothing).

The current code picks the worst corner: coherent *and* correlated with the secret. The fix is to keep coherence but anchor it to a **cover story independent of the plaintext** — a decoy topic derived from the key, or chosen explicitly by the user ("post like someone on a hiking trip"). Carriers are then ranked by similarity to the cover story, while the *cluster* still carries the payload byte. Coherence is preserved, correlation with the secret is gone.

This connects to whatever `docs/research/RESEARCH_LLM_NARRATIVE_COHESION_GUARD.md` is already exploring — worth reading that before finalising the design.

**Recommendation: adopt cover-story ranking. It is a small code change (swap the query string) with a large security and paper-narrative payoff.**

**DECIDED — lands inside WP-6, alongside the keyed mapping.** Both changes touch carrier selection, so they share one review and one round of tests, and the epoch key already exists at that point so the cover story can be derived from it at no extra cost.

---

## Work packages

Dependencies are real; the arrows matter.

```
WP-1 Unblock ──┬─→ WP-2 doctor ──┬─→ WP-3 codebook binding ──→ WP-6 keyed context
               │                 └─→ WP-4 path/config truth
               ├─→ WP-5 API + frontend correctness
               ├─→ WP-7 test tiers  (grows alongside everything)
               └─→ WP-8 training handoff  (independent — can run in parallel)
                                              WP-9 docs truth pass  (last)
```

### WP-1 · Unblock — hours

Install `reedsolo` (B-1). Add `fastapi`, `uvicorn[standard]`, `bert-score` to `requirements.txt`; add fastapi + uvicorn to the Dockerfile builder stage (B-2). Pin CLIP's install method properly rather than leaving it a comment. Then **run the test suite and record what passes** — there is no baseline today.

*Done when:* `import src.engine.encoder` succeeds and `pytest` produces a recorded pass/fail list.

### WP-2 · `dcass doctor` — 1 day

Per Decision 1. New CLI subcommand + a reusable `src/diagnostics/` module the API can also expose at `/api/doctor`. Retire the stale path logic in `cmd_status` (H-7) by pointing it at `resolve_indices_base_path()`. Sweep the other stale `data/…` paths (`tests/conftest.py:75`, `test_decoder.py:414,440`, `test_encoder.py:446,503`, `add_wikipedia_to_index.py`) and fix `SemanticBenchmark`'s dataset/results paths (H-8).

*Done when:* `dcass doctor` prints the Phase 0 table and exits non-zero on any hard failure.

### WP-3 · Codebook↔index binding — 1 day

Per Decision 2. Sidecar writer in `fit_voronoi_codebook.py`, fingerprint verification in `VCPPayloadMapper.load()`, `--bless` path for the existing codebook, surfaced in `doctor`.

*Done when:* deleting and rebuilding one index makes `load()` refuse rather than silently return wrong bytes — and there is a test that proves it.

### WP-4 · Path and config truth — 1 day

**M-16:** make `MediaItem._resolve_file_path` return `None` when the file genuinely isn't on disk, instead of returning `image_metadata.json` or an arbitrary `.arrow` shard. Frontend renders "carrier not on disk" for a null path. This is the fix that lets you *skip downloading raw media* on a 43 GB budget — it converts a storage problem into a correctness improvement.

**H-11 — DECIDED: wire the real keys, delete the aspirational rest.** `config/default.yaml` currently has zero consumers, two competing loaders, and contradicts the code in eight places.

*Keep and wire* (~10 keys): `paths.*`, `model.device`, enabled modalities, ECC parity bytes, default payload mode, `context.*` (needed by WP-6), `logging.level`.

*Delete* (~40 keys): `index.type` / `ivf.*`, `distribution.channels.github|imgur`, `encoding.error_correction.redundancy_factor`, `analysis.benchmark.message_sizes`, and the rest of the unimplemented surface.

*Correct before wiring:* `embeddings.text` says `all-MiniLM-L6-v2` / 384-d. The text index is CLIP ViT-B/32 / 512-d, and VCP hardcodes `dim=512`. Fix this first or the config will be wrong on day one.

*Collapse the loaders:* keep **one** `Config` class. `config/settings.py` is the better base — it has env-var overrides, project-root discovery, and `get_device()`. Delete `src/utils/config.py`.

Rationale: a config file documenting features the system doesn't have is a liability in review, but zero configuration surface is awkward to demo. Ten real keys is the right size.

### WP-5 · API and frontend correctness — 1 day

- **H-4** — add `_transmission_stop_requested` to the `global` statement. One line; the endpoint currently reports success while doing nothing.
- **H-5** — settle one URL convention. Recommend: env var holds the **origin** (`http://host:8000`), the client appends `/api`. Then fix `lib/api.ts`, `wire/page.tsx`, `.env.example`, and `docker-compose.yml` together. Today no single value satisfies both call sites.
- **H-6** — decide whether `delays[i]` means *before* or *after* item `i`, document it on `NoiseController`, and make `Scheduler.run` and `_transmit_packets_sync` agree. Recommend **"after"**, matching `NoiseController`'s existing gap-folding, which makes the API path already correct and localises the change to `Scheduler`.
- **H-9 / H-10** — implement the `auto` cascade or drop the option from `TransmitRequest`; anchor checkpoint paths to the project root and read model dims from the checkpoint rather than hardcoding 128/256.
- **P2-12** — mark `raw_codeword_hex` debug-only or remove it. It is the side channel the exact mode exists to eliminate; leaving it in the API invites the reviewer question "so is the payload in the media or not?"

### WP-6 · Keyed symbol mapping + payload framing — 2–3 days · the research contribution

Per Decisions 3, 4, 5. Roughly: a `ContextSource` interface (`epoch_id(t)`, `material(epoch)`) with `StaticKeySource` for tests and `TimeBucketSource` as default; HKDF key derivation; seeded Fisher–Yates permutation over the usable cluster set; the 5-byte header + CRC-16; epoch-window search on decode; cover-story query for carrier ranking.

**Build order inside this WP:** `StaticKeySource` first so the whole path is testable deterministically with no clock and no network. Add time and external sources only once round-trip tests pass.

*Done when:* encode at epoch `e`, decode at wall-clock `e+1`, and the epoch search recovers the message — with a test asserting a *wrong* key fails cleanly rather than returning plausible garbage.

### WP-7 · Test tiers — continuous

The structure that makes a remote handoff safe:

| Tier | Needs | Contents |
|---|---|---|
| 0 | nothing | imports + `doctor` self-check |
| 1 | nothing | chunker, ECC, base conversion, key derivation, permutation round-trip, header/CRC codec |
| 2 | indices + codebook | **real `VCPPayloadMapper`** — global-offset reconstruction, ID→symbol, fingerprint refusal. Closes P1-9. |
| 3 | nothing (CPU, seconds) | **gradient-flow tests** for R-22 / R-23. The artifact that lets the teammate trust the fix before spending GPU time. |
| 4 | indices + codebook | end-to-end encode→decode with 1–2 carriers corrupted **within** RS capacity, asserting recovery. Closes P1-8. |

**Also:** convert `test_voronoi_codebook.py`'s `assert CODEBOOK_PATH.exists()` to `pytest.skip`. Right now missing data reports as a failure, which makes a clean checkout look broken.

### WP-8 · Training handoff package — 2 days · can run in parallel

Code-only; no training here.

- **R-22** — `compute_gradient_penalty` must take `outputs=verdict.feature_importance["raw_critic_score"].sum()`, not `bot_probability.sum()`. Currently it constrains ‖∇σ(D)‖ instead of ‖∇D‖, and since σ′ ≤ 0.25 the Lipschitz condition is not enforced. This is live — two of the three training scripts set `use_gradient_penalty=True`.
- **R-23** — `sample_channels()` is `softmax → argmax`, so **the generator's channel head never receives gradient and has never been trained.** Replace with `F.gumbel_softmax(..., hard=True)` straight-through, keeping argmax for inference. Any claim about GAN-learned platform switching is unsupported until this lands.
- **R-24** — detach `fake_schedule.delays` in the warden loop; currently every one of the 5 critic steps backprops through the generator and discards it.
- **P1-5** — the RL state vector writes 16 slots while `_compute_state_dim()` returns 21, so 5 network inputs are permanently zero. Either populate the six unused history features (burstiness, peak-hour indicator, channel autocorrelation) or return 16. Recommend **returning 16** — cheaper, honest, and no invented features to justify in review.
- **P1-6** — pad `get_warden_score()` with a realistic mean delay or mask to true sequence length. Note the other audit's impact claim is wrong: `_compute_reward()` guards on window size and never pads, so this biases the *logged metric*, not the learning signal. Fix it, but don't file it as a training bug.
- **Consolidate** `scripts/{training,stealth}/train_*.py` into one canonical entry point with explicit flags.
- **Write `docs/TRAINING_HANDOFF.md`:** exact commands, expected wall-clock and VRAM, what "working" looks like (loss curves, warden score trajectory), where checkpoints land, and **"run tier-3 tests first — if they fail, don't start training."**

*Done when:* tier-3 tests pass locally on CPU and the handoff doc is something a teammate can execute without asking a question.

### WP-9 · Docs truth pass — 1 day · last

Reconcile against the audit's §5 table. Specifically: soften `06_SECURITY_AND_STEGANALYSIS_DEFENSE.md` — "perfectly secure" (line 97) and "Zero Detection" (lines 179–182) will not survive review, and the underlying result is strong enough to state as scoped rather than absolute. Correct the README stack table (argparse not Typer; hand-written PPO not Stable-Baselines3; CLIP not Sentence-Transformers for text). Strip the `file:///home/jeevan/...` paths from `SYSTEM_AUDIT_AND_HEALTH_REPORT.md`. Add the capacity/traffic trade-off (M-18) explicitly: bytes per carrier, carriers per message, observable events per message.

---

## Deliberately out of scope

- **No raw media downloads.** WP-4 makes missing files an honest `None`. Protects the 43 GB.
- **No corpus rebuild or codebook refit** unless Phase 0 check 5 fails. If it does, that becomes the top priority and this plan restarts at WP-3.
- **No GAN/RL training.** WP-8 ships code + tests + handoff doc only.
- **`semantic_legacy` is not deleted.** Recommend **demoting** it: rename to something honest like `semantic_similarity`, and reframe it in the paper as a *baseline ablation* — "here is what happens if you encode meaning instead of bytes" — rather than a transport mode. It has genuine research value as a comparison and none as a channel. That is a better outcome than either continuing to support it as a channel or throwing the code away.

---

## Sequencing summary

| Stage | Work | Gate to pass |
|---|---|---|
| 0 | Place files, measure | `sum(ntotal) == len(cluster_assignments)` |
| 1 | WP-1 | engine imports, test baseline recorded |
| 2 | WP-2, WP-3 | `doctor` green; rebuilt index is *refused* |
| 3 | WP-4, WP-5 | UI honest about missing files; stop button works; one URL convention |
| 4 | WP-6 | keyed round-trip across an epoch boundary; wrong key fails cleanly |
| 5 | WP-8 → teammate | tier-3 gradient tests pass; handoff doc executable |
| 6 | WP-9 | no claim in docs unsupported by code |

WP-8 has no dependency on WP-2–6 and can be done whenever it suits — worth starting early so the teammate isn't blocked behind the whole plan.

# DCASS Codebase Audit

**Date:** 2026-08-22 · **Commit:** `18b5739` · **Method:** full read of `src/`, `config/`, `scripts/`, `tests/`, `frontend/`, Docker + Make tooling. Docs treated as claims to verify, not as source of truth.

**Contents:** §1–3 what the system is and how every workflow runs · §4 21 findings by severity · §5 claim vs. code · §6 what's genuinely good · §7 build order · **§8 reconciliation with the team's `DCASS_FULL_AUDIT_REPORT.md`** (4 of its 14 items are already fixed; 3 new findings surfaced from the cross-check) · §9 open questions.

---

## 1. What the system actually is

DCASS encodes a secret message by **selecting existing, unmodified media items** from a pre-built corpus and transmitting only their **IDs** on a human-like schedule. The receiver looks each ID up in an identical corpus copy and recovers the payload. No carrier file is ever altered, so there is no pixel/LSB signal for classical steganalysis to find. The secret lives in *which* items were chosen and *when* they were sent.

There are **two completely different encoders** in the codebase, and this is the single most important thing to understand before building:

### Mode A — `exact_vcp` (the real, lossless channel; default in API + frontend)

| | |
|---|---|
| Unit of payload | **1 byte per media item** |
| Mechanism | 256 spherical k-means centroids partition the 512-d unit sphere. Every corpus vector belongs to exactly one cluster `0x00`–`0xFF`. To send byte `b`, pick any item whose cluster == `b`. |
| Decode | ID → cluster → byte → Reed–Solomon decode → **exact original string** |
| Requires | all three FAISS indices loaded **and** `voronoi_codebook.npz` |
| Code | `src/engine/vcp_payload.py`, `src/corpus/cluster/voronoi_codebook.py` |

### Mode B — `semantic_legacy` (the original, lossy channel)

| | |
|---|---|
| Unit of payload | one *chunk of meaning* per media item |
| Mechanism | chunk the message → CLIP-search the corpus per chunk → take the best-matching carrier |
| Decode | look up each ID, extract its caption/text, join with `" \| "` — an *approximation*, not the message |
| ECC | a Reed–Solomon codeword **is** computed, but it is not carried by the media at all — it is returned separately as `raw_codeword_hex` and passed back on decode. In this mode ECC is an out-of-band side channel. |
| Code | `src/engine/encoder.py::encode`, `src/engine/decoder.py::decode` |

Both modes share the chunker, the index, and the distribution/stealth layers. The benchmark suite and the CLI still exercise **Mode B only**; the API and the web UI default to **Mode A**. Most documentation describes Mode B while the shipped product runs Mode A.

---

## 2. Layer map

```
config/default.yaml ──── (dead: zero consumers, see F-13)

src/corpus/          corpus → vectors → FAISS
  loaders/           flickr_loader, wikipedia_loader, base_loader
  embedders/         image_embedder (CLIP ViT-B/32, 512d) ← also used for TEXT
                     text_embedder (MiniLM 384d) ← ORPHANED, nothing calls it
                     audio_embedder / clip_embedder / vector_engine
  index/unified_index.py   UnifiedSemanticIndex + ScoreNormalizer + MediaItem
  cluster/voronoi_codebook.py   spherical k-means, K=256, dim=512

src/engine/          the codec
  chunker.py         sentence → delimiter → length split, optional synonym expansion
  encoder.py         SemanticEncoder (both modes)
  decoder.py         SemanticDecoder (both modes)
  ecc.py             Reed–Solomon over GF(2^8) via `reedsolo`
  vcp_payload.py     byte ↔ media-ID bridge (Mode A)
  context/           EMPTY — the "Dynamic Context-Aware" in the project name

src/distribution/    getting IDs onto channels
  noise.py           NoiseController: jitter + idle gaps (+ optional drops)
  profiles.py        casual / steady / bursty / night_owl / debug
  dispatcher.py      round_robin | fixed | alternating
  scheduler.py       blocking time.sleep loop
  console_channel.py / local_folder_channel.py

src/stealth/         AI-driven timing
  gan/               TemporalPatternGenerator (GRU + causal gated conv), WGAN-GP trainer
  rl/                StealthEnvironment + PPOAgent (actor-critic, GAE, action masking)
  stealth_scheduler.py   static | gan | rl with fallback to static

src/analysis/
  adversarial/warden.py   DPI critic: BiLSTM + Transformer, unbounded (WGAN) score
  benchmarks/             CLIP similarity + BERTScore, report generation

src/api/server.py    FastAPI, 12 endpoints, lazy engine singletons, threaded transmitter
src/cli/main.py      argparse CLI: encode decode demo status search verify distribute benchmark
frontend/            Next.js 14 app router: / /encode /decode /wire /status /logs

DEAD WEIGHT
  src/embeddings_legacy/   step1..step8 one-off scripts, stale `data/` paths
  tools/                   download.py, test_step1.py
  FE_DCASS/dcass-demo/     superseded Flask demo + duplicated generator.py
  scripts/training/ vs scripts/stealth/   two copies of train_gan.py / train_rl.py
```

---

## 3. Workflows, end to end

### 3.1 Build the corpus (one-time, required before anything works)

```
download_flickr8k.py / download_flickr30k.py / download_wikipedia.py / audio_step1_download.py
   ↓
build_indices.py            → image.index + image_metadata.json
                            → text.index  + text_metadata.json  (CLIP text encoder, 512d)
build_flickr30k_index.py    → alternative image path (Kaggle/HF CSV formats)
audio_step2_build_index.py  → audio.index + audio_metadata.json (CLAP, 512d)
   ↓
fit_voronoi_codebook.py     → voronoi_codebook.npz
```

`fit_voronoi_codebook.py` stacks vectors in the order **image, text, audio** and `VCPPayloadMapper` reconstructs global row offsets in exactly that order. **These two must stay in lockstep** — if you rebuild one index or add a modality without refitting, every byte mapping silently shifts and the channel breaks. There is no version stamp or checksum tying the codebook to the indices. `scripts/data/download_hf_indices.py` exists to pull pre-built indices instead of rebuilding.

### 3.2 Encode → transmit → decode (the demo loop)

```
1  POST /api/encode  { message, payload_mode: exact_vcp, use_ecc: true }
2  RSErrorCorrection.encode(message)          → data bytes + 8 parity bytes
3  for each byte: VCPPayloadMapper.select_carrier(byte)
                  ├── candidates = all corpus items in cluster == byte
                  ├── filter by modality + already-used IDs
                  └── rank by cosine(query_embedding, item_vector)
4  → media_ids[]  (one ID per byte)  + resolved on-disk file paths
5  POST /api/transmit { media_ids, mode, base_delay, speed_multiplier }
6  StealthScheduler.schedule → delays[] + channels[]
7  background thread writes storage/shared_channel/<id>_<ch>_<seq>.json, sleeping between
8  GET /api/wire/packets       → the /wire page polls this every ~1s
9  POST /api/decode { media_ids }
10 ID → cluster → byte for every ID → RS decode → exact message
```

The Docker equivalent is `run_sender.py` (Alice) writing into a shared volume and `run_receiver.py` (Bob) watching it — `make docker-send`.

### 3.3 Train the stealth models (optional, currently unused)

```
generate_traffic_dataset.py / collect_real_traffic.py  → human traffic JSON
train_gan.py   Generator vs Warden, WGAN-GP, 5 critic steps per generator step
train_rl.py    PPO on StealthEnvironment; reward = throughput − 100·P(bot) + diversity bonus
→ storage/models/{gan_generator.pt, rl_agent.pt}
```

No checkpoints exist in the repo and both are `enabled: false`. **In its current state every "stealth" schedule the system produces is `NoiseController` jitter** — a seeded uniform jitter on a constant base delay plus occasional idle gaps. The GAN and RL code is complete and tested but has never been in the live path.

---

## 4. Findings

Severity: **B** blocker (nothing runs) · **H** high (wrong behaviour) · **M** medium · **L** low/hygiene.

### B-1 · The engine does not import: `reedsolo` is missing

`src/engine/ecc.py:16` does a top-level `import reedsolo`. `encoder.py` and `decoder.py` import from it at module level. The package is **not installed** in either Python environment on this machine (checked the system interpreter and `sem5/ML/vir`). Verified:

```
src/engine/encoder.py:24  from src.engine.ecc import RSErrorCorrection
src/engine/ecc.py:16      import reedsolo
ModuleNotFoundError: No module named 'reedsolo'
```

Consequence: the CLI, the API, the benchmark and **the entire test suite** fail at collection. `pip install reedsolo` is step zero. It *is* declared in `requirements.txt`, so this is an environment gap, not a missing declaration.

### B-2 · `requirements.txt` cannot stand up the API or the benchmark

Missing: **`fastapi`**, **`uvicorn`**, **`bert-score`**, and CLIP (CLIP is at least documented as a manual `pip install git+…`). `docker-compose.yml` runs `python -m uvicorn src.api.server:app` in the `dcass-api` service, and the Dockerfile installs torch, gymnasium, tensorboard and CLIP by hand — but never fastapi or uvicorn. **The `web` Docker profile cannot start.** `src/analysis/benchmarks/metrics.py` imports `bert_score` lazily inside the scoring functions, so the benchmark fails mid-run rather than at import.

### B-3 · `storage/` is empty

No indices, no `voronoi_codebook.npz`, no models. Every encode/decode path raises. Nothing is broken here — it is gitignored runtime state — but no one can run the system from a fresh clone without first doing §3.1 or `download_hf_indices.py`.

### H-4 · `POST /api/transmit/stop` never stops anything

`src/api/server.py:684-690` declares `global _transmission_active, _transmission_progress` but then assigns `_transmission_stop_requested = True`. Because that name is assigned in the function body without being declared global, Python makes it a **local**. The module-level flag the worker thread polls stays `False`. No exception is raised; the endpoint returns `{"success": true, "message": "Transmission stop requested"}` and the transmission runs to completion. Fix: add `_transmission_stop_requested` to the `global` statement.

### H-5 · The frontend's API base URL is inconsistent, and both env files set it wrong

Two different conventions coexist:

| File | Default base | Path appended |
|---|---|---|
| `frontend/src/lib/api.ts:9` | `http://localhost:8000/api` | `/encode`, `/decode`, `/status`, … |
| `frontend/src/app/wire/page.tsx:8` | `http://localhost:8000` | `/api/wire/packets`, `/api/transmit/status` |

Both read the same `NEXT_PUBLIC_API_URL`, so **no single value satisfies both.** And both places that set it omit `/api`:

- `.env.example`: `NEXT_PUBLIC_API_URL=http://localhost:8000`
- `docker-compose.yml`: `NEXT_PUBLIC_API_URL=http://dcass-api:8000`

With either value, every `lib/api.ts` call resolves to `http://host:8000/encode` and 404s — so encode, decode and status are broken in Docker while `/wire` works. It happens to work in local dev only because nobody sets the variable and the two hardcoded defaults differ. Fix: settle on the origin (no `/api`) as the env value and put `/api` in the client, in both files.

### H-6 · Timing semantics disagree between the two schedule consumers

`NoiseController.apply` (`src/distribution/noise.py:58-69`) folds an idle gap into `final_delays[-1]` — i.e. the delay list means *"wait this long **after** item i"*.

- `src/api/server.py:_transmit_packets_sync` writes the packet, **then** sleeps `delays[idx]`. Correct.
- `src/distribution/scheduler.py:23-26` sleeps `delays[idx]` **before** dispatching item `idx`. Off by one: the pause intended to follow item *i* is applied before it, and the final gap is never applied at all.

The CLI `distribute` path uses `Scheduler`; the API path does not. Both claim to reproduce the same behavioural profile.

### H-7 · `dcass status` always reports the indices as missing

`src/cli/main.py:451` reads `Path(__file__).parent.parent.parent / "data" / "indices"` — the pre-refactor layout. The real location is `storage/data/indices`, resolved everywhere else by `resolve_indices_base_path()`. The CLI has its own copy of the path logic and it is stale.

Same stale `data/…` prefix in: `tests/conftest.py:75`, `tests/test_engine/test_decoder.py:414,440`, `tests/test_engine/test_encoder.py:446,503`, `src/embeddings_legacy/step8_sentence_to_images.py:24`, `scripts/data/add_wikipedia_to_index.py`.

### H-8 · Benchmark writes where the API does not read

`SemanticBenchmark.__init__` defaults `dataset_path` to `<root>/data/benchmarks/test_messages.json` and saves results next to it in `data/benchmarks/results/`. `GET /api/benchmark/latest` reads `storage/data/benchmarks/results/`. The two never meet, and the dataset path is also stale, so the benchmark raises `FileNotFoundError` before it starts.

### H-9 · `mode: "auto"` silently degrades to static

`TransmitRequest.mode` accepts `static | rl | gan | auto`, but `StealthScheduler.schedule` types `mode` as `static | gan | rl` and its `if/elif/else` sends anything unrecognised — including `"auto"` — to `_schedule_static`. Meanwhile `scripts/runtime/run_sender.py` documents `auto` as an *RL → GAN → static cascade*. The cascade is not implemented anywhere.

### H-10 · GAN/RL checkpoints are looked up relative to the working directory

`StealthScheduler.schedule` builds `Path("storage/models/gan_generator.pt")` — CWD-relative, unlike the rest of the codebase which anchors to `Path(__file__).parent…`. Start uvicorn from any directory other than the project root and `mode=gan`/`mode=rl` fall back to static with only a `print` to say so. Relatedly, `_load_generator` hardcodes `latent_dim=128, hidden_dim=256, max_sequence_length=100` instead of reading them from the checkpoint, so any model trained with different dimensions fails `load_state_dict`.

### H-11 · `config/default.yaml` has zero consumers, and there are two competing Config classes

`config/settings.py` and `src/utils/config.py` are both singleton YAML loaders over the same file, with different APIs. Neither is imported anywhere in `src/`, `scripts/` or `tests/` — verified by grep. **Every parameter in the running system is hardcoded in Python.** The YAML is therefore not just unused but actively misleading:

| `default.yaml` says | The code actually does |
|---|---|
| `embeddings.text.model: all-MiniLM-L6-v2`, `dimension: 384` | text index is built with **CLIP ViT-B/32 at 512-d** (`build_indices.py` calls `ImageEmbedder` for text). `text_embedder.py` (MiniLM) is orphaned. VCP hardcodes `dim=512`. |
| `corpus.audio.enabled: false` | audio is in `default_modalities` and `exact_vcp` **requires** it |
| `encoding.min_similarity: 0.3` | encoder default is `min_score=0.0` |
| `encoding.error_correction.redundancy_factor: 1.5` | no such concept exists; ECC is RS parity bytes |
| `index.type` / `ivf.nlist` / `nprobe` | always `IndexFlatIP`, exact search |
| `distribution.channels.github`, `imgur` | only console + local folder exist |
| `context.*` (CoinGecko, weather, news) | **no implementation at all** — see H-12 |
| `stealth.gan.latent_dim: 100` | generator default is 128; scheduler hardcodes 128 |

### H-12 · "Dynamic Context-Aware" is unimplemented — and its absence is a security property, not just a missing feature

`src/engine/context/` and `src/engine/context/sources/` contain nothing but empty `__init__.py` files with `__all__ = []`. No module anywhere derives a context key, reads a time bucket, or fetches an external signal. Grep for `context_key`, `dynamic_context`, `coingecko`: zero hits outside the YAML.

This matters beyond naming. In `exact_vcp` mode the mapping from media ID to payload byte is **fully determined by the corpus and the codebook**. There is no key, no salt, no time dependence. Anyone holding the same public corpus and codebook decodes the traffic — and the codebook is derivable from the corpus, which is public data. The system currently provides **carrier unobtrusiveness, not confidentiality**. The research claims about dynamic context keys defeating static mappings describe a design intent, not this code.

### M-13 · `exact_vcp` fails hard on any byte with an empty cluster

`VCPPayloadMapper.load()` warns about symbols with zero carriers, then `select_carrier` raises `RuntimeError` when the encoder needs one. ASCII text only touches `0x20`–`0x7e`, but **Reed–Solomon parity bytes are uniform over 0x00–0xFF** — and `use_ecc: true` is the default in both the API and the frontend. So a message that would encode fine without ECC can fail unpredictably with it, depending on which parity bytes come out. There is no retry, no cluster-splitting fallback, and no pre-flight check that all 256 clusters are populated. Whether this bites depends entirely on the fitted codebook's cluster balance, which `fit()` reports but nothing enforces.

### M-14 · `avoid_duplicates` drains clusters on repeated bytes

`exact_vcp` passes `avoid_duplicates=True`, so each byte consumes a distinct carrier from its cluster. A message with many repeats of one byte (spaces, `e`, a long run) needs that many distinct items in a single cluster. With `min_density` low for some clusters this raises mid-encode. The failure message names the byte but not the cause.

### M-15 · Channel count mismatch between transmit and warden

`TransmitRequest.num_channels` defaults to 3 and is user-settable. `DeepPacketInspectionWarden` and the RL environment are constructed with `num_channels=req.num_channels`, but a *trained* RL checkpoint has a fixed `channel_logits` output width and a fixed `channel_embedding` size. Requesting `num_channels=5` against a 3-channel checkpoint fails to load; requesting 2 loads but wastes a head. Not currently reachable (no checkpoints) but it is a real coupling.

### M-16 · Misleading file-path resolution for images and audio

`MediaItem._resolve_file_path` (`unified_index.py:136-295`) walks a long candidate chain and, when it finds nothing, returns *the metadata JSON file itself* as the image's path (line 203-211) or `dataset_info.txt`/an arbitrary `.arrow` shard for audio (line 236-248). The API hands that string to the frontend as `file_path`, so the UI will display a JSON file as if it were the located carrier. Returning `None` and letting the UI show "not on disk" would be honest; the current behaviour makes a missing corpus look like a present one.

### M-17 · Cross-modal ranking rests on hardcoded calibration constants

`ScoreNormalizer.DEFAULT_CALIBRATION` holds per-modality `(mean, std)` pairs from a single one-off measurement ("20 queries × 5 results", dated 2026-02-17), fed through a z-score and a sigmoid. In `best` mode the modality mix of any encoding is a direct function of those six numbers, not of semantics — image scores are recentred from ~0.27 and text from ~0.885 onto a shared 0-1 scale. `update_calibration()` exists but is never called. Any corpus change invalidates the constants silently. This is the weakest link in the Mode B story and should be measured, not assumed, before it goes in a paper.

Related: `_candidate_text_score` in `encoder.py` adds a lexical-overlap bonus computed on `media.content`, plus a flat `+0.04` for `modality == "text"`. Audio metadata frequently has no transcript, so audio carriers get overlap 0 and no bonus — audio is structurally disadvantaged in `best` mode on top of the calibration effect.

### M-18 · Capacity and traffic cost of the exact channel

One byte per media item. `"Meet me at the cafe at noon"` = 27 bytes + 8 parity = **35 media items**, each a separate "post" the warden observes, with `base_delay` between them. This is the honest cost of losslessness and it should be stated explicitly in the write-up alongside the stealth claims, because the two pull against each other: the mode that guarantees recovery is also the mode that generates the most traffic.

### L-19 · Duplicated and dead code

- `scripts/training/train_gan.py` + `train_rl.py` vs `scripts/stealth/train_gan.py` + `train_rl.py` + `train_gan_extended.py` — four training entry points, unclear which is canonical
- `scripts/demo/` and `scripts/demos/` (two directories, similar names)
- `src/embeddings_legacy/` — eight superseded step scripts with stale paths
- `tools/` — `download.py`, `test_step1.py`
- `FE_DCASS/dcass-demo/` — superseded Flask UI carrying its own copy of `generator.py`
- `src/corpus/embedders/text_embedder.py`, `clip_embedder.py`, `vector_engine.py` — unreferenced by the build scripts
- `src/corpus/preprocessors/chunker.py` — a second chunker, not the one the engine uses

### L-20 · No project tooling

No `pyproject.toml`, `pytest.ini`, `setup.cfg`, or CI. `make lint` calls `ruff` and `make format` calls `black` with no configuration for either. Test discovery works only because `conftest.py` manually inserts the project root on `sys.path`.

### L-21 · API has no auth and one destructive endpoint

`DELETE /api/wire/packets` unlinks every non-underscore `.json` in `storage/shared_channel/`. Fine for a local demo; worth a note before this is exposed anywhere. CORS is correctly restricted to the frontend origins via `DCASS_CORS_ORIGINS` (a real improvement over wildcard-plus-credentials).

---

## 5. Claim vs. code

| Documented / README claim | Reality |
|---|---|
| "Dynamic context keys prevent static mappings" | Not implemented. `src/engine/context/` is empty. Mapping is static and unkeyed. |
| "GAN-based human behavior scheduler" | Code complete, no checkpoint, `enabled: false`, never in the live path. |
| "RL agent for adaptive stealth" | Same. |
| "0.0% Bit Error Rate guarantee" (audit doc) | True only for `exact_vcp` **within RS correction capacity** (`t = ⌊parity/2⌋ = 4` byte errors at the default 8 parity bytes) **and** only if every needed cluster is non-empty. Mode B cannot recover the message at all. |
| "153,281 FAISS vectors across 3 modalities" | Plausible for the authors' machine; `storage/` is empty here and nothing pins the count. |
| "Stable-Baselines3 / RLlib" (README stack table) | PPO is hand-written in `src/stealth/rl/agent.py`. Neither library is used or in `requirements.txt`. |
| "Typer / Click" for CLI | `argparse`. Typer is in `requirements.txt` but unused. |
| "Sentence-Transformers for text embeddings" | CLIP text encoder. Sentence-Transformers is installed but only used by the orphaned `text_embedder.py`. |
| "~85% complete / production-ready" (`PROJECT_COMPLETION_STATUS.md`) | The Mode A codec is genuinely solid. The named headline feature is absent, the config layer is dead, and the API/Docker web profile does not start from a clean install. |
| "Overall Status: OPERATIONAL" (`SYSTEM_AUDIT_AND_HEALTH_REPORT.md`) | That report contains `file:///home/jeevan/...` absolute paths from another machine and asserts benchmark numbers no artifact in the repo supports. |

---

## 6. What is genuinely good

Worth protecting while fixing the above:

- **`vcp_payload.py`** is the strongest module in the repo. The byte↔ID bridge is clean, the global-offset reconstruction is explicit about its coupling to the fitting script, it warns about empty clusters, and `_vector_score` deliberately returns a neutral score on failure so an error can never promote a candidate. That last comment shows real care.
- **Modality-correct query encoding.** `_encode_query` routes audio through CLAP and image/text through CLIP, with a per-encoder embedding cache and a loud warning when `transformers` is absent. A lot of multi-modal code gets this wrong.
- **The PPO implementation is correct where it usually isn't.** `act()` clamps the delay *before* computing `log_prob` so the stored `old_log_probs` matches the executed action, action masks are replayed into `evaluate()` during the update, and GAE-λ is properly handled. The inline comment explaining the clamp ordering is exactly right.
- **The WGAN-GP setup is coherent.** The warden's classification head deliberately has no sigmoid, `compute_warden_loss` is a true Wasserstein critic loss, `bot_probability` is a separate sigmoid view for the RL reward, and the gradient penalty disables cuDNN and flash attention for the double-backward. Someone debugged this properly.
- **`NoiseController` gates packet drops behind `drop_packets=False`** with a comment explaining that dropping an item in `exact_vcp` destroys a payload byte. Correct reasoning about the interaction between two layers.
- The transmission worker sleeps in 0.25 s slices so a stop request is honoured promptly — the mechanism is right, only the flag is broken (H-4).
- `/api/status` caches FAISS counts on `(path, size, mtime)` instead of re-reading indices per call.

---

## 7. Suggested build order

**Get it running (hours)**

1. `pip install reedsolo` — unblocks everything (B-1).
2. Add `fastapi`, `uvicorn[standard]`, `bert-score` to `requirements.txt`; add fastapi+uvicorn to the Dockerfile builder stage (B-2).
3. Populate `storage/` via `download_hf_indices.py` or §3.1, then `fit_voronoi_codebook.py` (B-3).
4. Run the test suite and record what actually passes. Right now there is no baseline.

**Fix what silently lies (a day)**

5. H-4 (`global` statement) — one line.
6. H-5 — pick one URL convention, fix `api.ts`, `wire/page.tsx`, `.env.example`, `docker-compose.yml`.
7. H-7 — replace the CLI's private path logic with `resolve_indices_base_path()`; sweep the remaining stale `data/…` paths.
8. H-6 — decide whether `delays[i]` means before or after item *i*, document it on `NoiseController`, and make both consumers agree.
9. H-8 — point `SemanticBenchmark` at `storage/data/benchmarks/`.

**Close the integrity gaps (a few days)**

10. Stamp the codebook with the index shapes it was fitted on, and have `VCPPayloadMapper.load()` refuse to run on a mismatch. This is the failure mode that would waste the most debugging time later.
11. Add a pre-flight `verify_codebook()`: report per-cluster population, and refuse `exact_vcp` + ECC if any of the 256 clusters is empty (M-13). Decide the fallback policy — nearest non-empty cluster, or re-fit with balanced k-means.
12. H-11 — either wire `config/default.yaml` into the code or delete it and both Config classes. Do not leave a config file that documents behaviour the system does not have. If wiring it up, correct the text-embedding entry to CLIP/512 first.
13. H-9/H-10 — implement the `auto` cascade or remove the option; anchor checkpoint paths to the project root and read model dims from the checkpoint.

**Then the research work**

14. **Dynamic context (H-12)** is the largest genuine gap and the one the project is named after. Minimum viable version: derive a key from a shared context source (time bucket + public data), use it to permute the cluster→byte mapping per epoch, and have both sides derive it independently. This turns the system from obfuscation into keyed steganography and makes the central claim true.
15. Re-measure `ScoreNormalizer` calibration on the actual corpus and wire up `update_calibration()` (M-17), or drop cross-modal score comparison from the claims.
16. Before training the GAN at all, fix **R-22** (gradient penalty on the sigmoid) and **R-23** (channel head gets no gradient). Training as-is would produce a model whose channel policy is random and whose critic is not Lipschitz-constrained — and any published WGAN-GP claim would be wrong. Then train both schedulers and get them into the live path, so "AI-driven stealth" describes a running system.
17. Add the two missing tests the other audit correctly identifies: within-capacity RS recovery *through the VCP carrier path* (P1-8), and an integration test against the real `VoronoiCodebook` covering global-offset reconstruction (P1-9). The second one is the natural home for the version-stamp check from step 10.
18. Fix the RL state vector (P1-5): either populate the six unused history features or correct `_compute_state_dim()` to 16. Any RL result reported against a state vector with 5 dead inputs invites the question in review.
19. Document the capacity/stealth trade-off (M-18) honestly: bytes per carrier, carriers per message, observable events per message, and warden detection rate as a function of all three.
20. Soften the security docs (P2-11): "perfectly secure" and "Zero Detection" will not survive review. The underlying result — that content-residual steganalysis cannot beat random guessing on unmodified carriers — is strong and true; state it as scoped rather than absolute.

---

## 8. Reconciliation with `DCASS_FULL_AUDIT_REPORT.md`

A second audit (dated the same day, cross-referencing Codex session `01a0202d`) lists 4 P0 / 6 P1 / 4 P2 issues, nearly all marked **OPEN**. Re-checked every item against `18b5739`. **Most of its P0s have already been fixed** — that report appears to have been written against a pre-fix tree, or read the deprecated functions still present in the files.

| # | Their claim | Verified status at `18b5739` |
|---|---|---|
| P0-1 | WGAN-GP is really Sigmoid + BCE | **Mostly fixed.** `classification_head` ends at `nn.Linear(64,1)` with an explicit *"No Sigmoid"* comment. `compute_warden_loss` is now `fake.mean() - real.mean()`. The BCE version survives as `compute_warden_loss_bce`, marked deprecated — that is the code their line numbers 387-398 point at. **But see R-22: the gradient penalty was not fixed.** |
| P0-2 | CLIP text encoder used for audio search | **Fixed.** `_encode_text_for_audio` (CLAP), `_encode_query` dispatches by modality, `search()` caches per encoder, warns when `transformers` is absent. |
| P0-3 | PPO delay log_prob mismatch | **Fixed**, exactly as they recommended, with a comment explaining the clamp-before-log_prob ordering. |
| P0-4 | VCP reranking uses CLIP on CLAP vectors | **Fixed.** `_vector_score` calls `_encode_query(query, modality)`; the embedding cache is keyed `f"{modality}:{query}"`. |
| P1-5 | 5 of 21 RL state inputs are dead zeros | **Open, confirmed.** 16 slots written, `_compute_state_dim()` returns 21, 5 zeros padded. |
| P1-6 | Warden zero-padding biases early rewards | **Open but the impact claim is wrong.** The padding is only in `get_warden_score()`. `_compute_reward()` guards on `len(history) >= warden_window_size` and never pads, so **rewards are not affected**. It biases the logged `warden_scores` metric, not the learning signal. |
| P1-7 | Generator/Warden objective mismatch | **Fixed.** `trainer.py:308` passes `fake_verdict.feature_importance["raw_critic_score"]` into the generator loss, with a comment saying why the sigmoid view must not be used. |
| P1-8 | No RS within-capacity recovery test | **Open, but overstated.** `test_ecc.py::test_rs_ecc_basic_encoding_decoding` *does* test within-capacity recovery (3 errors, parity 8, t=4) at the ECC layer. What is missing is that test *through the VCP carrier path* — no test corrupts 1–2 carrier IDs and asserts recovery. |
| P1-9 | All VCP tests use `FakePayloadMapper` | **Open, confirmed, and worse than stated.** `test_voronoi_codebook.py` touches the real `.npz` but only checks centroid norms and self-assignment. Nothing exercises `VCPPayloadMapper`'s global-offset reconstruction — the single most fragile piece of the system (see step 10 in §7). Both codebook tests also `assert CODEBOOK_PATH.exists()` rather than skipping, so with an empty `storage/` they fail rather than report "not applicable". |
| P1-10 | Static hardcoded calibration | **Open, confirmed** — same as M-17. Their addendum about audio scores being CLIP-to-CLAP cross-space products is now obsolete (P0-2 is fixed), but the calibration constants predate that fix and still need re-measuring. |
| P2-11 | Absolute language in security docs | **Open, confirmed.** `docs/modules/06_…:97` "perfectly secure"; lines 179-182 "Zero Detection". |
| P2-12 | `raw_codeword_hex` still returned | **Open, confirmed.** Returned by `/api/encode`, accepted by `/api/decode`. |
| P2-13 | First-transmission `delay_from_previous` dead branch | **Open, confirmed.** `environment.py:291-294` — both branches of the conditional are `delay`. |
| P2-14 | Codex doc patches failed | Historical; consistent with P2-11 still being open. |

**Net: 4 of their 14 items are already fixed** (P0-1 in part, P0-2, P0-3, P0-4, P1-7), one has an incorrect impact analysis (P1-6), one is overstated (P1-8), and the rest hold. That report also does not cover anything in §4 above — it audits model correctness only, and misses that **nothing currently runs at all** (B-1).

Cross-checking it did surface three findings neither audit had:

### R-22 · The gradient penalty is still computed on the sigmoid-squashed output

The other report's central WGAN-GP criticism was addressed everywhere **except the one place it specifically warned about.** `compute_gradient_penalty` (`warden.py:474-478`) still does:

```python
gradients = torch.autograd.grad(
    outputs=verdict.bot_probability.sum(),   # sigmoid(raw_critic_score)
    inputs=interpolated_delays, ...
)
```

So the penalty constrains `‖∇ sigmoid(D)‖ → 1`, not `‖∇D‖ → 1`. Since `sigmoid'(x) ≤ 0.25`, the constraint is nearly unsatisfiable in the intended direction and the Lipschitz condition on the critic is not enforced. Their own reasoning applies verbatim; the fix simply missed this line. It should read `outputs=verdict.feature_importance["raw_critic_score"].sum()`.

This is live: `scripts/stealth/train_gan.py:51` and `train_gan_extended.py:53` both set `use_gradient_penalty=True`. (The `TrainingConfig` default is `False`, so `scripts/training/train_gan.py` gates it behind a `--wgan-gp` flag — three training scripts, three different GP settings.)

### R-23 · The generator's channel head receives no gradient and is therefore never trained

`TimingSchedule.sample_channels()` does `torch.softmax(...)` then **`torch.argmax(...)`** — non-differentiable. The trainer feeds that Long tensor into the warden, where it hits `nn.Embedding`. No gradient path returns to `channel_head`. Nothing else in the codebase puts `channel_logits` in a loss.

Consequence: the generator's **delay** head learns from the adversarial signal; its **channel** head stays at its Xavier initialisation forever. Any claim about GAN-learned "multi-channel platform switching" is unsupported by the training loop. The docstring on `sample_channels` says "using Gumbel-Softmax or argmax" — only argmax is implemented. A real `F.gumbel_softmax(..., hard=True)` with straight-through gradients would close this.

### R-24 · Wasted backward pass in the warden loop

In the warden update, `fake_delays = fake_schedule.delays` is passed to the warden **undetached**, so `warden_loss.backward()` propagates all the way through the generator every one of the 5 critic steps. The gradients are discarded (only `warden_optimizer.step()` runs, and the generator's are zeroed before its own step), so this is wasted compute rather than a correctness bug — but it is ~5× more generator backward passes than the algorithm needs. `fake_schedule.delays.detach()` fixes it. Note the GP call already detaches correctly.

---

## 9. Open questions for the team

- Is `semantic_legacy` still a supported mode, or a historical artifact? The benchmark and CLI only test it; the product only ships Mode A. If it stays, it needs its own accuracy story; if it goes, a lot of code and documentation can be deleted.
- Which of the four training entry points is canonical?
- Is `voronoi_codebook.npz` intended to be a shared artifact between Alice and Bob, distributed out of band? If so it is effectively the key, and its distribution deserves the same treatment as a key.
- Was `config/default.yaml` ever wired up, or aspirational from the start? That determines whether H-11 is a regression to fix or a file to delete.

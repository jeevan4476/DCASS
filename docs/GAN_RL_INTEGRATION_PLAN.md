# Plan — Integrate GAN + RL into one working stealth-scheduling unit

## Context

DCASS already has three major pieces that are meant to form one adaptive covert-channel scheduler:

- a **GAN** (`TemporalPatternGenerator` + `DeepPacketInspectionWarden`, WGAN-GP) that learns what *human* traffic looks like,
- an **RL agent** (`PPOAgent` over `StealthEnvironment`) that learns a scheduling policy, and
- a **bridge** (`StealthScheduler` in [src/stealth/stealth_scheduler.py](../src/stealth/stealth_scheduler.py)) that already selects between `static`, `gan`, `rl`, and `auto` modes at inference.

The problem: as shipped, the two learners are not actually connected. The RL training path in [scripts/runtime/run_sender.py](../scripts/runtime/run_sender.py) builds a **random-weight** Warden, so its reward signal is noise. The training scripts that `docker-compose.yml` points to (`scripts/stealth/train_gan.py`, `scripts/stealth/train_rl.py`) **do not exist**. Checkpoint paths disagree between trainer output (`checkpoints/gan/final.pt`), RL output (`/app/checkpoints/rl/ppo_agent_final.pt`), and the scheduler's expected `storage/models/{gan_generator,rl_agent}.pt`. Several smaller correctness bugs (GP defaults off, first-sample seq-length assumption, delay-semantics mismatch, no action mask at inference, rate-limit violations still appended to the output schedule) prevent the integrated system from running end-to-end.

**Goal:** a reproducible build in which the GAN first trains the Warden on real/synthetic human traffic, that Warden is frozen and reused as the RL reward model, the PPO policy learns against it, and the scheduler loads both artifacts from one canonical location. The methodology section of the paper will describe exactly this pipeline, so each phase of this plan also doubles as a methodology step.

Sequential coupling chosen: **GAN → freeze Warden → RL**. Co-adaptive loops and GAN-as-prior are deliberately deferred.

The standalone subsystem under [scripts/stealth/](../scripts/stealth/) (its own WGAN-GP over timestamps + Double-DQN path selector) is out of scope and stays as-is — it is a parallel design, not part of the integrated pipeline.

---

## Phase A — GAN pretraining pipeline

**Add [scripts/stealth/train_gan.py](../scripts/stealth/train_gan.py)** — the file `docker-compose.yml` already calls. Thin CLI around existing `GANTrainer` / `train_gan` in [src/stealth/gan/trainer.py](../src/stealth/gan/trainer.py).

- Flags: `--data`, `--epochs`, `--batch-size`, `--latent-dim`, `--hidden-dim`, `--num-channels`, `--max-seq-len`, `--device`, `--checkpoint-dir`, `--out`, `--val-fraction`.
- Default `--out` = `storage/models/gan_generator.pt` so the scheduler finds it without manual copying. Keep per-epoch snapshots in `--checkpoint-dir`.
- Reuse `HumanTrafficDataset` ([src/stealth/gan/trainer.py:80-163](../src/stealth/gan/trainer.py#L80-L163)) and `train_gan()` ([src/stealth/gan/trainer.py:446-478](../src/stealth/gan/trainer.py#L446-L478)).
- `--val-fraction` (default 0.1): split `HumanTrafficDataset` into train/val for Phase D's AUC metric. The dataset currently has no split — add a `.split(fraction)` helper or use `torch.utils.data.random_split` with a fixed seed in the training script.

**Fix `TrainingConfig` and `GANTrainer.train_step`** in [src/stealth/gan/trainer.py](../src/stealth/gan/trainer.py):

- Flip `use_gradient_penalty` default to `True` ([trainer.py:59](../src/stealth/gan/trainer.py#L59)) — a WGAN without GP is unstable, and `compute_gradient_penalty` already exists in [src/analysis/adversarial/warden.py](../src/analysis/adversarial/warden.py).
- Replace the first-sample sequence-length assumption at [trainer.py:246](../src/stealth/gan/trainer.py#L246) with a per-sample mask, so mixed-length batches train correctly. Build a `(B, T)` validity mask from `seq_lengths` and pass it through the Warden (add a mask arg to `DeepPacketInspectionWarden.forward` that zeros-out padded steps in the pooling).
- In the Warden-phase forward at [trainer.py:264-265](../src/stealth/gan/trainer.py#L264-L265) switch the fake channels to `channel_probs_straight_through()` for GP continuity (currently uses hard `sample_channels()` which breaks channel-head gradients in GP interpolation).

**Extend `compute_gradient_penalty` in [src/analysis/adversarial/warden.py:467-529](../src/analysis/adversarial/warden.py#L467-L529)** to interpolate **both** delays and channel soft-probabilities. This is bigger than a one-line change — do it carefully:

1. Convert `real_channels` (long `(B, T)`) → one-hot `(B, T, C)` via `F.one_hot(real_channels, C).float()`.
2. Build `interp_channel_probs = α * real_onehot + (1 - α) * fake_softprob`, where `fake_softprob` is the straight-through probability tensor from the generator's channel head.
3. Mark `requires_grad_(True)` on *both* `interpolated_delays` and `interp_channel_probs`.
4. Call `warden(interp_delays, interp_channel_probs)` — the Warden's existing soft-prob code path (matmul with embedding weights) handles this.
5. `torch.autograd.grad(outputs=raw_critic.sum(), inputs=[interp_delays, interp_channel_probs], create_graph=True, retain_graph=True)` returns two gradient tensors; flatten each to `(B, -1)`, concatenate along dim=1, then compute the single L2 norm per sample. GP = `λ_gp * ((||g||_2 - 1)^2).mean()`.
6. The `cudnn.flags(enabled=False)` + `cuda.sdp_kernel(math=True)` guard at [warden.py:505-510](../src/analysis/adversarial/warden.py#L505-L510) must still wrap the forward+backward for double-backward compatibility.

**Transformer positional encoding caveat.** The Warden is trained at `max_sequence_length=100` (padded) but the env evaluates it on 20-step windows ([environment.py:346-368](../src/stealth/rl/environment.py#L346-L368)). If the Warden uses learned absolute positional embeddings, generalization to window 20 will be poor and detection at inference won't match training. Fix one of:

- (a) use sinusoidal positional encodings in the Warden Transformer stack,
- (b) train the Warden on mixed window lengths (sample a random `L ∈ [20, 100]` per batch),
- (c) set `warden_window_size = max_sequence_length` and pad with the mean-delay trick already in `get_warden_score` ([environment.py:402-435](../src/stealth/rl/environment.py#L402-L435)).

Pick one before training. (a) is the cleanest and most paper-defensible.

**Dataset provenance.** Document in a new `data/README.md` where `human_traffic.json` is expected to come from (user-supplied capture, or `HumanTrafficDataset._generate_synthetic_data` as a documented fallback). The synthetic path already works ([trainer.py:115-138](../src/stealth/gan/trainer.py#L115-L138)); the plan does not require real captures to run, only to produce publishable results.

**Validation metric.** Add `scripts/stealth/eval_warden.py` that loads the final checkpoint and reports Warden AUC on the held-out `--val-fraction` split (real vs. generator samples). Target behavior at convergence: AUC should drop toward 0.5 (indistinguishable). This matches the research-paper claim.

**Checkpoint schema** (keep what the scheduler already reads at [stealth_scheduler.py:148-168](../src/stealth/stealth_scheduler.py#L148-L168)):

```
{"generator_state", "warden_state", "generator_optimizer", "warden_optimizer",
 "config": TrainingConfig, "epoch", "global_step", "metrics_history"}
```

---

## Phase B — RL training pipeline against the frozen Warden

**Add [scripts/stealth/train_rl.py](../scripts/stealth/train_rl.py)** — the file `docker-compose.yml` already calls, with a working `--warden-checkpoint` flag that the current in-sender training path lacks.

- Flags: `--warden-checkpoint` (REQUIRED; path to the GAN checkpoint from Phase A), `--episodes`, `--num-channels`, `--lambda-stealth`, `--min-seq-len`, `--max-seq-len`, `--device`, `--out`.
- Default `--out` = `storage/models/rl_agent.pt`.
- Behavior:
  1. Load the GAN checkpoint; instantiate `DeepPacketInspectionWarden(num_channels)`; call `warden.load_state_dict(ckpt["warden_state"])`; `warden.eval()`; freeze (`for p in warden.parameters(): p.requires_grad_(False)`).
  2. Build `StealthEnvironment(num_channels, warden=warden, lambda_stealth=...)` — the env already accepts an injected warden ([environment.py:88-110](../src/stealth/rl/environment.py#L88-L110)), it just was never given a trained one.
  3. Build `PPOAgent` with `PPOConfig(state_dim=env.state_dim, ...)` — explicitly pass `env.state_dim`, not the `PPOConfig.state_dim` default of 21 which disagrees with the env's 16 ([agent.py:28](../src/stealth/rl/agent.py#L28), [environment.py:143](../src/stealth/rl/environment.py#L143)).
  4. Call `agent.train(num_episodes, media_sequence_generator=...)` with a custom generator that respects `--min-seq-len`/`--max-seq-len` (see short-sequence fix below).
  5. Save via `agent.save(out)` — this already works and the file goes directly to the scheduler's expected path.

**λ-curriculum.** Start `lambda_stealth=10` and ramp to the target (default `50`) over the first ~20% of episodes. Implemented as a `lambda_schedule` arg on `train_rl.py` that mutates `env.lambda_stealth` between episodes.

> ⚠️ **Caveat:** `compute_gae` normalizes advantages ([agent.py:400](../src/stealth/rl/agent.py#L400)), so changing `λ_stealth` shifts the raw advantage distribution but normalization re-centers it. The curriculum still changes the gradient *direction* via per-step reward structure, but its effect on gradient magnitude is partially absorbed. Consider a stepped schedule (λ=0 for first N episodes, then jump to target) as an alternative — may be more effective than a linear ramp. Try both and report whichever stabilizes training.

**Delete or redirect** the random-Warden training in `scripts/runtime/run_sender.py:train_agent_in_container` (lines ~257-286). Either remove `--mode train` from the sender, or make it `subprocess.run([...train_rl.py, "--warden-checkpoint", DEFAULT_GAN_CHECKPOINT, ...])` so there is exactly one RL training path.

**Fix delay semantics** in [src/stealth/rl/environment.py:267-284](../src/stealth/rl/environment.py#L267-L284). Adopt the "delay AFTER send" convention used by the API transmitter and `NoiseController`. The fix is NOT just reordering lines — the rate-limit check needs to probe the *post-delay* time:

```python
# Current (buggy for the convention we want):
self.current_time += delay      # advance BEFORE check
if not channel.can_send(self.current_time): ...   # too late — item never sent
# Replace with:
projected_time = self.current_time + delay
if not self.channels[channel_id].can_send(projected_time):
    return self._get_state(), -10.0, False, {"reason": "rate_limit_violation"}
media_id = self.media_queue.popleft()
record = TransmissionRecord(media_id, channel_id,
                            timestamp=self.current_time,  # send-time
                            delay_from_previous=delay)
self.transmission_history.append(record)
channel.last_transmission_time = self.current_time
self.current_time += delay      # advance AFTER send
```

Document the convention in a `# Delay convention:` comment block at the top of `environment.py` and in [src/stealth/stealth_scheduler.py](../src/stealth/stealth_scheduler.py).

**Fix rate-limit handling for inference.** The training loop already uses the mask correctly ([agent.py:342-343](../src/stealth/rl/agent.py#L342-L343)) — no change needed there. The only fix is at inference (Phase C). Rate-limit violations during training training stay as negative-reward signal to the agent; that behavior is correct.

**Fix short-sequence reward sparsity.** `_compute_reward` only applies the Warden penalty once `len(history) >= warden_window_size=20` ([environment.py:346](../src/stealth/rl/environment.py#L346)). The default training sequence length is 10-29 ([agent.py:506-508](../src/stealth/rl/agent.py#L506-L508)), so for ~⅔ of episodes the agent never sees a stealth signal. Pick one:

- Lower `warden_window_size` to 10 (quickest).
- Change the default `media_sequence_generator` in `train_rl.py` to produce sequences of at least 30 items.
- Mirror `get_warden_score`'s mean-delay padding in `_compute_reward` so the penalty applies from step 1 onward (cleanest, matches inference behavior).

Recommend the padding fix — it also removes a train/eval behavior gap.

---

## Phase C — Inference / integration glue

**Pass the action mask at inference** in [src/stealth/stealth_scheduler.py:230-260](../src/stealth/stealth_scheduler.py#L230-L260). The agent API already supports this fully — `ActorCritic.act()` and `PPOAgent.select_action()` both accept `channel_mask` ([agent.py:156, 296](../src/stealth/rl/agent.py#L156)). The scheduler just drops it. One-line fix:

```python
# in _schedule_rl, replace:
action, _, _ = self._rl_agent.select_action(state)
# with:
mask = env.get_action_mask()
action, _, _ = self._rl_agent.select_action(state, channel_mask=mask)
```

With the mask threaded through, rate-limit violations essentially stop happening at inference, and the "violations still appended to the output schedule" bug fixes itself. For belt-and-suspenders, still skip appending on an env-reported `rate_limit_violation`.

**Unify delay semantics** across the four places that touch delays:

| File | Current | Target |
|---|---|---|
| [src/stealth/rl/environment.py](../src/stealth/rl/environment.py) | before-send | after-send (Phase B) |
| [src/stealth/stealth_scheduler.py](../src/stealth/stealth_scheduler.py) | after-send (ambiguous) | after-send, documented |
| [src/api/server.py](../src/api/server.py) `_transmit_packets_sync` | after-send | after-send, unchanged |
| [src/distribution/noise.py](../src/distribution/noise.py) `NoiseController` | after-send | after-send, unchanged |

Document in a `# Delay convention:` block at the top of `stealth_scheduler.py` so future contributors do not re-introduce the mismatch.

**Scheduler state hygiene.** `StealthScheduler._rl_agent` is lazy-loaded once and reused across every `schedule()` call ([stealth_scheduler.py:52](../src/stealth/stealth_scheduler.py#L52)). The agent's `env.transmission_history` carries state from previous calls, which can leak into `get_warden_score` metrics and into any state features computed over cross-call history. `env.reset(media_ids)` is called in `_schedule_rl`, which already resets `transmission_history`, `channels`, `current_time` — so action-level state is clean. Add two safeguards:

- Explicitly `self._rl_agent.buffer.clear()` at the start of `_schedule_rl` to drop any stale rollout data.
- Document in the scheduler docstring that `schedule()` is not re-entrant (shared env/agent state) — serialize concurrent calls at the API layer, or deep-copy the agent/env per request if real concurrency is needed later.

**Enforce payload / ECC constraints.** The scheduler currently generates a `(delays, channels)` schedule whose length equals `len(media_ids)` from `SemanticEncoder.encode(..., use_ecc=True)`. Add a validator in `StealthScheduler.schedule()` that asserts the output length matches `len(media_ids)` and that no delay is negative. If GAN generates `< seq_len`, pad from the tail of the generated distribution (not with `base_delay`, which looks static to the Warden).

**Keep the auto cascade** at [stealth_scheduler.py:81-96](../src/stealth/stealth_scheduler.py#L81-L96) (`rl → gan → static`) exactly as-is. It already works correctly; `docs/CODEBASE_AUDIT.md` is wrong about this.

**GAN-as-prior (deferred).** Not in this plan. Note in `docs/modules/08_INTEGRATION.md` as future work; the sequential pipeline is enough for the paper's methodology.

---

## Phase D — Evaluation

**Fix [scripts/testing/evaluate_stealth.py](../scripts/testing/evaluate_stealth.py)**. At line ~63, replace the untrained Warden with `warden.load_state_dict(ckpt["warden_state"])` where `ckpt` comes from `storage/models/gan_generator.pt`. Default checkpoint paths should match the scheduler's.

**Metrics the paper will report:**

- **Warden detection AUC** on policy-generated traffic, measured on the Phase A held-out `--val-fraction` real sample (lower = better; target ≈ 0.5).
- **Stealth-throughput Pareto curve** by sweeping `λ_stealth ∈ {0, 10, 25, 50, 100, 200}`.
- **Channel distribution Shannon entropy** (higher = better diversity).
- **Delay distribution KS statistic** vs. the real human-traffic sample.
- **End-to-end wall-clock throughput** (items/minute) measured through the API transmitter.

**Baselines to compare against:**

1. Static `NoiseController` (`mode="static"`).
2. GAN-only (`mode="gan"` with Phase A checkpoint).
3. RL-only trained against a **random** Warden — the current broken path. Included as an ablation to show the Warden-handoff matters.
4. RL trained against a **statistical-feature-only** Warden (strip the BiLSTM+Transformer, keep only `extract_statistical_features`). This is the orthogonal ablation showing the *deep* critic specifically matters — avoids the "straw-man random baseline" framing.
5. **GAN + RL (ours)** — RL trained against the frozen Phase A deep Warden.

**Integration tests.** Extend [tests/test_stealth/](../tests/test_stealth/):

- `test_integration.py`: train a tiny GAN (2 epochs, synthetic data) → freeze Warden → train tiny RL (5 episodes) → load both into `StealthScheduler` → run `.schedule()` end-to-end → assert the output lengths and value ranges.
- `test_rl.py` additions: assert that `_schedule_rl` passes the mask and that rate-limit violations do not land in the final schedule.

---

## Phase E — Repo hygiene / docs

**[docker-compose.yml](../docker-compose.yml)** — verify (manual inspection) that the training services point to the new `scripts/stealth/train_gan.py` and `scripts/stealth/train_rl.py` paths. If any `command:` or `COPY` line disagrees, align it with `storage/models/`.

**Refresh [docs/CODEBASE_AUDIT.md](CODEBASE_AUDIT.md)** to note that `auto` cascade is implemented and that the scheduler uses project-root-anchored paths.

**Add [docs/modules/08_INTEGRATION.md](modules/08_INTEGRATION.md)** — the single-page spec of the integrated pipeline. Contents:

1. Convention table (delay semantics, checkpoint paths, state/action shapes).
2. Sequential training protocol (Phase A → Phase B).
3. Inference call graph (API → `StealthScheduler.schedule` → transmitter).
4. The methodology summary from the box below.

**Scope-out notice.** Add a one-paragraph note at the top of [scripts/stealth/README.md](../scripts/stealth/README.md) stating that the standalone subsystem there is a parallel design and is not integrated with `src/stealth/` or the API.

---

## Phase F — Multi-user pipeline (auth + routing + unified pipeline module)

**Why this phase exists.** Phases A–E give us a working GAN+RL stealth scheduler and a single-tenant demo. For the project to be *complete* as a usable system — and for the paper's methodology section to describe a realistic "A sends to B" scenario rather than an anonymous packet dump — we need: identity (who is Alice? who is Bob?), routing (which packets belong to which recipient?), and a single programmatic entry point that proves the full round-trip (`message in → message out, identical`). The research contribution remains the GAN+RL scheduling; this phase is the thin product shell that makes the research demonstrable.

**Design choices (locked in with the user):**
- Minimal demo auth: bcrypt + SQLite + JWT (no OAuth, no email verification, no password reset).
- 1-to-1 routing only (A picks ONE recipient).
- Minimal UI: `/login`, `/register`, a recipient dropdown on the existing `/encode` or `/transmit` page, and a new `/inbox`.
- Pipeline test uses the "same-process, shared-channel-faithful" model: sender and receiver run in two threads of one Python process, but still go through the real `storage/shared_channel/<recipient>/` directory with real `time.sleep` delays. This is faithful to deployment (file-based transport is tested) but assertable in one pytest run.

### F.1 — Backend auth (FastAPI)

**New [src/api/auth.py](../src/api/auth.py)**:
- SQLite DB at `storage/auth.db` (gitignored). Table `users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)`. Hand-rolled migration on first server start — no SQLAlchemy / Alembic needed for a users table.
- `hash_password(pw: str) -> str` and `verify_password(pw: str, hash: str) -> bool` via `passlib[bcrypt]`.
- `create_access_token(user_id, username, ttl_seconds=86400) -> str` via `python-jose[cryptography]` (HS256, secret from `DCASS_JWT_SECRET` env var — fail-fast at startup if unset in production; generate+log a dev secret otherwise).
- `get_current_user()` FastAPI dependency that parses `Authorization: Bearer <jwt>` → DB lookup → `AuthenticatedUser` pydantic model (or 401).

**New endpoints in [src/api/server.py](../src/api/server.py)**:
- `POST /api/auth/register` — body `{username, password}`. Hash + insert. Returns 201 or 409 on username conflict. Minimum password length: 8.
- `POST /api/auth/login` — body `{username, password}`. Returns `{access_token, token_type: "bearer", user: {id, username}}`.
- `GET /api/auth/me` — returns the current user (gated by `get_current_user`).
- `GET /api/users` — returns all usernames EXCEPT the current user (for the recipient dropdown). Gated.

**Seed users for demo** (idempotent on startup): `alice/alice1234`, `bob/bob1234`, `charlie/charlie1234`. Prints a one-time startup banner naming them and reminding to change passwords for anything beyond local demo use.

**Dependencies to add to [requirements.txt](../requirements.txt)**: `python-jose[cryptography]`, `passlib[bcrypt]`.

### F.2 — Routing (per-recipient shared channel)

**Modify `TransmitRequest`** in [src/api/server.py](../src/api/server.py):
- Add `recipient_username: str` field (required).
- `/api/transmit` becomes `/api/transmit` with `Depends(get_current_user)` — the sender identity is the authenticated user; the recipient is the request body field.

**Modify `_transmit_packets_sync`** to write packets to a per-recipient subdirectory:
```
storage/shared_channel/
├── <recipient_id>/
│   ├── _manifest.json              # includes sender_id, recipient_id
│   ├── <media_id>_<channel>_<seq>.json
│   └── ...
```
The subdirectory is derived from the recipient's integer `user_id` (stable, URL-safe, no quoting). The manifest gets two new fields: `sender_id` and `recipient_id` (both integers, cross-referenced against the users table at read time).

**New endpoints:**
- `GET /api/inbox` — list manifests under `storage/shared_channel/<current_user_id>/`, newest first. Returns `[{manifest_id, sender_username, timestamp, mode_used, total_items, decoded: bool}]`.
- `POST /api/inbox/{manifest_id}/decode` — reassemble packets for that manifest, run `SemanticDecoder.decode`, return the reconstructed message. First-call-decodes-and-caches; subsequent calls return the cached result from a `_decoded.json` sidecar.
- `DELETE /api/inbox/{manifest_id}` — purge the manifest + its packets + sidecar (recipient-only).

**Important: this changes the shared-channel layout.** Existing single-tenant consumers ([run_receiver.py](../scripts/runtime/run_receiver.py), the frontend `/wire` page) must adapt or be scoped-out:
- `run_receiver.py` grows a `--user <username>` flag that watches `storage/shared_channel/<their_user_id>/` instead of the root. Backward-compat flag `--legacy-flat-channel` keeps it watching the root for the non-authenticated demo.
- `/wire` page gains a user-filter and uses the inbox API instead of raw-polling the shared dir.

### F.3 — Unified pipeline module (programmatic + test)

**New [src/stealth/pipeline.py](../src/stealth/pipeline.py)** — the "one function that proves the whole thing works":

```python
@dataclass
class PipelineResult:
    original_message: str
    decoded_message: str
    sender: str
    recipient: str
    mode_used: str              # which scheduling mode actually ran
    round_trip_seconds: float
    decoded_ok: bool            # original == decoded
    verification_rate: float    # fraction of items verified in corpus on decode
    schedule: dict              # {delays, channels, items} from StealthScheduler
    ecc_errors_fixed: list[int] # from the decoder

def run_pipeline(
    message: str,
    sender: str,
    recipient: str,
    mode: Literal["static", "gan", "rl", "auto"] = "auto",
    speed_multiplier: float = 100.0,  # fast-forward delays for tests; 1.0 = real wall clock
    shared_dir: Optional[Path] = None,
    gan_checkpoint: Optional[Path] = None,
    rl_checkpoint: Optional[Path] = None,
) -> PipelineResult:
    """
    Full round-trip through the real transport:
      SemanticEncoder.encode  →  StealthScheduler.schedule  →  transmitter thread
      (writes packets to <shared>/<recipient_id>/ with real sleeps)  →
      receiver thread (ReassemblyBuffer in another thread watches the same dir)  →
      SemanticDecoder.decode  →  compare to original.

    Both threads are joined before the function returns. shared_dir defaults
    to a tmp directory per call so parallel pipeline runs don't collide.
    """
```

Implementation outline:
1. Resolve sender/recipient user_ids from `storage/auth.db` (skip auth-check in programmatic mode — the function argument is authoritative).
2. `SemanticEncoder.encode(message, use_ecc=True)` → `media_ids`.
3. Spawn receiver thread: `ReceiverDaemon(watch_directory=shared_dir / str(recipient_id), silence_threshold=1.0, exit_after_decode=True)`. Capture its final `decoded_messages[-1]`.
4. `StealthScheduler.schedule(media_ids, mode)` → schedule dict.
5. Run `_transmit_packets_sync(schedule, shared_dir / str(recipient_id), message, speed_multiplier)` on this thread (not a thread — we want it to block for correct ordering with the receiver thread).
6. Join receiver thread (with a timeout = `sum(delays)/speed_multiplier * 3 + 10s`).
7. Compare, populate `PipelineResult`, return.

**New [scripts/runtime/run_pipeline.py](../scripts/runtime/run_pipeline.py)** — CLI wrapper over `run_pipeline()`. Flags: `--message`, `--sender`, `--recipient`, `--mode`, `--speed-multiplier`. Prints a formatted summary with ✓/✗ for round-trip success.

**New [tests/test_stealth/test_pipeline.py](../tests/test_stealth/test_pipeline.py)**:
- `test_round_trip_static`: short message, mode=static, speed_multiplier=1000. Assert `result.decoded_ok`.
- `test_round_trip_gan` (gated behind `pytest -m needs_models`): mode=gan, needs a Phase-A checkpoint.
- `test_round_trip_rl` (gated): mode=rl.
- `test_round_trip_auto` (gated): mode=auto.
- `test_pipeline_recipient_isolation`: run two pipelines concurrently with different recipients; assert each receiver only sees their own packets (no leakage via the shared root directory).

### F.4 — Frontend (Next.js, minimal UI)

**New dependency**: `jose` (JWT decode on the client, for expiration checks only — never re-verify signatures client-side). No `next-auth` — keeps the stack simple and matches the "minimal demo auth" choice.

**New files in [frontend/src/](../frontend/src/)**:
- `lib/auth.ts`: `AuthContext`, `useAuth()` hook, `login(username, password)`, `logout()`, `register(username, password)`. Stores JWT in `localStorage` under `dcass_token`. Pros/cons noted in a comment: localStorage is accessible to any XSS; acceptable for a research demo but a note in the README says "use httpOnly cookies before any public deployment".
- `lib/api.ts` **modification**: add an axios request interceptor that reads `dcass_token` and sets `Authorization: Bearer <token>` on every request. Add a response interceptor that triggers logout + redirect to `/login` on any 401.
- `app/layout.tsx` **modification**: wrap children in `<AuthProvider>`.
- `app/login/page.tsx`: username/password form. On success, store token and redirect to `/encode` (or wherever they came from).
- `app/register/page.tsx`: username/password/confirm form. On success, auto-login and redirect.
- `app/inbox/page.tsx`: list manifests from `GET /api/inbox`. Click a row → `POST /api/inbox/<id>/decode` → show decoded message, sender, time, mode_used, verification rate. Delete button.
- Add route guards: any page that calls authenticated endpoints should check `useAuth().user` and redirect to `/login` if null. Minimum: `/encode`, `/decode`, `/wire`, `/inbox`.

**Modifications to existing pages**:
- `app/encode/page.tsx` (or whichever page currently calls `/api/transmit`): add a "Send to:" dropdown populated from `GET /api/users`. The dropdown's selected username is passed as `recipient_username` on the transmit call.
- `app/layout.tsx` or the top nav: show the logged-in username and a logout button when authenticated; show "Login" / "Register" links when not.

### F.5 — Verification

Round-trip pipeline (new, additive to the Phase D verification):
```bash
# 1. Programmatic (fast, no delays)
python scripts/runtime/run_pipeline.py \
  --message "Meet at the cafe at noon" \
  --sender alice --recipient bob \
  --mode static --speed-multiplier 1000
# expect:  ✓ round-trip OK, decoded = original

# 2. Repeat with --mode auto once you have trained checkpoints
python scripts/runtime/run_pipeline.py \
  --message "..." --sender alice --recipient bob --mode auto

# 3. Integration tests
pytest tests/test_stealth/test_pipeline.py -v

# 4. API round-trip via the real server
#    (terminal 1) uvicorn src.api.server:app --reload
#    (terminal 2):
curl -X POST localhost:8000/api/auth/login -d '{"username":"alice","password":"alice1234"}' -H 'content-type: application/json'
# → copy access_token; then transmit; then as bob: curl /api/inbox and /api/inbox/<id>/decode

# 5. UI smoke test
#    cd frontend && npm run dev
#    Browser: register two users, log in as alice, send a message to bob,
#    log out, log in as bob, open /inbox, decode the message, verify it matches.
```

**Success criteria for Phase F:**
- `pytest tests/test_stealth/test_pipeline.py::test_round_trip_static` passes.
- `GET /api/inbox` for Bob returns ONLY messages addressed to Bob (not to Alice or Charlie).
- The UI flow above decodes the message end-to-end.
- Phases A–E's tests still pass (`pytest tests/test_stealth/ -v`).

### F.6 — Risks / open design questions

- **R6 — localStorage JWTs.** XSS-exposable. Fine for a research demo, flagged in the README. Switching to httpOnly cookies is a one-day cleanup if the demo ever goes public — belongs in a follow-up plan, not this one.
- **R7 — `_rl_agent` state across concurrent users.** `StealthScheduler` is lazy-loaded as a module-level singleton in the API process. Two users transmitting concurrently will race on `_rl_agent.env` (Phase C's state hygiene note). For single-user demo this is latent; for the multi-user API it is live. Mitigation: wrap `/api/transmit` in a per-user lock, or instantiate a fresh `StealthScheduler` per request (small cost since lazy-load is cheap on the second call).
- **R8 — Receiver thread lifetime in `run_pipeline`.** `ReceiverDaemon` was written to run forever. The `--exit-after-decode` flag already exists and will stop it after one successful decode; the pipeline module relies on that. If decode never succeeds (e.g. the sender crashes mid-send), the receiver's `silence_threshold` + the pipeline's outer timeout ensures the thread exits. Document the timeout formula.
- **R9 — Shared channel directory namespace collision.** If anyone still uses the "flat" root-level `storage/shared_channel/` (via `run_receiver.py` without `--user`), their packets will collide with per-recipient subdirectory contents. Mitigation: migration step — moves any existing flat-mode packets into `storage/shared_channel/_legacy/` on first server startup after Phase F lands.
- **R10 — Decoder cold-start cost in `run_pipeline`.** `SemanticDecoder.load()` touches FAISS indices and takes seconds on the first call. In the pipeline function, cache the decoder as a module-level singleton (same pattern as `_get_engine()` in `src/api/server.py`) so repeated test runs don't repay the cost.

---

## Risks / open design questions

- **R1 — Positional encoding generalization** (Phase A). See the "Transformer positional encoding caveat" above. Default picked: sinusoidal encodings. If retraining the Warden is undesirable, fall back to padding Warden inference windows to `max_sequence_length`.
- **R2 — λ-curriculum effect partially absorbed by advantage normalization** (Phase B). See the caveat box. Try linear ramp and stepped warmup; report whichever stabilizes training.
- **R3 — Train/eval window mismatch**. The Warden sees max_sequence_length=100 at train time and 20 at inference. Even with (a) sinusoidal encodings, the *statistical* feature extractors (`extract_statistical_features` — autocorr, skew, kurtosis) behave differently on 20 vs 100 samples. Validate the Warden's AUC specifically on 20-step windows before trusting the RL reward signal.
- **R4 — Scheduler re-entrancy**. See "Scheduler state hygiene" in Phase C. Fine for single-threaded CLI/API use; needs attention if the API grows concurrent `/transmit` requests.
- **R5 — Ablation framing**. The "RL against random Warden" baseline is strong but risks looking like a straw man in review. The statistical-feature Warden ablation (baseline #4) is a defensive add; include both.

---

## Verification

Copy-pasteable sequence (run from repo root):

```bash
# 1. GAN pretraining
python scripts/stealth/train_gan.py \
  --data data/human_traffic.json \
  --epochs 20 --batch-size 32 \
  --val-fraction 0.1 \
  --out storage/models/gan_generator.pt

# 2. Verify Warden learned something
python scripts/stealth/eval_warden.py \
  --checkpoint storage/models/gan_generator.pt
# expect: AUC trending toward 0.5 (indistinguishable), G_loss stable
# ALSO expect: AUC at window=20 close to AUC at window=100 (R3)

# 3. RL training against the frozen Warden
python scripts/stealth/train_rl.py \
  --warden-checkpoint storage/models/gan_generator.pt \
  --episodes 500 --lambda-stealth 50 \
  --min-seq-len 30 --max-seq-len 60 \
  --out storage/models/rl_agent.pt

# 4. End-to-end schedule
python -c "
from src.stealth.stealth_scheduler import StealthScheduler
s = StealthScheduler(num_channels=3)
r = s.schedule(['m%d' % i for i in range(20)], mode='auto')
print(r['mode_used'], len(r['delays']), sum(r['delays']))
"
# expect: mode_used=='rl', 20 delays, non-static distribution

# 5. Full evaluation + baselines
python scripts/testing/evaluate_stealth.py \
  --gan-checkpoint storage/models/gan_generator.pt \
  --rl-checkpoint  storage/models/rl_agent.pt

# 6. Tests
pytest tests/test_stealth/ -v
```

Success criteria: all tests pass; `auto` mode returns `mode_used="rl"`; Warden AUC on policy-generated traffic is lower than on static-NoiseController traffic; the λ-sweep produces a monotone throughput/AUC Pareto curve.

---

## Methodology summary (lift into the paper)

> **Pipeline.** DCASS schedules a covert sequence of media IDs $x_{1:T}$ onto $C$ channels by learning two models in sequence. First, a Wasserstein GAN with gradient penalty (WGAN-GP) trains a generator $G_\phi(z, h) \to (\delta_{1:T}, c_{1:T})$ to mimic human inter-arrival delays and channel usage. The critic $D_\psi$ — a BiLSTM–Transformer "Warden" with statistical features — is updated $k{=}5$ times per generator step with
> $\mathcal{L}_D = \mathbb{E}_{\tilde{x}}[D_\psi(\tilde{x})] - \mathbb{E}_{x}[D_\psi(x)] + \lambda_{gp}\,\mathbb{E}_{\hat x}[(\|\nabla_{\hat x} D_\psi(\hat x)\|_2 - 1)^2],\quad \lambda_{gp}=10,$
> where $\hat x$ interpolates both delays and channel soft-probabilities, and the generator is updated with $\mathcal{L}_G = -\mathbb{E}_z[D_\psi(G_\phi(z,h))]$. The channel head is trained through a straight-through Gumbel-Softmax so gradients reach the discrete action. Sinusoidal positional encodings in $D_\psi$ let it generalize across window lengths.
>
> **Reward model.** The trained critic $D_\psi^\ast$ is frozen (eval mode, no gradients) and reused as a detection oracle. A proximal policy optimization (PPO) agent $\pi_\theta$ then learns a scheduling policy over state $s_t$ (queue, cyclic hour-of-day, per-channel cooldown, recent-window delay/entropy statistics) and action $a_t = (\delta_t, c_t)$, with reward
> $R_t = \text{throughput}_t - \lambda_{stealth}\cdot \sigma(D_\psi^\ast(\tau_t)) + \beta\cdot\text{div}_t,$
> where $\tau_t$ is the most recent $w$-step traffic window (mean-delay padded for $t<w$ so the stealth signal is non-sparse early), $\sigma$ is sigmoid, and $\text{div}_t$ is a channel-switch and Shannon-entropy bonus. PPO updates minimize the standard clipped surrogate $\mathcal{L}^{CLIP} = -\mathbb{E}_t[\min(r_t(\theta) A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon) A_t)]$ with $\epsilon=0.2$, GAE $\lambda=0.95$, $\gamma=0.99$, plus value and entropy terms. A stealth-penalty curriculum ramps $\lambda_{stealth}$ from 10 to the target over the first 20% of episodes.
>
> **Inference.** The scheduler loads $(G_\phi^\ast, \pi_\theta^\ast)$ and dispatches media IDs with the policy's $(\delta_t, c_t)$ masked to legal channels by the environment's per-channel rate limits; the GAN serves as a fallback and a runtime stealth monitor via $D_\psi^\ast$. A static handcrafted noise profile is a terminal fallback when no checkpoint is present.

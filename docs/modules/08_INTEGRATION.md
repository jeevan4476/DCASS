# Module 08 — GAN + RL Integration

This is the single-page spec of the integrated stealth-scheduling pipeline:
GAN (WGAN-GP) pretrains a Warden, the Warden is frozen and reused as the RL
reward model, and `StealthScheduler` serves whichever trained policy is
available at inference. See [docs/GAN_RL_INTEGRATION_PLAN.md](../GAN_RL_INTEGRATION_PLAN.md)
for the build plan this module implements.

**Out of scope:** the standalone subsystem under [scripts/stealth/](../../scripts/stealth/)
that predates this integration (its own WGAN-GP over timestamps,
`gan/gan.py`, plus a separate Double-DQN path selector under `path_agent/`)
is a parallel design and is **not** part of this pipeline. See the notice at
the top of [scripts/stealth/README.md](../../scripts/stealth/README.md).

## 1. Convention table

| Concept | Convention |
|---|---|
| Delay semantics | `delays[i]` is the pause AFTER item i is sent. Dispatch, then sleep. Followed by `NoiseController`, the API transmitter, `StealthEnvironment`, and `StealthScheduler`. |
| GAN checkpoint path | `storage/models/gan_generator.pt` (host) ↔ `/app/models/gan_generator.pt` (training containers) ↔ `/app/storage/models/gan_generator.pt` (sender/api containers) — same host file, different container mount points. |
| RL checkpoint path | `storage/models/rl_agent.pt`, same mount-point pattern as above. |
| GAN checkpoint schema | `{generator_state, warden_state, generator_optimizer, warden_optimizer, config: TrainingConfig, epoch, global_step, metrics_history}` |
| RL checkpoint schema | `{actor_critic_state, optimizer_state, config: PPOConfig, episode_rewards, episode_lengths, warden_scores}` |
| RL state vector | 16-dim for `num_channels=3`: queue fraction (1) + hour sin/cos (2) + per-channel [rate_limit, idle_time, usage_share] (3×3) + history [mean_delay, delay_std, channel_diversity, tx_rate] (4). `PPOConfig.state_dim` default (21) is always overridden by `env.state_dim` at construction time. |
| RL action | `{"delay": float seconds >= 0.5, "channel": int}` |
| Warden critic score | `raw_critic_score` (unbounded) is used for WGAN-GP losses and the gradient penalty. `bot_probability = sigmoid(raw_critic_score)` is used for RL reward and human-readable reporting. Never mix the two. |

## 2. Sequential training protocol

```
Phase A (GAN)                          Phase B (RL)
──────────────                         ─────────────
HumanTrafficDataset                    frozen Warden (from A)
   │ split(val_fraction)                      │
   ▼                                           ▼
GANTrainer.train()                     StealthEnvironment(warden=...)
 ├─ Warden: WGAN-GP critic loss                │
 │   + gradient penalty (delays                ▼
 │   AND channel soft-probs)           PPOAgent.train()
 └─ Generator: -E[D(fake)]              ├─ collect_rollout (masked actions)
   │                                    └─ update (clipped surrogate, GAE)
   ▼                                           │
storage/models/gan_generator.pt                ▼
   │                                   storage/models/rl_agent.pt
   ▼
eval_warden.py → AUC ≈ 0.5 target
```

Run order: `scripts/stealth/train_gan.py` → `scripts/stealth/eval_warden.py`
(sanity-check) → `scripts/stealth/train_rl.py --warden-checkpoint <gan ckpt>`.

## 3. Inference call graph

```
POST /api/transmit  (src/api/server.py)
   │  or  scripts/runtime/run_sender.py
   ▼
StealthScheduler.schedule(media_ids, mode)
   │
   ├─ mode="auto": _schedule_rl → (fallback) _schedule_gan → (fallback) _schedule_static
   ├─ mode="rl":   _schedule_rl   (loads PPOAgent + untrained-placeholder Warden container;
   │                               the env's Warden here is only used if a *new* model were
   │                               trained live — normal inference reuses the trained policy's
   │                               learned behavior, not the Warden's live score)
   ├─ mode="gan":  _schedule_gan  (loads TemporalPatternGenerator, samples z ~ N(0,I))
   └─ mode="static": _schedule_static (NoiseController + ACTIVITY_PROFILES)
   │
   ▼
_validate_schedule()  — asserts len(delays) == len(media_ids), no negative delays
   │
   ▼
{items, delays, channels, mode_used}
   │
   ▼
Transmitter (API thread _transmit_packets_sync, or run_sender.send_sequence)
   writes one JSON packet per item to storage/shared_channel/, sleeps `delay`
   after each send (see delay convention above)
   │
   ▼
Receiver (run_receiver.py) — ReassemblyBuffer sorts by sequence_number
```

## 4. Methodology (for the paper)

> **Pipeline.** DCASS schedules a covert sequence of media IDs $x_{1:T}$ onto $C$ channels by learning two models in sequence. First, a Wasserstein GAN with gradient penalty (WGAN-GP) trains a generator $G_\phi(z, h) \to (\delta_{1:T}, c_{1:T})$ to mimic human inter-arrival delays and channel usage. The critic $D_\psi$ — a BiLSTM–Transformer "Warden" with statistical features — is updated $k{=}5$ times per generator step with
> $\mathcal{L}_D = \mathbb{E}_{\tilde{x}}[D_\psi(\tilde{x})] - \mathbb{E}_{x}[D_\psi(x)] + \lambda_{gp}\,\mathbb{E}_{\hat x}[(\|\nabla_{\hat x} D_\psi(\hat x)\|_2 - 1)^2],\quad \lambda_{gp}=10,$
> where $\hat x$ interpolates both delays and channel soft-probabilities, and the generator is updated with $\mathcal{L}_G = -\mathbb{E}_z[D_\psi(G_\phi(z,h))]$. The channel head is trained through a straight-through Gumbel-Softmax so gradients reach the discrete action.
>
> **Reward model.** The trained critic $D_\psi^\ast$ is frozen (eval mode, no gradients) and reused as a detection oracle. A proximal policy optimization (PPO) agent $\pi_\theta$ then learns a scheduling policy over state $s_t$ (queue, cyclic hour-of-day, per-channel cooldown, recent-window delay/entropy statistics) and action $a_t = (\delta_t, c_t)$, with reward
> $R_t = \text{throughput}_t - \lambda_{stealth}\cdot \sigma(D_\psi^\ast(\tau_t)) + \beta\cdot\text{div}_t,$
> where $\tau_t$ is the most recent $w{=}20$-step traffic window (mean-delay padded for $t<w$ so the stealth signal is non-sparse early), $\sigma$ is sigmoid, and $\text{div}_t$ is a channel-switch and Shannon-entropy bonus. PPO updates minimize the standard clipped surrogate $\mathcal{L}^{CLIP} = -\mathbb{E}_t[\min(r_t(\theta) A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon) A_t)]$ with $\epsilon=0.2$, GAE $\lambda=0.95$, $\gamma=0.99$, plus value and entropy terms. A stealth-penalty curriculum ramps $\lambda_{stealth}$ from 20% of the target to the target over the first 20% of episodes.
>
> **Inference.** The scheduler loads $(G_\phi^\ast, \pi_\theta^\ast)$ and dispatches media IDs with the policy's $(\delta_t, c_t)$ masked to legal channels by the environment's per-channel rate limits; the GAN serves as a fallback and a runtime stealth monitor via $D_\psi^\ast$. A static handcrafted noise profile is a terminal fallback when no checkpoint is present.

## 5. Open risks (see docs/GAN_RL_INTEGRATION_PLAN.md for full detail)

- **R1/R3** — the Warden is trained on batches up to `max_sequence_length=100`
  (masked/padded correctly as of this integration) but evaluated by the RL
  environment on 20-step windows. Validate AUC at both lengths before
  trusting the reward signal (`eval_warden.py` reports both).
- **R2** — the λ-stealth curriculum's effect is partially absorbed by PPO's
  advantage normalization; try both a linear ramp and a stepped warmup.
- **R4** — `StealthScheduler` is not re-entrant; serialize concurrent
  `/transmit` requests or give each request its own scheduler instance.

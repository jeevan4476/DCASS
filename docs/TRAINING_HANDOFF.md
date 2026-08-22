# Training Handoff - GAN / RL Stealth Models

> For the teammate running training on the GPU machine.
> **Run the tier-3 tests first. If they fail, do not start training** -
> you would be training a broken objective.

## 0. Pre-flight (CPU, < 1 minute)

```bash
# 1. Environment sanity + artifact binding (must print VERDICT: READY)
python -m src.cli.main doctor

# 2. Tier-3 gradient-flow proofs for the trainer fixes
pytest tests/test_engine/test_payload_framing.py -v -k "Gradient or ChannelHead or Detach"
```

What those tests prove, so you can trust the code before spending GPU time:

| Test | Fix it proves |
|---|---|
| `test_penalty_computes_and_backprops` | R-22: WGAN-GP penalty computes and backprops |
| `test_raw_score_gradient_not_sigmoid_squashed` | R-22: penalty constrains ‖∇D‖ on `raw_critic_score`, not the sigmoid view |
| `test_generator_channel_head_gets_nonzero_grad` | R-23: straight-through Gumbel-Softmax delivers gradient to the channel head |
| `TestCriticLoopDetachesGenerator` | R-24: critic loop detaches fake delays |

Also verify `RL state_dim == 16` (P1-5 fix): `python -c "from src.stealth.rl.environment import StealthEnvironment; print(StealthEnvironment().state_dim)"` → expect `16`.

## 1. Data collection

```bash
# Collect real traffic delays (or reuse an existing dataset)
python scripts/stealth/collect_real_traffic.py --help
python scripts/stealth/generate_traffic_dataset.py --help
```

Canonical entry points (the old `scripts/training/*` duplicates are deleted;
docker-compose now calls these):

- `scripts/stealth/train_gan.py` - GAN training
- `scripts/stealth/train_rl.py` - PPO training
- (`train_gan_extended.py` exists for long-run experiments; keep flags identical)

## 2. GAN training

```bash
python scripts/stealth/train_gan.py --epochs 200 [see script for flags]
```

Expectations:

- Wall-clock: ~1-2 h for 200 epochs on a single consumer GPU (batch 64,
  seq len <= 100); scales roughly linearly in epochs.
- VRAM: < 4 GB.
- Checkpoints land in `storage/checkpoints/gan/` via the trainer's
  `save_checkpoint()` (config included in each checkpoint).
- "Working" looks like:
  - warden loss oscillating near its Nash equilibrium (not diverging),
  - generator loss bounded (Wasserstein estimate, no blow-up),
  - gradient penalty term small and stable after warmup,
  - generated delay histograms visually matching the real data's
    burstiness (compare with `scripts/stealth/benchmark_gan_timing.py`).

Red flags: generator loss exploding to ±100s, warden loss pinned at one
value (mode collapse), or NaNs - stop and report.

## 3. PPO training

```bash
python scripts/stealth/train_rl.py --episodes 500 [see script for flags]
```

Expectations:

- Wall-clock: ~2-4 h for 500 episodes on GPU (environment is CPU-bound;
  the warden forward dominates).
- "Working" looks like:
  - episode reward trending up over ~100 episodes,
  - warden score in the environment logs drifting toward chance (~0.5),
  - channel usage entropy rising then stabilising (the agent should spread
    across channels, not collapse to one).

Note: the channel head trains through straight-through Gumbel-Softmax now
(R-23). If channel logits stay uniform forever, something regressed.

## 4. After training

```bash
# Copy checkpoints where the scheduler expects them
cp storage/checkpoints/gan/<best>.pt storage/models/gan_generator.pt
cp storage/checkpoints/rl/<best>.pt  storage/models/rl_agent.pt

# Scheduler reads dims from the checkpoint config; verify load works on CPU:
python scripts/testing/test_stealth_system.py   # smoke test

# Benchmark before declaring victory
python scripts/stealth/benchmark_gan_timing.py
python scripts/stealth/benchmark_rl_agent.py
```

The API `/api/status` endpoint reports checkpoint presence; the frontend
picks modes up automatically (`auto` = rl -> gan -> static cascade).

## 5. Reporting back

Include in your report: final generator/warden loss curves, warden score
trajectory, benchmark outputs, and the git commit you trained from
(`git rev-parse --short HEAD`; checkpoints also store it if blessed).

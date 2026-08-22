# Roadmap - Next Phase (V1 → Paper)

**Context:** All audit findings and IMPLEMENTATION_PLAN work packages are closed
(see [FIX_AND_DECISION_LOG.md](./FIX_AND_DECISION_LOG.md)). Three tracks now run
in parallel: **R — Model Retraining** (owner: Jeevan, on GPU machine),
**A — Research Paper** (starts when V1 exits), **B — System Improvisation**
(continuous). V1 is done when Phase R completes its promotion gate.

---

## Track R — Model Retraining (owner: Jeevan)

The GAN/RL checkpoints do not exist yet; the system currently runs in static
mode. Every code-side dependency for training is provable on CPU before any
GPU time is spent. Full operational detail lives in
[TRAINING_HANDOFF.md](../TRAINING_HANDOFF.md); this section is the *gated
checklist*.

### R-0 · Pre-flight gate — DO NOT SKIP

- [ ] `git pull` to commit ≥ `14abc4e` on the training machine
- [ ] `pip install -r requirements.txt` + CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- [ ] Copy/symlink `storage/data/indices/` (incl. `voronoi_codebook.meta.json`) to the training machine
- [ ] `python -m src.cli.main doctor` → **VERDICT: READY**
- [ ] `pytest tests/test_engine/test_payload_framing.py -v -k "Gradient or ChannelHead or Detach"` → all green (proves R-22/R-23/R-24 fixes before trusting any loss curve)
- [ ] `python -c "from src.stealth.rl.environment import StealthEnvironment; print(StealthEnvironment().state_dim)"` → `16`
- [ ] Record `git rev-parse --short HEAD` and torch/CUDA versions in your run notes

> If ANY box above fails: stop. File the failure; do not start training a broken objective.

### R-1 · Dataset

- [ ] Collect/refresh real traffic delays via `scripts/stealth/collect_real_traffic.py`
- [ ] Build the training set with `scripts/stealth/generate_traffic_dataset.py`
- [ ] Sanity-check the delay distribution histogram (bimodal? gaps? outliers?) before training on it

### R-2 · GAN training (WGAN-GP traffic mimicry)

Entry point: `scripts/stealth/train_gan.py`

- [ ] Run with gradient penalty ENABLED (now meaningful post-R-22)
- [ ] Watch: warden loss oscillating near equilibrium, generator loss bounded, GP term small after warmup
- [ ] Red flags → abort: NaNs, |G loss| exploding past ~100s, warden pinned at one value (mode collapse)
- [ ] Acceptance: generated delay histogram visually matches real data's burstiness (`scripts/stealth/benchmark_gan_timing.py`)
- [ ] Save best checkpoint + record wall-clock & VRAM used

### R-3 · PPO training (closed-loop scheduler)

Entry point: `scripts/stealth/train_rl.py`

- [ ] Confirm channel-head gradients flow (post-R-23): if channel logits stay perfectly uniform forever, something regressed
- [ ] Watch: episode reward trending up by ~100 episodes; warden score drifting toward ~0.5; channel-usage entropy rising then stabilizing
- [ ] Red flags → abort: reward collapse, agent collapsing onto one channel, warden score stuck near 1.0
- [ ] Acceptance: `scripts/stealth/benchmark_rl_agent.py` beats static-mode baseline on stealth metrics without tanking throughput
- [ ] Save best checkpoint

### R-4 · Promotion gate (checkpoints → runtime)

- [ ] Copy chosen checkpoints to `storage/models/gan_generator.pt` and `storage/models/rl_agent.pt`
- [ ] `/api/status` reports both present; scheduler `auto` mode resolves rl→gan→static
- [ ] End-to-end smoke: encode a message in the UI, transmit with `auto`, receive+decode exact
- [ ] Re-run `dcass doctor` — still READY
- [ ] Tag the commit + checkpoint pair (e.g., git tag `v1-rc1`)

### R-5 · Rollback story

- [ ] Keep the previous `storage/models/*.pt` (or absence = static mode) restorable; deleting the files returns the system to guaranteed-working static scheduling
- [ ] Any regression found post-promotion: delete/rename checkpoint file first, investigate second

### R-6 · Report back

- [ ] Loss curves (G/W), warden-score trajectory, benchmark outputs
- [ ] Wall-clock + VRAM per phase, dataset summary stats
- [ ] Commit hash trained from + checkpoint filenames

---

## V1 exit criteria

All of: R-4 complete · doctor READY with checkpoints present · keyed end-to-end demo through the UI succeeds · test suite green on the promoting commit. When these hold, tag **v1.0** and start Track A in earnest.

---

## Track A — Research Paper (parallel from V1 onward)

Goal: a defensible paper whose every claim maps to measured evidence.

### A-1 · Claims ↔ evidence matrix (first task)
For each scoped claim in `docs/modules/06_SECURITY_AND_STEGANALYSIS_DEFENSE.md`,
list: claim → experiment → dataset → metric → result slot. Nothing enters the
paper without a row. Known open rows:
- "Content-residual steganalysts perform at chance on unmodified carriers" —
  currently *theoretical* (~50% expected); needs SRNet/Zhu-Net/SRM runs on
  transmitted carriers vs corpus originals
- "DPI timing mimicry reduces detection" — currently qualitative; needs the
  warden-vs-schedule benchmark numbers per mode (static/GAN/RL)

### A-2 · Required experiments
- [ ] Empirical content-steganalysis benchmark (tier: GPU) — carriers vs originals through 2-3 published detectors
- [ ] Selection-pattern analysis — can an adversary distinguish DCASS carrier picks from corpus sampling statistics? (the D_KL=0 argument covers content, NOT selection)
- [ ] DPI/timing study — warden detection rate vs {static jitter, GAN, RL} schedules across message lengths (ties to module 08 capacity math)
- [ ] Ablation suite — context keying off / framing off / ECC off / `semantic_legacy` baseline vs exact_vcp (legacy mode kept precisely for this)
- [ ] Cover-story evaluation — coherence vs plaintext-independence trade-off measurement; hook into `docs/research/RESEARCH_LLM_NARRATIVE_COHESION_GUARD.md` work for LLM-generated narrative cover stories
- [ ] Keyed-channel security discussion — time-bucket obfuscation vs secret-keyed derivation; wrong-key fail-closed rates from the test suite

### A-3 · Writing
- [ ] System architecture section drafts directly from `docs/modules/01..08` (already truth-passed)
- [ ] Threat-model section from module 06 scoped language
- [ ] Capacity/cost section from module 08 (carriers-per-byte math)

---

## Track B — System Improvisation (continuous, prioritized)

| # | Item | Why | Size |
|---|---|---|---|
| B1 | Frontend controls for dynamic context keys + cover story | Features exist backend-only; UI makes them demonstrable | M |
| B2 | GPU CI runner job (tier-2/4 suites need indices) | Keeps artifact-dependent tests honest on every PR | S |
| B3 | External source hardening study (CoinGecko quantization, pinned endpoints, multi-source quorum) | Live signals are the weakest link of epoch sync | M |
| B4 | LLM narrative cover stories (research link above) | Turns Decision 5's decoy list into programmable personas | L |
| B5 | Base-K′ transport contingency | Only if a future corpus measures K′<256; validation error already points here | M (contingent) |
| B6 | Warden-in-the-loop co-training cycle (retrain warden on transmitted schedules, retrain scheduler against it) | Continuous adversarial pressure; paper-worthy dynamics | L |
| B7 | torch≥2.6 upgrade / safetensors CLAP | Restores native CLAP audio ranking (currently CLIP fallback) | S |
| B8 | Packaging/release (versioned artifacts, release notes automation) | V1 polish | S |

Suggested order while waiting on GPU availability: B1 → B7 → B2 → B3 → B4 → B6.

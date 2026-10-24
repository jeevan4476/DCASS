# Component 3: The Traffic Detector, the RL Pacing Defects, and the Held-Out Evaluation

**Date:** 2026-10-21
**Status:** the C3 soft branch is measured against a held-out detector, the RL
agent is retrained against a working one, and the PPO row is a measurement
rather than an artifact.

This document records the work that followed
[`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md). That diagnostic
established *why* the shipped warden could not detect anything. This one
records what was done about it, and what the measurement now says.

---

## 1. The summary

The shipped pipeline never optimized for stealth, and the Component 3
result was measuring the consequence.

The RL environment computes its stealth penalty as
`verdict.bot_probability * lambda_stealth`. The warden the shipped agent
trained against is the WGAN-GP critic stored inside
`storage/models/gan_generator.pt`, and that critic returns
`bot_probability = 0.0` for **every** input, including a fixed 3.0 second
metronome. The penalty was therefore identically zero for the whole of
training, and the reward function reduced to throughput plus channel
diversity, which prefers fast traffic. Measured on the same episodes:

| traffic | collapsed critic (shipped) | trained detector |
| :--- | :--- | :--- |
| fixed 3.0 s metronome | score 0.0000, reward **+21.57** | score 1.0000, reward **−28.43** |
| fixed 20.0 s (30 clamped to the canonical window) | score 0.0000, reward +7.67 | score 1.0000, reward −42.33 |

Both rows disclosed: `canonical_rl_delay` clamps every step into the
pacing window `[0.5, 20.0]` inside `env.step`, so the 30 s probe row
scores as a fixed 20.0 s clock. The metronome row is stretched by the
environment's rate-limit back-offs — on the `i%3` rotation the cooldowns
are 6/12/4 s, so each channel-1 step of a 3.0 s metronome waits out the
full 12 s channel-1 cooldown before it is scored. The frozen
`rl_stealth_reward_probe.json` records the probes under their original
labels and is not rewritten here.

So the agent was not merely unstealthed. It was rewarded for the most
bot-like traffic possible, and it is a throughput-and-diversity policy
rather than a stealth policy. The measured rewards are in
`storage/benchmarks/rl_stealth_reward_probe.json`. The generator has the same problem: its
adversarial signal came from the same collapsed critic.

## 2. Four defects in the RL path

Fixing the measurement was not enough. The RL path had four defects that
had to be fixed before an agent's evasion could mean anything.

Three of the four defects below are quantified by percentages — 98.3,
31.4, and 23.5 — that are **code-derived, not measurements**: they were
computed from the archived code path's arithmetic rather than recorded
into a benchmark artifact, and no artifact of that intermediate state
exists. The fourth (the raw-signal defect itself) is established by
`rl_stealth_reward_probe.json`.

### 2.1 The agent optimized delays that never reached the network

The environment scored the agent's **raw** delay. The scheduler put
`clip(raw * 0.35, 7.0, 20.0)` on the wire. The reward was a function of
a signal that never reached the network, and the transform's 7 second
floor sat above almost the whole of the agent's action range: **98.3
percent of the choices the agent made collapsed onto the single value
7.00 seconds**, and the transmitted traffic was a constant clock.

The fix is a canonical pacing contract in
[`src/stealth/rl/pacing.py`](file:///home/img1/projects/DCASS/src/stealth/rl/pacing.py),
used by the agent, the environment and the scheduler alike. The agent
clamps its action into the window *before* computing the log-probability,
so the PPO ratio stays consistent, and the scheduler deploys the delay
the environment actually used, back-off included.

### 2.2 The pacing floor excluded the short-delay mass human traffic has

The 7 second floor was a global constraint on every transmission. The
per-channel cooldown is a *per-channel* constraint, and the pipeline
already enforces it with the action mask, which masks out any channel
still cooling down. The global floor was unnecessary, and it was
destructive: the pooled training corpus (both traffic files, 539,540
delays — the corpus the detector actually sees) has a median inter-arrival
of 4.18 seconds and 65.2 percent of its delays below 7 seconds. A 7 second floor
excludes exactly the mass that makes human traffic look human.

The window is now 0.5 to 20 seconds, the lower bound being the shortest
physically meaningful gap, with the per-channel cooldown left to the
mask.

### 2.3 The environment's cooldown test was more permissive than the benchmark's

The environment recorded a channel's clock at the time an item was
**sent**; the benchmark measures the gap between consecutive **landings**
on a channel. The environment's test was therefore permissive by one
delay, and an agent that satisfied the environment still violated the
benchmark's cooldowns in 31.4 percent of steps. The channel clock now
records the landing time, which is the quantity the platform sees and the
quantity the benchmark measures.

### 2.4 A rate-limit violation was priced below a soft failure

The environment dropped a step that violated a cooldown and charged a
flat −10, while a detection cost −50. The agent therefore took the hard
failure, because a dropped transmission was cheaper than being flagged.
Against a working detector it produced traffic that violated the
per-channel cooldown in 23.5 percent of steps: the evasion was real
against the detector and worthless in deployment, because a rate-limit
violation is an HTTP 429, not a soft signal.

A channel that is still cooling down is now a **back-off**, which is what
a rate-limited client actually does. The delay is extended to the
earliest time the channel is free, the transmission proceeds, and no
transmission is ever illegally dropped.

## 3. The evaluation is held out on purpose

If the agent trains against detector **A** and Component 3 then measures
it with **A**, the number measures how well the agent gamed its own
opponent. That is precisely the failure mode this whole remediation
exists to eliminate.

So there are two detectors:

| | path | seed | role |
| :--- | :--- | :--- | :--- |
| adversary | `storage/models/traffic_detector.pt` | 42 | the RL agent optimizes its bot probability |
| evaluator | `storage/models/traffic_detector_eval.pt` | 1337 | Component 3 measures with it |

The two are independently trained: different seeds give different
splits, different initializations and different synthesized bot
sessions, and no weight tensor is shared. The held-out rows overlap by
250 of 1,800, which is what two independent random splits of the same
corpus give.

The margin is not a knife edge. The agent's bot probability is 0.0040 to
0.0047 across its 50 trials, the human population's 95th percentile is
0.0133, and the generators score 0.994. The verdict does not depend on
where exactly the threshold falls.

## 4. A modelling error in the human reference

The benchmark's synthetic human reference was a **session-level**
mixture: 70 percent exponential, 20 percent log-normal, and 10 percent
periodic session gaps at 3600 seconds. The benchmark measures 50-carrier
**bursts**, and a burst is a within-session sequence.

A 50-item draw from a pool carrying 10 percent hour-scale gaps contains
one with near certainty, 299 times out of 300 in measurement, and the
corpus the detector was trained on is bounded at 120 seconds. So
essentially every surrogate burst was outside the detector's training
distribution, and the detector scored them at 0.9696 instead of 0.0174.

The consequences were real (this episode is narrative — the intermediate
calibration run was not retained as an artifact). Including the surrogate
in the threshold calibration put the threshold at 0.97, at which the detector flags almost
nothing, which is not a meaningful operating point. And the KS reference
was comparing a burst against a session mixture, so it was measuring the
presence of session gaps rather than the timing fidelity of a burst.

The reference is now burst-level, and the session-gap component is gone.
The recorded KS distances changed accordingly, and the positive control
returned to its earlier value of 12 percent, all of it through the burst
rule.

## 5. What the measurement now says

| Scheduler | Interception | KS to reference | Mean delay | Cooldown violations | Score band |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Static Cadence | 100.0% | 0.5330 | 3.00 s | 2450 | 0.980 |
| Stochastic Poisson | 54.0% | 0.2211 | 3.04 s | 1276 | 0.004 to 0.890 |
| DCASS Temporal WGAN | 100.0% | 0.5595 | 7.50 s | 982 | 0.3300 to 0.9945 |
| DCASS Hybrid Engine | 100.0% | 0.5553 | 7.44 s | 13 | 0.3976 to 0.9945 |
| **DCASS PPO RL Agent** | **0.0%** | 0.2538 | 5.20 s | **15** | 0.004 |

The soft branch's threshold is 0.0181, calibrated on 457 human bursts at a
measured 4.60 percent false-positive rate. The positive control flags 6 of
50 surrogate bursts (12 percent, every one through the burst rule) and 1
of 50 held-out corpus sessions (2 percent).

The PPO arm is the first row in this component that is not flagged. It is
also the only one that respects the platform's cooldowns: 15 violations
across 50 trials, against 2,450 for the static baseline, which is a
constant 3.0 second clock and therefore breaches the 12 second channel-1
cooldown on nearly every step.

One precision caveat about the violation column. Up to the 2026-10-23
re-audit the counter accumulated the landing times in float32 (a cast
between the exported schedule and the counting loop). Float32 rounding is
not trustworthy at a cooldown boundary in either direction — it can erode
an exact-boundary landing below the threshold and count a violation that
never happened, or it can round a genuinely sub-threshold gap back onto
the boundary and hide one; both directions were reproduced against seeded
traffic when the finding was verified. The counter now measures in the
float64 the executors produce. The table values above are the frozen
recorded artifacts and are not re-measured here, so no claim is made
about which of the recorded counts, if any, a float64 re-run would move.

### 5.1 What this does not establish

The human side of the comparison is the repository's own traffic corpus
and a synthetic surrogate. "Human" here means *matches that corpus's
delay distribution*. It is not a statement about a real observer, and
three limits travel with the number:

1. **The detector is narrow.** It was trained to catch three families,
   a near-fixed cadence, a bursty delivery pattern, and a human session
   compressed into a narrow band. The agent found a delay distribution
   outside all three, which is what the detector was asked to find. A
   detector with a richer model of human timing might not agree.
2. **The corpus is synthetic.** Its time-of-day distribution is uniform,
   which real browsing never is. A real traffic capture is a data
   problem, not a code problem, and it is the one thing that would move
   this from a claim about the repository's corpus to a claim about the
   world.
3. **The evasion is against a timing detector.** The channel structure of
   the agent's traffic is unlike the corpus's, and a warden that looked
   at channel routing as well as timing would see it. The component
   measures a timing warden, and says so.

## 6. What is still open

The generator's adversarial signal came from a WGAN-GP critic whose
training diverged. That work is in
[`C3_GENERATOR_ADVERSARY.md`](C3_GENERATOR_ADVERSARY.md), which diagnoses
the divergence, rules out the architectural hypothesis the diagnostic
proposed, identifies the mechanism behind it, and records the repair: the
critic is now given the traffic the scheduler sends, the delay mapping no
longer leaves the generator without a gradient, and the critic's score
offset is anchored. The generator was retrained against the repaired
path.

The shipped `storage/models/warden.pt` is still the RL environment's
stealth penalty for any code path that uses it directly — though no
current `src/` path does (only the diagnostics in `scripts/analysis/` and
this benchmark's historical note load the file). The pipeline
uses the trained detector instead. It is retained as the subject of the
diagnostic, not as a usable model.

## 7. Artifacts

| artifact | what it records |
| :--- | :--- |
| `storage/models/traffic_detector.pt` | the adversary, seed 42 |
| `storage/models/traffic_detector_eval.pt` | the held-out evaluator, seed 1337 |
| `storage/models/rl_agent.pt` | the retrained RL agent |
| `storage/benchmarks/traffic_detector_results.json` | adversary training summary and holdout AUC |
| `storage/benchmarks/traffic_detector_eval_results.json` | evaluator training summary and holdout AUC |
| `storage/benchmarks/traffic_detector_generalization.json` | the detector's bot probability on six traffic populations |
| `storage/benchmarks/rl_stealth_reward_probe.json` | the reward the RL environment assigns under each warden |
| `storage/benchmarks/soft_branch_design_comparison.json` | each candidate soft-branch score at a 5 percent false-positive rate |
| `storage/benchmarks/component_3_results.json` | the recorded measurement, including the threshold and its false-positive rate |
| `storage/benchmarks/archive/2026-10-21-pre-remediation/rl_agent.pt` | the agent that was shipped before this work |

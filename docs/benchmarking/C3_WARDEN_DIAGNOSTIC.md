# Component 3 Warden Diagnostic

**Date:** 2026-10-21
**Scope:** Remediation item B2 (timeboxed warden diagnostic, maximum one day)
**Question:** Why does the shipped warden return a constant anomaly score of
0.62493 and `bot_probability = 0.0` on every input?

**Conclusion in one line:** the root cause is a degenerate checkpoint, not a
threshold or normalization bug. The shipped warden's transformer collapsed
to a constant-output function during WGAN-GP training, and its final
sigmoid saturates into a fixed value. The `bot_probability = 0.0` figure is
a float32 underflow of `sigmoid(-90.25)`, not a measurement.

**Outcome:** the C3 result is frozen as a negative/limited result. No
positive evasion claim is made or supportable from these artifacts. See
"Repairs attempted" and "Why the diagnostic stops here".

---

## 1. What the recorded artifacts show

From `component_3_results.json`, every scheduler row carries the same
`mean_neural_anomaly`:

| Scheduler | `mean_neural_anomaly` | `interception_rate` | Triggering rule |
|---|---|---|---|
| Static Cadence | 0.6249254942 | 100.0% | periodic_cv 50/50 |
| Stochastic Poisson | 0.6249254942 | 60.0% | burst 30/50, soft 20/50 |
| DCASS Temporal WGAN | 0.6249254942 | 0.0% | soft 50/50 |
| DCASS Hybrid Engine | 0.6249254942 | 0.0% | soft 50/50 |
| DCASS PPO RL Agent | 0.6249225271 | 30.0% | soft 35/50, periodic_cv 15/50 |

The interception rates in this table are the ones recorded against the
collapsed warden. They are superseded: the current artifact records 100%
for the WGAN and Hybrid rows, and 0% for the retrained PPO arm, because the soft branch is no longer
the branch that could not flag. See section 9.

The anomaly score is identical to four decimal places across a constant
3.0 s metronome, an exponential process, and a 50-item generative schedule:
the spread across the six probes is 9.95e-06, as section 3 measures.
A detector that cannot tell those apart is not measuring anything.

## 2. Diagnostic method

Three scripts were written, all reproducible from the repository root:

| Script | What it establishes |
|---|---|
| `scripts/analysis/diagnose_warden.py` | The warden's outputs against six deliberately different delay sequences, with per-layer activation statistics |
| `scripts/analysis/diagnose_warden_layers.py` | Where in the network the input signal dies, via forward hooks on every submodule |
| `scripts/analysis/diagnose_warden_components.py` | Which component is at fault, by swapping the trained transformer and anomaly head against fresh initializations |

## 3. Finding 1: the outputs are constant across all inputs

`diagnose_warden.py` feeds the shipped checkpoint six sequences with very
different statistical character (constant 3.0 s, jittered, exponential,
bimodal burst/pause, constant 0.5 s, ramp):

| Probe | anomaly mean | anomaly std over time | bot_probability | raw critic score |
|---|---|---|---|---|
| constant_3s | 0.624925 | 3.34e-07 | 0.0000 | -90.2509 |
| jittered_3s | 0.624925 | 3.24e-07 | 0.0000 | -90.2509 |
| exponential | 0.624924 | 3.40e-07 | 0.0000 | -90.2509 |
| bimodal_burst_pause | 0.624916 | 2.72e-07 | 0.0000 | -90.2513 |
| constant_0p5s | 0.624926 | 3.22e-07 | 0.0000 | -90.2509 |
| ramp | 0.624921 | 3.67e-07 | 0.0000 | -90.2511 |

The anomaly spread across all six probes is **9.95e-06**. The bot
probability spread is exactly **0.0**.

## 4. Finding 2: the signal dies inside the transformer

`diagnose_warden_layers.py` hooks every submodule and measures how much the
output moves between probes:

| Layer | spread of per-probe means | spread of per-probe stds | max element spread |
|---|---|---|---|
| delay_embedding | 9.89e-05 | 7.35e-04 | 4.41e-01 |
| input_projection | 2.06e-02 | 1.83e-01 | 2.80e+00 |
| bilstm | 1.13e-02 | 7.09e-02 | 9.46e-01 |
| **transformer** | **2.61e-08** | **1.19e-07** | **1.04e-04** |
| anomaly_head | 9.83e-06 | 9.38e-08 | 1.06e-05 |
| classification_head | 3.81e-04 | nan | 3.81e-04 |

The transformer is where the signal dies. Tracing it layer by layer with
the maximum difference between two maximally different inputs:

| Stage | Shipped checkpoint | Fresh (untrained) init |
|---|---|---|
| after BiLSTM | 8.83e-01 | 2.51e-01 |
| transformer layer 0 | 1.37e-01 | 3.19e+00 |
| transformer layer 1 | 1.27e-02 | 3.73e+00 |
| transformer layer 2 | 1.25e-03 | 3.29e+00 |
| transformer layer 3 | 9.93e-05 | 3.31e+00 |
| anomaly score | 1.03e-05 | 2.04e-01 |

The shipped checkpoint attenuates the input difference by roughly a factor
of ten at every transformer layer, for a total of about 1e-4. A fresh
network passes the difference through essentially unchanged. The collapse
is therefore a property of the trained weights, not of the architecture as
written.

The mechanism is visible in the per-layer statistics. The feed-forward
branch of each transformer layer produces a large, input-independent
constant that dominates the residual stream:

| Layer | attn std | ff std | ff max | residual std | after LayerNorm std |
|---|---|---|---|---|---|
| 0 | 0.502 | 0.605 | 1.912 | 1.192 | 1.004 |
| 1 | 1.742 | 3.457 | 8.730 | 4.096 | 1.013 |
| 2 | 3.692 | 4.650 | 15.623 | 5.220 | 1.009 |
| 3 | 3.345 | 3.401 | 10.522 | 3.867 | 1.028 |

The residual stream grows to a standard deviation of 5.2 while the
input-dependent component shrinks by an order of magnitude per layer. The
final `LayerNorm` then renormalizes to unit variance, so the output is
dominated by the constant and the input signal is scaled away. This is the
classic signature of a post-LN residual stack whose feed-forward branch
learned a large constant offset.

## 5. Finding 3: the anomaly head was never trained

A gradient test on the shipped architecture, using the same loss the
trainer uses (`compute_warden_loss` plus `compute_gradient_penalty`),
shows that the anomaly head receives **no gradient at all**:

```
anomaly-head parameters with NO gradient:
  ['anomaly_head.0.weight', 'anomaly_head.0.bias',
   'anomaly_head.3.weight', 'anomaly_head.3.bias',
   'anomaly_head.5.weight', 'anomaly_head.5.bias']
```

This is structural, not accidental. `compute_warden_loss` uses only
`real_verdict.feature_importance["raw_critic_score"]` and
`fake_verdict.feature_importance["raw_critic_score"]`. The anomaly head's
output reaches the loss only through `feature_importance["anomaly_scores"]`
and `feature_importance["global_attention"]`, and neither is used by
`compute_warden_loss` or by `compute_gradient_penalty`, which differentiates
`raw_critic_score`. The anomaly head is a dead branch: it is never in the
gradient path.

Its weights are therefore whatever the initializer left, and since it sits
on top of a collapsed transformer, its output is a fixed constant.

Confirming with `diagnose_warden_components.py`, which swaps the trained
and fresh components:

| Transformer | Anomaly head | anomaly mean (A / B) | anomaly diff | critic diff |
|---|---|---|---|---|
| shipped | shipped | 0.624857 / 0.624939 | 9.13e-05 | 5.65e-05 |
| shipped | fresh | 0.320012 / 0.320024 | 1.61e-05 | 5.65e-05 |
| fresh | shipped | 0.387418 / 0.216679 | **1.83e-01** | 1.56e-02 |
| fresh | fresh | 0.205096 / 0.369932 | **2.04e-01** | 1.56e-02 |

With a fresh transformer, the shipped anomaly head responds normally to
input (1.83e-01 spread). With the shipped transformer, no anomaly head can
respond. The transformer is the proximate cause; the untrained anomaly head
is why the constant is a benign-looking 0.62 rather than an obvious
divergence.

## 6. Finding 4: `bot_probability = 0.0` is a float32 underflow

The raw critic score is a constant **-90.2509**. The verdict applies
`sigmoid(raw_critic_score)`:

```
sigmoid(-90.25) in float32: 0.0
sigmoid(-90.25) in float64: 6.38e-40
```

`e^-90.25 = 6.38e-40` is below the float32 normal range
(`tiny = 1.18e-38`), so the sigmoid underflows to exactly zero. The
reported `bot_probability = 0.0` is therefore a numerical artifact of a
saturated unbounded critic output, not a calibrated statement about any
input. Any downstream consumer that reads `bot_probability` as a
probability is reading a rounding artifact.

This has a second, more serious consequence. `src/stealth/rl/environment.py`
uses `verdict.bot_probability` as the stealth penalty in the RL reward
(`warden_penalty = bot_probability * self.lambda_stealth`). With the warden
collapsed, that penalty is identically zero, so the PPO agent was trained
with **no stealth signal whatsoever**. The RL checkpoint's recorded
`warden_scores` are all exactly `0.0`, confirming this. The PPO arm of C3
is therefore a throughput-and-diversity policy, not a stealth policy.

## 7. Finding 5: the training run itself diverged

The `gan_generator.pt` checkpoint carries the training history. Its final
entries show the run had collapsed before the checkpoint was written:

| epoch | step | generator_loss | warden_loss | real_bot_prob | fake_bot_prob | gradient_penalty |
|---|---|---|---|---|---|---|
| 0 | 0 | 2.346 | 1.932 | 0.996 | 0.168 | 12.432 |
| 0 | 1 | 5.814 | -13.907 | 1.000 | 0.010 | 2.734 |
| 11 | 344 | 110.313 | 25.990 | 0.000 | 0.000 | 9.9996 |
| 11 | 345 | 111.298 | 13.438 | 0.000 | 0.000 | 9.9997 |
| 11 | 346 | 123.175 | 31.920 | 0.025 | 0.000 | 9.9996 |
| 11 | 347 | 127.851 | 10.357 | 0.000 | 0.000 | 9.9997 |

Three things stand out:

1. The generator loss grew from 2.3 to 127.9, a monotone blow-up. In a
   Wasserstein setup the generator loss is `-mean(D(fake))`, so this means
   the critic's score on fake traffic is being driven steadily more
   negative.
2. The gradient penalty saturates at exactly `lambda_gp = 10.0`, which is
   the value it takes when the critic's input gradient norm is exactly 0.
   A constant function has zero gradient, and the penalty's `(‖∇D‖ - 1)²`
   term is then `(0 - 1)² = 1`, giving the maximum penalty of 10.0. The
   critic had already become constant.
3. Both `real_bot_prob` and `fake_bot_prob` read 0.000 from early on,
   because the sigmoid underflow described in Finding 4 hides the actual
   critic behavior from the training log.

The saved checkpoint is a `state_dict` for the warden alone, written at
`epoch 11` of a 20-epoch run, with the same collapsed weights that
`gan_generator.pt` carries in its `warden_state`. The warden was frozen
from this state, and the RL agent was then trained against it.

Note also that the checkpoint's own training config records
`device='cpu'` and `checkpoint_dir='/home/karash1/projects/DCASS/...'`, so
this run was executed on a different machine from the current one.

## 8. Repairs attempted

| Candidate repair | Result |
|---|---|
| Checkpoint loading, key coverage, `strict=False` fallbacks | Clean. 85 of 89 parameters differ from a fresh initialization, so the file is a trained checkpoint, not a stale or mis-pickled one. No missing or unexpected keys. |
| Input normalization, feature extraction | Correct. `extract_statistical_features` responds properly to the probes (stat feature spread 1.34 to 21.36 across probes). |
| Threshold logic in the benchmark | Correct. The composite decision function's threshold is 0.50 and the soft branch is clipped to 0.45, exactly as documented. |
| float32 sigmoid underflow | Confirmed and characterized, but a symptom. Fixing the underflow would turn `bot_probability = 0.0` into `6.38e-40`, which is still a constant. |
| Anomaly-head gradient path | Confirmed dead. The head is structurally excluded from the loss. |
| Transformer signal collapse | Confirmed as the proximate cause. A fresh transformer restores full input sensitivity with the shipped anomaly head. |

Every one of these is a property of the checkpoint or of the training
objective, not of a threshold or a normalization step that a one-line fix
would repair.

## 9. What was done instead, and what it costs

The diagnostic stopped short of retraining the WGAN-GP critic, for three
reasons that still hold:

1. **Fixing the divergence in the WGAN-GP loop is a redesign.** The
   gradient penalty sitting at its maximum with a diverging generator loss
   is a training-dynamics problem. The layer statistics point at the
   post-LN residual stack and the feed-forward branch's constant offset,
   which suggests a pre-LN or normalized-residual architecture and a lower
   learning rate, but that carries its own validation burden.

2. **There is no real human traffic corpus.**
   `storage/data/traffic/real_human_traffic.json` is not usable as one.
   Its time-of-day distribution is uniform (chi-square p = 0.33, which real
   browsing never is), it has no delays beyond 120 s, and 4.7% of its
   values sit exactly on the 0.5 s floor with another 0.9% exactly on the
   120 s ceiling. It is a synthetic surrogate that was given a reassuring
   filename.

3. **A pre-registered analysis is required before any positive claim.**

But the diagnostic's conclusion did not have to be "nothing can be
measured". The finding was that the *shipped critic* cannot detect
anything. That is a statement about one checkpoint, not about the
benchmark's question. The question can be asked of a detector that
actually works.

### 9.1 The replacement detector

`scripts/analysis/train_traffic_detector.py` trains
`storage/models/traffic_detector.pt`: the same BiLSTM-plus-Transformer
architecture, but as a supervised binary classifier rather than a WGAN-GP
critic. The objective is `compute_detector_loss_bce` in
`src/analysis/adversarial/warden.py`, which supervises both heads:

- a sequence-level binary cross-entropy on `bot_probability` against the
  bot/human label, which trains the classification head into a calibrated
  probability;
- an auxiliary masked binary cross-entropy on the per-timestamp
  `anomaly_scores` against the sequence label broadcast across the valid
  steps, so the per-timestamp head is no longer a dead branch. Its mean
  over the valid steps is a sequence-level anomaly aggregate.

The classes are the repository's own traffic corpus and synthesized bot
sessions derived from it: a near-fixed cadence, a bursty delivery pattern,
and a human session with its spread compressed into a narrow band, each
interpolated toward the human session by a strength parameter so that the
task is not trivially separable.

### 9.2 Three defects found while building it

- **The split leaked.** The first implementation split the combined human
  and bot set, so a bot session synthesized from a held-out human session
  landed in the training set. 214 of the 300 "held-out" human sessions
  had a perturbed twin in training. The split is now over the human
  sessions, and each subset's bots are synthesized only from that subset.
- **The benchmark fed the detector the wrong shape.** The detector was
  trained on sequences padded to 64 steps with a validity mask, and its
  transformer excludes the padded steps from attention. The benchmark
  passed unpadded sequences, so the score for a given traffic depended on
  how the trial happened to be batched. Trials are now padded and masked
  the same way the detector was trained.
- **The threshold calibration overshot.** The rule reacted to a quantile
  landing slightly above the target by pushing the threshold above the
  human maximum. On the Component 3 calibration population before the fix the 95th
  percentile was 0.16 and the maximum was 0.83, so the fallback discarded
  all the resolution between them and drove the measured false-positive
  rate to 0% rather than the 5% that was asked for. The rule is now to
  place the threshold just above the score at the rank the false-positive
  budget allows, and it lives in `scripts/analysis/detector_scoring.py` so
  the trainer and the benchmark cannot disagree about it.

### 9.3 What the replacement detector achieves, and what it does not

On a session-disjoint held-out split the detector reaches a bot-probability
AUC of 0.9983. On human traffic from a generator it has never seen in any
form, its bot probability is 0.004 to 0.006; on the DCASS generators it
takes per-population means of 0.5086 (temporal WGAN, n = 20) and 0.4809
(hybrid engine, n = 20), with individual scores up to 0.9996 — an earlier
revision of this section recorded a constant 1.000 on the DCASS
generators, which does not match the artifact it cites. The
per-population scores are in `traffic_detector_generalization.json`.

That is a real separation, and it is what the C3 benchmark now measures.
But the human side of that separation is still the repository's own
synthetic corpus. The measurement is therefore **separability from that
corpus**, not detectability to a real network observer, and the plan's
upgrade condition for a positive evasion claim is still not met: there is
no real human or background traffic baseline, and there is no
pre-registered analysis.

The upgrade condition is unchanged in substance and unchanged in what it
would take to satisfy it. A real traffic capture is a data problem, not a
code problem, and it is the single thing that would move C3 from "the
traffic is detectable by a detector trained on our own corpus" to a
statement about the real world.

## 10. Consequence for the C3 claim

The C3 result stands as a **negative/limited result**, and its direction
is now unambiguous: the DCASS traffic is **detected**, not evading.

Concretely:

- The earlier `0/50` interception figure for the WGAN and Hybrid
  schedulers is **not evidence of evasion** and is withdrawn. It was the
  soft branch's arithmetic: the branch was `clip(0.10 * anomaly + 0.35 *
  ks, 0, 0.45)` against a 0.50 threshold, so its maximum was 0.45 and it
  would have read `0/50` for any smooth, non-bursting, multi-channel
  traffic regardless of how human-like it was.
- With a detector that works, the WGAN and Hybrid rows record 100%
  interception against a threshold of 0.0181, calibrated on 457 held-out
  human bursts at a measured 4.60% false-positive rate. That is a
  measurement, and it says the traffic is distinguishable. The PPO arm,
  retrained against a working detector and measured by a detector held out
  from its own training, records 0%; see
  [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md).
- The recorded anomaly score of 0.62493 is a constant of a collapsed
  model and carries no information.
- The recorded `bot_probability = 0.0` is a float32 underflow, not a
  measurement.
- The PPO arm was trained with a zero stealth penalty, so it was a
  throughput-and-diversity policy, and its interception rate was driven
  entirely by its own delay regularity rather than by any response to the
  warden. It has since been retrained against a working detector; the
  archived predecessor is kept in
  `storage/benchmarks/archive/2026-10-21-pre-remediation/rl_agent.pt`.
- The human baseline is the benchmark's own synthetic corpus. There is no
  real human or background traffic in this comparison.

The plan's upgrade condition for a positive evasion claim is a functional,
calibrated warden **plus** a real human/background traffic baseline **plus**
a pre-registered analysis. The first precondition is now met; the second
and third are not. So no positive evasion claim is made, and the measured
detection result does not become one.

## 11. What this changes in the documentation

The following rows in `CLAIM_LEDGER.md` carry the root cause and the
remediation:

- `C3-WARDEN-02` carries the transformer collapse, the dead anomaly head,
  and the float32 underflow as the explanation, and notes that the
  benchmark no longer uses that checkpoint.
- `C3-CAP-01` is retracted: the 0/50 figure was arithmetic, and the
  current artifact records 100% for both rows.
- `C3-STRUCT-03` records that the anomaly head and the RL environment's
  use of an underflowed `bot_probability` are structural defects in the
  training objective, not benchmark findings.
- `C3-PPO-05` records that the PPO arm trained against a zero stealth
  penalty.
- `C3-DETECT-06` is new. It records the replacement detector, the
  calibrated threshold and its measured false-positive rate, the
  held-out AUC, the new interception rates, and the scope limit that the
  human side of the comparison is a synthetic corpus.

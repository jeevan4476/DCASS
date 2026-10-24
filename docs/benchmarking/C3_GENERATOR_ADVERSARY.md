# Component 3: The Generator's Adversarial Signal Was Never Real

**Date:** 2026-10-21
**Status:** the three defects are diagnosed, repaired, and measured. The
repaired generator is the one the Component 3 benchmark measures.

## 1. The summary

The RL arm was repaired in
[`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md).
The generator had the same disease, for the same reason, and it turned out
to have two more defects underneath.

The generator's adversarial signal is the WGAN-GP critic. The critic
collapses during training, and it collapses in a way that makes its own
constraint inert. The generator therefore optimises against an adversary
that is not discriminating anything, and the WGAN's delays come out as a
near-constant clock.

Underneath that, two defects make the generator's situation worse than the
critic's. The critic is fed the generator's **raw** output while the real
traffic is in **seconds**, so it never sees the traffic the scheduler
sends. And the delay mapping that converts one into the other is so badly
conditioned that a freshly initialised generator receives **no gradient at
all** through it.

## 2. The data is not the cause

The shipped generator was trained against
`data/human_traffic.json`, which is not in the repository, so
`HumanTrafficDataset` fell back to a synthetic corpus: exponential delays
with a per-session rate, uniformly random channels per step, and a time of
day drawn independently of the delays. The detector, by contrast, was
trained on the traffic corpus in `storage/data/traffic/`, which has a
different structure. It was reasonable to suspect the data.

It is not the data. The same WGAN-GP training was run against both
training sets, six epochs each, recording the losses, the gradient
penalty, the generator's output spread, and the critic's sensitivity to
its input (`wgan_training_data_probe.json`):

| training set | generator loss | generator delay spread (`generator_delay_std`) | critic output spread (`critic_anomaly_spread`) |
| :--- | ---: | ---: | ---: |
| synthetic fallback | 83 → 418 | 1.03 → 6.21 | 3.3e-02 → 2.3e-02 |
| traffic corpus | 239 → 130 | 1.93 → 5.62 | 3.2e-02 → 2.7e-02 |

The two arms behave the same way. The corpus arm is the more stable of the
two, but neither reaches an equilibrium, and neither critic's sensitivity
changes materially.

## 3. The real trainer diverges on the corpus too

Running the shipped trainer for 27 epochs against the traffic corpus
reproduces the shipped checkpoint's behaviour exactly:

| epoch | generator loss | warden loss | gradient penalty | real bot prob | fake bot prob |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | −156.2 | −13.4 | 4.90 | 1.000 | 1.000 |
| 4 | 133.1 | −27.5 | 3.46 | 0.000 | 0.115 |
| 8 | −200.1 | −21.0 | **9.99** | 0.999 | 1.000 |
| 12 | −577.6 | 56.4 | **9.98** | 1.000 | 1.000 |
| 16 | −493.7 | 11.8 | **9.99** | 1.000 | 1.000 |
| 20 | −570.8 | −32.7 | **9.97** | 1.000 | 1.000 |
| 26 | −537.8 | −69.7 | **9.98** | 1.000 | 1.000 |

From epoch 8 the gradient penalty sits at exactly `lambda_gp`, and the
critic's sigmoid output is saturated at 1.000 on real data and on
generated data alike.

This 27-epoch trace and the loss figures quoted from the same run below
(the `−132` by-epoch-7 spike in §7, the `−577.6` at-epoch-12 and the
`−47` to `+13` envelopes in §7–8) are a narrative diagnostic: the
divergence run was not retained as a benchmark artifact and is not
re-executed by the repository's current scripts. The artifact-backed
record of the same collapse is the epoch-by-epoch tracking cited in §8
(`critic_sensitivity_final_run.json`, `critic_sensitivity_repaired_run.json`,
and the four `critic_sensitivity_*` probes).

## 4. Why the penalty is inert

The gradient penalty is `lambda_gp * (||grad_x D|| - 1)^2`. It equals
`lambda_gp` exactly when the critic's gradient with respect to its input
is zero, so a penalty sitting at `lambda_gp` is not a constraint that is
merely violated. It is a constraint that has stopped applying.

Measuring the critic's input gradient directly
(`critic_input_gradient_probe.json`) confirms the mechanism:

| critic | ‖grad_x D‖ | penalty |
| :--- | ---: | ---: |
| fresh, shipped post-LN stack | 0.1864 | 6.62 |
| fresh, pre-LN residuals | 0.1518 | 7.19 |
| the shipped, trained checkpoint | **0.000013** | **9.999** |

The critic's input gradient is crushed by three orders of magnitude during
training. The penalty asks for a gradient of one, is offered a gradient
of 1.3e-5, and the gap is not closed.

**This also rules out the architectural hypothesis the diagnostic
proposed.** The diagnostic suggested a pre-LN or normalised-residual
stack. Measured directly, pre-LN gives a *smaller* input gradient than
the shipped post-LN stack, 0.1518 against 0.1864. The architecture is
not what crushes the gradient, and the pre-LN redesign it pointed at
would not have fixed the training.

## 5. The critic never saw the traffic the scheduler sends

This is the second defect, and it is not a tuning problem.

The trainer fed the critic two different things:

| side | what the critic was given |
| :--- | :--- |
| real | corpus delays, in **seconds**, 0.5 to 60.0 |
| fake | the generator's **raw** output, 0.500069 to 0.539126 on the corpus, never mapped |

The generator's delays are turned into seconds by the canonical mapping
`(raw - 0.5) * 1500 + 1.5`, clipped to `[0.5, 30]`, and the scheduler
deploys the mapped value. The critic was trained on the unmapped value.
The two are in different units and different ranges.

So the critic never had a timing shape to learn. It could separate real
from generated traffic on the range alone, which is a task with an
uninformative solution, and it did. The generator, in turn, learned to put
its raw output just above the corpus's lower bound, which is what the
shipped checkpoint does: its raw output has a median of 0.50118 and a
standard deviation of 0.00172, and its mapped delays run from 1.60 to
30.0 seconds with a median of 3.28
(`generator_path_defects.json:unit_mismatch`).

The gradient penalty interpolated between the two as well, so it was
enforcing a Lipschitz constraint on a function of two differently scaled
quantities.

## 6. The delay mapping gave a fresh generator no gradient at all

Underneath the unit mismatch is a conditioning problem in the mapping
itself. The generator's delay head ends in a Softplus, so its raw output
is `softplus(x) + 0.5` and is unbounded above. The usable delay range,
1.5 to 30 seconds, corresponds to

    raw in [0.50000, 0.51900]

which is **1.90 percent** of the Softplus output's range, and everything
outside it is clipped. Measured on a freshly initialised generator:

| quantity | value |
| :--- | ---: |
| raw output, median | 1.032 |
| mapped delays at the 30 s ceiling | 100.0% |
| mapped delays in the usable band | 0.0% |
| **gradient d(mapped)/d(raw) that is zero** | **100.0%** |

A fresh generator's every output was at the upper clip, so the gradient
through the mapping was identically zero and the generator could not be
trained against anything at all.

## 7. The mechanism: the score's offset is free, and it runs away

The critic's loss is `E[D(fake)] - E[D(real)]`. A constant added to every
score **cancels out of it**. The offset is therefore free for the critic:
the critic's own loss has no gradient with respect to it.

The generator's loss is `-E[D(fake)]`, and the offset does *not* cancel out
of that. The generator's optimisation pushes the critic's score up, the
critic has no force resisting the offset, and the offset runs away.

Once the offset dominates, the critic's output is a large constant plus a
small input-dependent part. The input gradient collapses, the gradient
penalty becomes inert, and the training is dead. That is the whole
mechanism, and it is visible in the unbounded run: the generator's loss
reaches −132 by epoch 7, which is the critic's score at +132, while the
critic's own loss is a mere 6.3 and both of its bot probabilities are
saturated at 1.000.

**A bound on the score does not fix this, and was measured doing harm.**
A saturating transform such as `B * tanh(x / B)` kills the gradient
penalty's own gradient exactly when it is needed, because the penalty is
computed on the bounded score, and the bound also forces the critic's
working range into saturation. Re-measured over eight epochs each
against the traffic corpus, with the offset penalty of the final design
in place: a bound of ten has the critic over-Lipschitz for one epoch
(input gradient 5.19) and 0.03 by epoch 2, then dead — from epoch 3
through epoch 6 it scores real and generated traffic identically (score
gap 0.017 or less, input gradient 0.003 or less), and its final gap has
the generated traffic above the corpus; a bound of a hundred lets the
critic swing across a 38-point range for three epochs, then holds the
same dead state from epoch 4 through the final epoch, and the run ends
collapsed (`storage/benchmarks/critic_sensitivity_bound10.json`,
`storage/benchmarks/critic_sensitivity_bound100.json`).

**The offset is anchored instead, by a penalty on its square.** The
critic's score stays unbounded, its gradient stays live, and the gap,
which is the part the gradient penalty is there to constrain, is left
alone. Measured over ten epochs against the traffic corpus, the offset
penalty at 0.1 holds the generator's loss between −47 and +13, where the
unbounded run reached −132 by epoch 7 and the bounded runs saturated.

## 8. The repair

Three changes, in the order they have to be made.

1. **The critic is given the mapped delays**, in the same units as the
   real traffic, through a differentiable version of the canonical
   mapping so the gradient reaches the generator's delay head.
2. **The delay head's output is folded into the band of raw values the
   mapping can use.** The mapping is then never clipped, the gradient is
   live everywhere, and the deployed delay range is the full 0.5 to 30
   seconds. The contract lives in one place, `src/stealth/delay_contract`,
   which the generator and the scheduler both read.
3. **The critic's score is left unbounded and its offset is anchored**, by
   a penalty on the offset's square.

Measured on the repaired path (`generator_path_repair.json`):

| quantity | before | after |
| :--- | ---: | ---: |
| fresh generator's delays at the mapping's clip | 100.0% | 0.0% |
| gradient through the mapping that is zero | 100.0% | 0.0% |
| generator's loss over thirty epochs (narrative run, not retained — see §3) | −577.6 at epoch 12 | −6.0 to 45.3 |
| gradient penalty during training (narrative run, not retained — see §3) | pinned at 10.00 | 1.0 to 47.2 |
| trained critic's ‖grad_x D‖, on the corpus | 0.000013 | 0.5199 |
| trained critic's score gap, real against generated | 0.000 | 11.3146 |
| generator's mapped delays | median 3.28 s | median 3.43 s, p10 2.74 s, p90 5.24 s, max 29.30 s |
| corpus's delays, for comparison | — | median 5.49 s, p90 24.29 s, max 60.0 s |

**The gradient penalty was too weak, and that was the last of it.**
Tracking the first repaired run's critic against the corpus it trained
on (`critic_sensitivity_final_run.json`,
`scripts/analysis/track_critic_sensitivity.py`) showed the critic
entering a collapsed state in 12 of its 30 epochs, in which its gradient
with respect to its input is about 1e-4 and the gap between its score on
real traffic and on the generator's own traffic is 0.000:

| epoch | ‖grad_x D‖ | penalty | score on real | score on the generator's traffic | gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2.1929 | 14.23 | +11.415 | −33.835 | 45.250 |
| 11 | 2.0439 | 10.90 | −5.270 | −22.449 | 17.179 |
| 17 | 0.0012 | 9.98 | −2.656 | −2.663 | 0.006 |
| 21 | 0.0116 | 9.77 | −4.863 | −4.942 | 0.079 |
| 24 | 0.5129 | 2.37 | −20.735 | −35.992 | 15.257 |
| 29 | 0.0001 | 10.00 | −5.418 | −5.419 | 0.000 |

The run ended in that state, so the checkpoint it produced had a critic
that was a constant function.

The cause is the *weight* of the constraint rather than its form. The
penalty is `lambda_gp * (||grad_x D|| - 1)^2`, and at `lambda_gp = 10`
its maximum value of 10 is the same order as the Wasserstein term it is
supposed to restrain. The critic can therefore afford to stop reading
its input: the price of doing so is no larger than the price of doing
its job. Raising `lambda_gp` to 100 makes the constraint dominant, and
the run is then stable in every epoch:

| run | `lambda_gp` | collapsed epochs | ‖grad_x D‖ | score gap |
| :--- | ---: | ---: | ---: | ---: |
| the shipped checkpoint | 10 | collapsed state (point probe) | 0.000013 | 0.000 |
| first repaired run | 10 | 12 of 30 | 4.5e-05 to 2.39 | −5.544 to 45.2 |
| **the shipped run** | **100** | **0 of 30** | **0.11 to 0.78** | **0.33 to 19.40** |
| ten-epoch probe | 1000 | 0 of 10 | 0.42 to 0.92 | −3.0 to 2.1 |

The shipped checkpoint was not tracked epoch by epoch; its row quotes the
direct probe of its trained state (`critic_input_gradient_probe.json`,
with the score constantly 0.62493 across every probe in
`warden_diagnostic.json`).

An earlier version of this document described a `lambda_gp` of 1000 as
over-sensitive (its "input gradient settles near 9" claim), which was a
mis-derivation of the penalty identity from the penalty value; the
ten-epoch probe supersedes that description. The shipped weight stays
100 because it is the smallest measured weight that holds for a full
thirty-epoch run, not because 1000 was measured failing
(`critic_sensitivity_lambda1000.json`).

The final epoch's critic has an input gradient of 0.52 and a score gap
of 11.1 between real traffic and the generator's own traffic, so the
generator's delays are now the product of an adversary that was
discriminating (`critic_sensitivity_repaired_run.json`).

**What the working critic produced.** The generator's delays now sit on
the corpus's mode and not on its tail: a 10th percentile of 2.74
seconds, a median of 3.43, and a 90th percentile of 5.24, against a
corpus whose median is 5.49 and whose 90th percentile is 24.29. A
critic that discriminates properly drives the generator toward the
densest part of the human distribution, which is where the Wasserstein
distance is cheapest to reduce, and that is what the training does. The
traffic is still flagged at 100 percent, with a KS distance to the
surrogate of 0.5595 that is worse than Poisson's 0.2211.

**What was tried and did not work**, so it is not tried again:

1. **A bound on the critic's score.** Measured over eight epochs each,
   with the offset penalty of the final design in place. A bound of ten
   keeps the critic sensitive for one epoch (input gradient 5.19) and
   then it scores real and generated traffic identically from epoch 3
   to epoch 6, with a gradient of 0.003 or less; a bound of a hundred
   swings for three epochs and then holds the same dead state from
   epoch 4 through the end. A saturating transform kills the gradient
   penalty's own gradient exactly when it is needed
   (`critic_sensitivity_bound10.json`,
   `critic_sensitivity_bound100.json`).
2. **A stronger critic.** Eight critic steps per generator step, at the
   old `lambda_gp` of 10, did not produce a better generator. Across its
   thirty epochs the critic's input gradient never exceeds 0.69 (median
   0.37), and its score gap — real minus fake — is negative in 22 of the
   30 sampled epochs, swinging from −16.02 to +16.69: the extra steps
   still leave the adversary scoring the generated traffic above the
   corpus most of the time and overreaching when it does not
   (`storage/benchmarks/critic_sensitivity_stronger_critic.json`). The
   artifact tracks the critic's scores, not an accompanying delay series;
   the delay-median sweep of 0.88 to 10.70 seconds quoted in earlier
   revisions was withdrawn as unmeasured.

## 9. What this means for the recorded C3 result

The recorded result does not change in kind. The Component 3 measurement
is a timing warden with a calibrated threshold, measured against held-out
human traffic. The WGAN and Hybrid schedules are still machine-like
traffic, and the measurement still says so.

What changes is the traffic, and not the verdict. The WGAN and Hybrid
rows used to record a near-constant clock at 1.60 to 30 seconds with a
raw output spread of 0.0017, produced by a generator that had never been
given a usable gradient and had never faced a critic that could read
its input. They now record a generator trained against a critic that
discriminates in every one of its thirty epochs, whose delays sit on
the corpus's mode, and whose bot probabilities are a spread rather than
a point mass. The interception rate is still 100 percent and the KS
distance is still worse than Poisson's.

That is a *narrower* negative, and it is narrower in the right
direction. The traffic is no longer an artifact of a broken pipeline, so
the result is about the method rather than about a bug. The remaining
distance between the generator and the human corpus is a real distance:
a working critic drives the generator toward the densest part of the
human distribution, and the tail of the human delay distribution is
exactly what a Wasserstein-trained generator does not reproduce.

The scope limit already attached to the C3 numbers, that the human side
is the repository's own synthetic corpus and the detector is a narrow
timing model, is unchanged and is the operative limit. A real capture
would settle it; see
[`REAL_TRAFFIC_CAPTURE.md`](REAL_TRAFFIC_CAPTURE.md).

## 10. Artifacts

| artifact | what it records |
| :--- | :--- |
| `storage/benchmarks/wgan_training_data_probe.json` | the same WGAN-GP training against both training sets |
| `storage/benchmarks/critic_input_gradient_probe.json` | the critic's input gradient, which the penalty constrains |
| `storage/benchmarks/generator_path_defects.json` | the unit mismatch and the mapping's conditioning, before the repair |
| `storage/benchmarks/generator_path_repair.json` | the same quantities, after the repair |
| `storage/benchmarks/critic_sensitivity_final_run.json` | the first repaired run's critic, epoch by epoch: the input gradient, the penalty, and the score gap |
| `storage/benchmarks/critic_sensitivity_repaired_run.json` | the shipped run's critic, epoch by epoch: 0 collapsed epochs in 30 |
| `storage/benchmarks/generator_detector_adversary_probe.json` | the generator trained against a bounded objective, measured with a held-out detector |
| `scripts/analysis/probe_wgan_training_data.py` | reproduces the two-arm comparison |
| `scripts/analysis/probe_critic_input_gradient.py` | reproduces the gradient measurement |
| `scripts/analysis/measure_generator_path_defects.py` | reproduces the defect measurements |
| `scripts/analysis/verify_generator_path.py` | reproduces the repair verification |
| `scripts/analysis/track_critic_sensitivity.py` | tracks a run's critic, epoch by epoch |
| `scripts/analysis/probe_generator_detector_adversary.py` | reproduces the bounded-objective training |
| `tests/test_stealth/test_delay_contract.py` | pins the repaired contract |

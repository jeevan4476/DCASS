# Real Human Traffic Capture: Format, Ingestion, and Re-run

**Status:** the ingestion path is built and tested. The missing input is the
capture itself.

## Why this exists

Every claim Component 3 makes is a claim about the difference between
DCASS traffic and *human* traffic. The human side of that comparison is
the traffic corpus in `storage/data/traffic/`, and that corpus is
synthetic. It was measured, not assumed:

| property | the shipped corpus | real human traffic |
| :--- | :--- | :--- |
| time-of-day distribution | uniform (chi-square p = 0.33 against uniform) | concentrated on waking hours |
| maximum delay | 120.0 s, exactly | unbounded |
| fraction of delays at the 0.5 s floor | 4.7% | varies |
| same-channel reuse gap | median 5.5 s | varies |

A uniform time-of-day distribution is not something real browsing produces.
The corpus is good enough to build and test the pipeline against, and not
good enough to support a claim about real human behaviour.

Until a real capture is supplied, the Component 3 result is bounded
accordingly: it measures separability from *this* corpus, and the scope
limit travels with every number. The measurement machinery does not need
to change. Only the corpus does.

## The capture format

A JSON file holding a list of sessions. Each session is a list of
transmission events in time order, and each event carries the timestamp
and the channel it went out on:

```json
[
  {"events": [
    {"t": 1712345678.12, "channel": 0},
    {"t": 1712345681.90, "channel": 1},
    {"t": 1712345689.44, "channel": 1}
  ]},
  {"events": [ ... ]}
]
```

- `t` is a Unix timestamp in **seconds**. Sub-second precision is fine and
  is preserved.
- `channel` is an integer channel index. The pipeline uses three
  channels; map your capture's channels onto 0, 1, 2. If the capture has
  more, pick the three busiest and record the mapping in the capture
  report.

A flat list of events is also accepted, in which case sessions are split
on a gap of `--session-gap` seconds (900 by default).

The expected sources are a proxy log, a browser extension, or a platform
API log. What matters is that the timestamps are **real inter-arrival
times of a person's own activity**, and that the channel labels mean
something like "which platform or egress path", because the detector uses
the channel structure.

## Ingesting it

```bash
.venv/bin/python scripts/analysis/ingest_real_traffic.py \
    --capture my_capture.json \
    --out storage/data/traffic_real/human_traffic.json
```

The ingester derives the inter-arrival delays, the hour of day, and the
session boundaries, and writes the corpus in the format the detector
trainer and the benchmark already read. It also writes
`human_traffic.provenance.json` beside it, reporting:

- the delay distribution (mean, percentiles, maximum, the fraction below
  1.5 s and above 120 s);
- the time-of-day distribution, with a chi-square statistic against
  uniform, so a capture with a flat hour distribution is visible;
- the same-channel reuse gap and the channel run lengths, because the
  detector reads the channel structure.

**Read that report before using the corpus.** A capture that fails these
tests is no better than the synthetic one for the purposes of the claim,
and the report says so rather than leaving it to be discovered later.

## Re-running against it

The corpus directory is resolved from the environment, so nothing else has
to change:

```bash
export DCASS_TRAFFIC_DIR=storage/data/traffic_real

# 1. Retrain the detector on the real corpus. This is the adversary the
#    RL agent optimises against.
.venv/bin/python scripts/analysis/train_traffic_detector.py

# 2. Train the held-out evaluator, on a different split and a different
#    seed, so Component 3 can measure with a detector the agent never
#    saw.
.venv/bin/python scripts/analysis/train_traffic_detector.py \
    --seed 1337 \
    --model-out storage/models/traffic_detector_eval.pt \
    --results-out storage/benchmarks/traffic_detector_eval_results.json

# 3. Retrain the RL agent against the new adversary.
.venv/bin/python scripts/stealth/train_rl.py \
    --warden-checkpoint storage/models/traffic_detector.pt \
    --episodes 800 --out storage/models/rl_agent.pt

# 4. Re-measure Component 3.
.venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py
.venv/bin/python scripts/analysis/benchmark_3b_warden_score_distributions.py
```

`DCASS_TRAFFIC_DIR` is read by the detector trainer, the Component 3
benchmark, and the measurement scripts. The resolution lives in
`scripts/analysis/traffic_corpus.py`.

## What would change, and what would not

The **conclusion** would not change: C3 is a negative/limited result and
no positive evasion claim is made either way. What a real capture buys is
the *scope* of the claim. Right now the result says the DCASS schedules
are distinguishable from a synthetic corpus. With a real capture it
would say they are distinguishable from human traffic, which is the only
form of the claim that means anything to a reader.

The other thing a real capture would settle is whether the *corpus-shaped*
structure of the DCASS traffic is actually a weakness. The detector's
human class is a corpus shape, and the agent found a way to look
corpus-shaped while respecting the platform's cooldowns. Against a real
capture, that finding either holds or it does not, and the current
measurement cannot tell.

## What is still missing either way

A capture is necessary and not sufficient. The plan's upgrade condition
also asks for a **pre-registered analysis**: the decision rule, the
threshold, and the comparison fixed before the capture is looked at.
Without that, a capture can still be mined for a favourable number, and
the resulting claim would be worth no more than the one it replaces.

The generator's adversarial signal is a separate open item, tracked in
`C3_TRAFFIC_DETECTOR_EVALUATION.md` §6. It does not depend on the
capture.

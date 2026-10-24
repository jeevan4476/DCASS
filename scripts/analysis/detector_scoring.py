#!/usr/bin/env python3
"""
Shared scoring helpers for the Component 3 traffic detector.

The detector's flag threshold has to be set on human traffic, and the rule
for setting it has to be the same everywhere it is used: in the detector's
own training script, in the Component 3 benchmark, and in the measurement
scripts. A threshold that is set one way in the trainer and another way in
the benchmark is a threshold whose reported false-positive rate does not
mean anything.

`select_threshold` implements the rule: given the scores a detector
produced on human traffic, return the *smallest* threshold whose
false-positive rate is at or below the target.

The subtlety is that a quantile is not that threshold. The
`target_fpr` quantile of the human scores has, by construction, about
`target_fpr` of the sample at or above it, and with a small sample the
interpolated quantile can leave slightly *more* than the target fraction
at or above it. A rule that reacts by pushing the threshold above the
human maximum then discards all the resolution between the quantile and
the maximum: on the Component 3 calibration population before the fix the 95th percentile
was 0.16 and the maximum was 0.83, so a threshold of 0.83 would flag only
traffic the detector is almost certain about, and the measured
false-positive rate would be 0% rather than the 5% that was asked for.

The rule here is instead to sort the human scores and place the threshold
just above the score at the rank that the false-positive budget allows.
That is the smallest threshold that satisfies the constraint, and it
reproduces the requested rate instead of overshooting it.
"""

from __future__ import annotations

import numpy as np


def select_threshold(
    human_scores: np.ndarray, target_fpr: float = 0.05
) -> tuple[float, float]:
    """Return the smallest threshold whose false-positive rate is at target.

    Args:
        human_scores: The detector's scores on human traffic.
        target_fpr: The false-positive rate the threshold may reach.

    Returns:
        A tuple of the threshold and the false-positive rate it produces.
    """
    scores = np.asarray(human_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot calibrate a threshold without human scores")

    # The number of human samples the false-positive budget allows to be
    # flagged. `floor` because a budget of 5% over 457 samples is 22
    # samples, not 22.85.
    budget = int(np.floor(target_fpr * scores.size))
    spread = float(np.ptp(scores))
    step = max(spread, 1.0) * 1e-6

    if budget <= 0:
        # No human sample may be flagged, so the threshold goes just below
        # the lowest human score.
        threshold = float(scores.min()) - step
    else:
        # Sort descending: the `budget`-th score is the lowest one that
        # may still be flagged. Placing the threshold just above it means
        # at most `budget` samples reach it.
        ordered = np.sort(scores)[::-1]
        threshold = float(ordered[budget - 1]) + step

    measured = float((scores >= threshold).mean())
    return threshold, measured


def describe_threshold(human_scores: np.ndarray, target_fpr: float = 0.05) -> dict:
    """Select a threshold and report the distribution it was chosen from."""
    scores = np.asarray(human_scores, dtype=np.float64)
    threshold, measured = select_threshold(scores, target_fpr)
    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "measured_fpr": measured,
        "n_human": int(scores.size),
        "human_score_mean": float(scores.mean()),
        "human_score_p50": float(np.percentile(scores, 50)),
        "human_score_p95": float(np.percentile(scores, 95)),
        "human_score_max": float(scores.max()),
    }

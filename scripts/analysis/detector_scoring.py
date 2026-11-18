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


def select_threshold_by_group(
    score_groups: dict[str, np.ndarray], target_fpr: float | dict[str, float] = 0.05
) -> tuple[float, dict[str, float]]:
    """Choose the lowest threshold meeting the FPR target in every group.

    A pooled threshold can hide a high false-positive rate in a smaller
    population. Candidate thresholds are drawn from observed scores and
    checked against each group independently; a value just above the
    selected score preserves the benchmark's inclusive ``score >=
    threshold`` flag rule.
    """
    groups = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in score_groups.items()
        if np.asarray(values).size
    }
    if not groups:
        raise ValueError("cannot calibrate a threshold without group scores")
    targets = {
        name: float(target_fpr.get(name, 0.0) if isinstance(target_fpr, dict) else target_fpr)
        for name in groups
    }
    if any(not 0.0 <= value <= 1.0 for value in targets.values()):
        raise ValueError("target_fpr values must be between 0 and 1")

    all_scores = np.concatenate(list(groups.values()))
    step = max(float(np.ptp(all_scores)), 1.0) * 1e-6
    candidates = sorted(set(float(value) + step for value in all_scores))
    candidates.append(float(all_scores.max()) + 2.0 * step)

    for threshold in candidates:
        rates = {
            name: float(np.mean(scores >= threshold))
            for name, scores in groups.items()
        }
        if all(rate <= targets[name] for name, rate in rates.items()):
            return threshold, rates

    # The max-plus-step candidate must produce zero false positives. Keep a
    # defensive error if malformed numeric input ever violates that premise.
    raise RuntimeError("failed to find a threshold satisfying all group FPR limits")


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

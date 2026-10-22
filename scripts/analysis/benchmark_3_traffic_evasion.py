#!/usr/bin/env python3
"""
Component 3: Traffic Timing & Traffic Analysis Evasion Benchmark
Evaluates Deep Packet Inspection (DPI) Warden evasion, timing entropy,
Kolmogorov-Smirnov distributional drift to human traffic, latency overhead,
and diurnal circadian rhythm adaptations.

Schedulers compared:
1. Static Cadence (Fixed Metronome baseline)
2. Stochastic Poisson (Classical stochastic stego baseline)
3. DCASS Temporal WGAN (Generative temporal mimicry)
4. DCASS Hybrid Engine (WGAN timing + Closed-Loop Action-Masked Channel Routing)
5. DCASS PPO RL Agent (Optimized closed-loop policy)
Plus: diurnal conditioning check (hour 14:00 vs 03:00 under one identical delay
mapping) and a warden positive control that runs the decision function on the
surrogate human reference itself.
"""

from __future__ import annotations

import sys
import json
import pathlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import platform
from dataclasses import dataclass, asdict, field
from scipy.stats import ks_2samp

import torch
from src.utils.checkpoint_io import load_trained_checkpoint

# Compatibility shim for Python 3.12 + PyTorch unpickling
sys.modules.setdefault("pathlib._local", pathlib)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.stealth_scheduler import (
    StealthScheduler,
    canonical_delay_mapping as scheduler_canonical_delay_mapping,
)
from src.analysis.adversarial.warden import DeepPacketInspectionWarden

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traffic_corpus import corpus_files, traffic_dir  # noqa: E402
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    recorded_commit,
    scale_table_to_textwidth,
    write_manifest,
)
from detector_scoring import select_threshold  # noqa: E402

# Benchmark Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CHANNELS = 3
NUM_TRIALS = 50           # 50 independent message bursts per scheduler
CARRIERS_PER_MESSAGE = 50  # 50 carrier packets per message transmission
BASE_DELAY = 3.0          # Baseline delay parameter
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

# Enterprise DPI Channel Cooldown Timers (items/min -> min inter-arrival seconds)
CHANNEL_COOLDOWNS = {0: 6.0, 1: 12.0, 2: 4.0}

# Soft-branch scoring.
#
# The recorded branch was `clip(0.10 * anomaly + 0.35 * ks, 0, 0.45)` against
# a 0.50 flag threshold. Both terms are bounded by 1, so the branch's
# maximum was 0.45: it could never flag any traffic, for any input. The
# zero interception recorded for the WGAN and Hybrid rows was a property of
# the arithmetic, not of the traffic.
#
# The branch is now the trained detector's own bot probability, with the
# flag threshold calibrated on held-out human traffic so the
# false-positive rate is a measured quantity rather than an assumption.
#
# The KS statistic is deliberately NOT part of the score. A 50-item burst
# drawn from the human reference itself already scores about 0.12 against
# it, and a whole human session about 0.28, so the raw statistic carries
# little information about whether traffic is machine-like: adding it to a
# well-calibrated detector dilutes the detector. Measured on held-out human
# traffic, at a 5% false-positive rate:
#
#   detector only            human FPR 5.0%  DCASS WGAN 100%  Hybrid 100%
#   detector + KS excess     human FPR 5.0%  DCASS WGAN   0%  Hybrid  10%
#
# The KS distance and its p-value are still recorded per trial and still
# appear in the results table. See docs/benchmarking/C3_WARDEN_DIAGNOSTIC.md.
SOFT_FLAG_THRESHOLD = 0.30

# The detector used for the neural term. The WGAN-GP critic in
# `warden.pt` is not a detector: its transformer collapsed to a constant
# output and its anomaly head never received a gradient. The trained
# detectors are supervised classifiers with a calibrated threshold.
#
# The benchmark measures with the HELD-OUT detector, not the one the RL
# agent was trained against. `traffic_detector.pt` is the adversary: the
# RL agent optimizes its bot probability, so measuring the agent with it
# would report how well it gamed its own opponent. `traffic_detector_eval.pt`
# is an independently trained detector, on a different split and a
# different initialization, that the agent has never seen. See
# docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md.
ADVERSARY_DETECTOR_PATH = PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt"
DETECTOR_PATH = PROJECT_ROOT / "storage" / "models" / "traffic_detector_eval.pt"

# The detector's input length. The detector was trained on sequences
# padded to this length with a validity mask, and the transformer
# excludes padded steps from attention. A trial must therefore be padded
# the same way before it is scored: feeding an unpadded sequence changes
# the attention pattern and the statistical features, and the detector's
# score for the same traffic changes with it.
DETECTOR_SEQ_LEN = 64

torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class SchedulerResult:
    scheduler: str
    interception_rate: float
    evasion_rate: float
    delay_entropy: float
    ks_statistic: float
    ks_pvalue: float
    mean_latency: float
    total_duration: float
    warden_scores: List[float]
    rules: List[str]  # triggering rule per trial: periodic_cv/periodic_entropy/burst/hammer/soft
    mean_cv: float  # mean coefficient of variation across trials
    mean_log_entropy: float  # mean log-bin entropy across trials (the warden feature)
    mean_neural_anomaly: float  # mean warden neural anomaly term across trials
    cooldown_violations: int  # per-channel cooldown breaches across all trials
    ks_values: List[float]  # per-trial KS distance to the surrogate baseline
    cv_values: List[float]  # per-trial coefficient of variation
    log_entropy_values: List[float]  # per-trial log-bin entropy


@dataclass
class DiurnalResult:
    regime: str
    hour: float
    mean_raw: float  # mean generator output before the canonical delay mapping
    mean_latency: float
    std_latency: float
    delay_entropy: float
    ks_statistic: float
    interception_rate: float
    evasion_rate: float
    rules: List[str]


@dataclass
class HumanSelfTest:
    """Positive-control: run the same warden decision function on human
    traffic. A calibrated warden should not flag its own ground truth,
    except through the burst rule's small false-positive floor.

    Two populations are recorded. The surrogate is the benchmark's KS
    reference, a synthetic burst-level mixture that is not the population
    the detector was trained on, so it is a second, independent reading.
    The corpus rows are the detector's own held-out sessions, so their rate
    is the operating point's false-positive rate."""

    n_trials: int
    interception_rate: float
    evasion_rate: float
    scores: List[float]
    rules: List[str]
    corpus_n: int = 0
    corpus_interception_rate: float = float("nan")
    corpus_scores: List[float] = field(default_factory=list)
    corpus_rules: List[str] = field(default_factory=list)


@dataclass
class Component3Result:
    schedulers: List[SchedulerResult]
    human_baseline_entropy: float
    human_baseline_entropy_log: float
    human_ks_distances: Dict[str, float]
    diurnal_breakdown: List[DiurnalResult]
    human_self_test: HumanSelfTest
    soft_branch_calibration: Dict[str, float]
    detector: Dict[str, object]
    provenance: Dict[str, object]


def generate_human_baseline_delays(n: int = 5000) -> np.ndarray:
    """
    Generate human-like inter-arrival times based on empirical distributions:
    - Component 1 (80%): Exponential with mean 5s (active reading, typing, posting)
    - Component 2 (20%): Log-normal (median 2.7s, long heavy tail for distraction pauses)

    The reference is at the BURST level: the distribution of delays between
    consecutive transmissions within one message transmission.

    It used to carry a third component, 10 percent periodic session gaps at
    3600 seconds. That component is a BETWEEN-session feature, and mixing
    it into a burst-level reference was a modelling error with two
    consequences. Every 50-item burst drawn from a pool carrying 10
    percent hour-scale gaps contains one, 299 times out of 300 in
    measurement, so essentially every burst was outside the range of
    anything the traffic detector had been trained on, and the detector
    scored them at 0.97 rather than 0.02. The KS reference was also
    comparing a burst against a session mixture, so it measured the
    presence of session gaps rather than the timing fidelity of a burst.

    The reference is now burst-level, and the corpus traffic in
    `storage/data/traffic/` remains the empirical cross-check: it is
    bounded at 120 seconds, which is the burst scale.
    """
    np.random.seed(SEED)
    comp1 = np.random.exponential(scale=5.0, size=int(0.8 * n))
    comp2 = np.random.lognormal(mean=1.0, sigma=1.5, size=int(0.2 * n))

    delays = np.concatenate([comp1, comp2])
    delays = np.clip(delays, 0.05, 10000.0)
    np.random.shuffle(delays)
    return delays[:n]


def compute_delay_entropy(delays: np.ndarray, bins: int = 50) -> float:
    """Compute Shannon entropy H_T = -sum p_b log2(p_b) of delay distribution in bits.

    Descriptive statistic over linear bins spanning the data range. Kept for
    continuity with the recorded result tables; note that for heavy-tailed
    inter-arrival data most mass lands in the first bin by construction.
    """
    delays_clipped = np.clip(delays, 0.01, 10000)
    hist, _ = np.histogram(delays_clipped, bins=bins, density=False)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def compute_log_bin_entropy(delays: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy over log-spaced bins, used by the warden's periodic rule.

    Inter-arrival times span decades, from sub-second typing pauses to
    multi-second pauses. Linear bins collapse human-like mass into the
    first bin and under-report entropy, which made the warden flag its own
    human reference. Decade-scale bins measure cadence regularity on the
    scale the delays actually occupy, while a constant clock still lands in
    a single bin and scores 0 bits.
    """
    delays_clipped = np.clip(delays, 0.01, 10000)
    edges = np.logspace(np.log10(0.01), np.log10(10000.0), bins + 1)
    hist, _ = np.histogram(delays_clipped, bins=edges, density=False)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def evaluate_dpi_warden(
    delays: np.ndarray,
    channels: np.ndarray,
    warden_model: DeepPacketInspectionWarden,
    human_ref: np.ndarray
) -> Dict[str, float | bool | str]:
    """
    Evaluates transmission against Deep Packet Inspection (DPI) Warden criteria.
    Combines:
    1. Periodic Rhythm Anomaly: Low CV (< 0.20) or low log-bin Shannon entropy (< 1.0 bits)
    2. Sliding Window Burst Density: >= 6 packets within sliding 5.0s window
    3. Single-channel hammering detection
    4. Neural Anomaly Score: trained traffic detector, per-timestep head
    5. Two-Sample Kolmogorov-Smirnov Test against the human traffic baseline

    Every trial additionally reports which rule decided its score, so the
    recorded interception rate can be attributed mechanistically.
    """
    timestamps = np.cumsum(delays)

    # 1. Periodic Rhythm Analysis
    mu = float(np.mean(delays))
    sigma = float(np.std(delays))
    cv = sigma / (mu + 1e-8)
    entropy = compute_delay_entropy(delays)
    entropy_log = compute_log_bin_entropy(delays)
    is_periodic = (cv < 0.20 or entropy_log < 1.0)

    # 2. Sliding Window Burst Analysis (W = 5.0s, burst threshold = 6 packets)
    max_burst = max(np.sum((timestamps >= t) & (timestamps < t + 5.0)) for t in timestamps)
    is_burst = (max_burst >= 6)

    # 3. Channel Rate-Limit Violations
    violations = 0
    last_tx = {}
    for t, c in zip(timestamps, channels):
        ch = int(c)
        if ch in last_tx and (t - last_tx[ch]) < CHANNEL_COOLDOWNS.get(ch, 6.0):
            violations += 1
        last_tx[ch] = t

    # 4. Statistical KS Test
    ks_stat, ks_pval = ks_2samp(delays, human_ref)

    # 5. Trained traffic detector
    # The detector was trained on sequences padded to DETECTOR_SEQ_LEN with
    # a validity mask, and its transformer excludes the padded steps from
    # attention. A trial is padded the same way here, so the score for a
    # given traffic does not depend on how the trial was batched.
    n = min(len(delays), DETECTOR_SEQ_LEN)
    pad_delays = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_channels = np.zeros(DETECTOR_SEQ_LEN, dtype=np.int64)
    mask = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_delays[:n] = np.asarray(delays[:n], dtype=np.float32)
    pad_channels[:n] = np.asarray(channels[:n], dtype=np.int64)
    mask[:n] = 1.0

    d_t = torch.tensor(pad_delays).unsqueeze(0).to(DEVICE)
    c_t = torch.tensor(pad_channels).unsqueeze(0).to(DEVICE)
    m_t = torch.tensor(mask).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        verdict = warden_model(d_t, c_t, mask=m_t)
        neural_anomaly = float(verdict.anomaly_scores[0, :n].mean().item())
        bot_probability = float(verdict.bot_probability.item())

    # Composite Interception Scoring (rule attribution recorded per trial)
    if is_periodic:
        score = 0.98  # Fixed periodic metronome
        rule = "periodic_cv" if cv < 0.20 else "periodic_entropy"
    elif is_burst:
        score = min(0.95, 0.65 + 0.06 * (max_burst - 5))  # Packet flush anomaly
        rule = "burst"
    elif len(set(channels)) == 1 and np.any(delays < 4.0):
        score = 1.0  # Single-channel metronome hammering
        rule = "hammer"
    else:
        # Soft branch: the trained detector's calibrated bot probability.
        # The branch can now reach the flag threshold, and the threshold is
        # calibrated on held-out human traffic, so the false-positive rate
        # is measured rather than assumed.
        score = bot_probability
        rule = "soft"

    flagged = (score >= SOFT_FLAG_THRESHOLD) if rule == "soft" else (score >= 0.50)
    return {
        "score": score,
        "flagged": flagged,
        "rule": rule,
        "entropy": entropy,
        "entropy_log": entropy_log,
        "cv": cv,
        "neural_anomaly": neural_anomaly,
        "bot_probability": bot_probability,
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "latency": float(delays.sum()),
        "mean_delay": mu,
        "max_burst": int(max_burst),
        "violations": int(violations)
    }


def run_scheduler_benchmark_multi(
    scheduler_name: str,
    scheduler: StealthScheduler,
    mode: str,
    media_ids: List[str],
    base_delay: float,
    warden: DeepPacketInspectionWarden,
    human_delays: np.ndarray,
    n_trials: int = NUM_TRIALS
) -> Tuple[SchedulerResult, np.ndarray]:
    """Run n_trials independent transmissions and collect empirical metrics."""
    print(f"  Evaluating {scheduler_name} ({n_trials} trials, {len(media_ids)} packets/trial)...")

    flagged_count = 0
    scores = []
    violations_total = 0
    entropies = []
    ks_stats = []
    ks_pvals = []
    mean_latencies = []
    total_durations = []
    rules = []
    cvs = []
    log_entropies = []
    neural_anomalies = []
    all_delays_collected = []

    for trial in range(n_trials):
        if mode == "static":
            delays = np.full(len(media_ids), base_delay, dtype=np.float32)
            channels = np.zeros(len(media_ids), dtype=np.int64)
        elif mode == "poisson":
            delays = np.random.exponential(scale=base_delay, size=len(media_ids)).astype(np.float32)
            channels = np.random.randint(0, NUM_CHANNELS, size=len(media_ids)).astype(np.int64)
        elif mode == "gan":
            # Drive the deployed scheduler path so the measured row is the
            # shipped row. The hour is pinned per trial so the diurnal
            # comparison stays under a controlled conditioning input.
            hour = float((trial * 2) % 24)
            sched = scheduler.schedule(
                media_ids=media_ids,
                mode="gan",
                base_delay=base_delay,
                time_of_day=hour,
            )
            delays = np.array(sched["delays"], dtype=np.float32)
            channels = np.array(sched["channels"], dtype=np.int64)
        elif mode == "hybrid":
            sched = scheduler.schedule(media_ids=media_ids, mode="hybrid", base_delay=base_delay)
            delays = np.array(sched["delays"], dtype=np.float32)
            channels = np.array(sched["channels"], dtype=np.int64)
        elif mode == "rl":
            sched = scheduler.schedule(media_ids=media_ids, mode="rl", base_delay=base_delay)
            delays = np.array(sched["delays"], dtype=np.float32)
            channels = np.array(sched["channels"], dtype=np.int64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_delays_collected.extend(delays)

        eval_res = evaluate_dpi_warden(delays, channels, warden, human_delays)
        if eval_res["flagged"]:
            flagged_count += 1

        scores.append(eval_res["score"])
        violations_total += eval_res["violations"]
        entropies.append(eval_res["entropy"])
        ks_stats.append(eval_res["ks_stat"])
        ks_pvals.append(eval_res["ks_pval"])
        mean_latencies.append(eval_res["mean_delay"])
        total_durations.append(eval_res["latency"])
        rules.append(eval_res["rule"])
        cvs.append(eval_res["cv"])
        log_entropies.append(eval_res["entropy_log"])
        neural_anomalies.append(eval_res["neural_anomaly"])

    interception_rate = (flagged_count / n_trials) * 100.0
    evasion_rate = 100.0 - interception_rate

    result = SchedulerResult(
        scheduler=scheduler_name,
        interception_rate=interception_rate,
        evasion_rate=evasion_rate,
        delay_entropy=float(np.mean(entropies)),
        ks_statistic=float(np.mean(ks_stats)),
        ks_pvalue=float(np.mean(ks_pvals)),
        mean_latency=float(np.mean(mean_latencies)),
        total_duration=float(np.mean(total_durations)),
        warden_scores=scores,
        rules=rules,
        mean_cv=float(np.mean(cvs)),
        mean_log_entropy=float(np.mean(log_entropies)),
        mean_neural_anomaly=float(np.mean(neural_anomalies)),
        cooldown_violations=violations_total,
        ks_values=[float(v) for v in ks_stats],
        cv_values=[float(v) for v in cvs],
        log_entropy_values=[float(v) for v in log_entropies]
    )

    print(f"    Interception: {result.interception_rate:5.1f}% | Evasion: {result.evasion_rate:5.1f}% | "
          f"Entropy: {result.delay_entropy:5.4f} bits | KS: {result.ks_statistic:5.4f} | Mean Latency: {result.mean_latency:5.2f}s")
    print(f"    Triggering rules: {_rule_summary(rules)}")

    return result, np.array(all_delays_collected)


def canonical_delay_mapping(raw: np.ndarray) -> np.ndarray:
    """Map generator outputs to delays exactly as the deployed paths do.

    This is `src.stealth.stealth_scheduler.canonical_delay_mapping`:
    (raw - 0.5) * 1500 + 1.5, clipped to [0.5, 30.0] seconds. The deployed
    GAN path, the deployed hybrid path, and this benchmark now share that
    one definition, so the measured row and the shipped row cannot drift
    apart. Any hour-to-hour difference measured with this mapping is
    attributable to the generator's conditioning, not to analysis
    constants."""
    return scheduler_canonical_delay_mapping(raw).astype(np.float32)


def evaluate_diurnal_circadian_split(
    scheduler: StealthScheduler,
    warden: DeepPacketInspectionWarden,
    human_delays: np.ndarray,
    n_trials: int = 25
) -> Tuple[List[DiurnalResult], Dict[str, np.ndarray]]:
    """Evaluates the generator under two hour-of-day conditions under one
    identical delay mapping (see canonical_delay_mapping above).

    Earlier revisions of this script applied per-regime affine constants
    (scale 1200/offset 1.2 for 14:00 and scale 3500/offset 8.0 for 03:00),
    which manufactured a 4.1x day/night latency separation that was then
    reported as learned circadian adaptation. Those constants are removed:
    the hour is the only input that differs between the two regimes.
    """
    print("  Evaluating Diurnal Conditioning under one identical mapping...")
    diurnal_results = []
    diurnal_delays_map = {}

    regimes = [
        ("Daytime Hour (14:00)", 14.0),
        ("Night Hour (03:00)", 3.0)
    ]

    for regime_name, hour in regimes:
        delays_collected = []
        raw_means = []
        flagged_count = 0
        mean_latencies = []
        entropies = []
        ks_stats = []
        rules = []

        for trial in range(n_trials):
            # Same deployed scheduler path as the main GAN row, with the
            # hour pinned. The raw generator output comes back with the
            # schedule, so the conditioning delta can be reported in
            # generator units without re-implementing the mapping here.
            sched = scheduler.schedule(
                media_ids=[f"carrier_{i:04d}" for i in range(CARRIERS_PER_MESSAGE)],
                mode="gan",
                time_of_day=hour,
            )
            delays = np.array(sched["delays"], dtype=np.float32)
            channels = np.array(sched["channels"], dtype=np.int64)
            raw = np.array(sched["raw_delays"], dtype=np.float32)

            delays_collected.extend(delays)
            raw_means.append(float(np.mean(raw)))

            eval_res = evaluate_dpi_warden(delays, channels, warden, human_delays)
            if eval_res["flagged"]:
                flagged_count += 1
            mean_latencies.append(eval_res["mean_delay"])
            entropies.append(eval_res["entropy"])
            ks_stats.append(eval_res["ks_stat"])
            rules.append(eval_res["rule"])

        int_rate = (flagged_count / n_trials) * 100.0
        ev_rate = 100.0 - int_rate

        res = DiurnalResult(
            regime=regime_name,
            hour=hour,
            mean_raw=float(np.mean(raw_means)),
            mean_latency=float(np.mean(mean_latencies)),
            std_latency=float(np.std(delays_collected)),
            delay_entropy=float(np.mean(entropies)),
            ks_statistic=float(np.mean(ks_stats)),
            interception_rate=int_rate,
            evasion_rate=ev_rate,
            rules=rules
        )
        diurnal_results.append(res)
        diurnal_delays_map[regime_name] = np.array(delays_collected)
        print(f"    {regime_name:22s} | raw mean: {res.mean_raw:.6f} | Mean Latency: {res.mean_latency:5.2f}s "
              f"(std: {res.std_latency:5.2f}s) | Entropy: {res.delay_entropy:5.4f} bits | Evasion: {res.evasion_rate:5.1f}%")

    day, night = diurnal_results
    delta_raw = day.mean_raw - night.mean_raw
    delta_latency = day.mean_latency - night.mean_latency
    print(f"    Conditioning delta: raw {delta_raw:+.6f} ({100.0 * delta_raw:+.4f}% of the unit output range) | "
          f"mapped {delta_latency:+.2f}s of the [0.5, 30.0]s window")

    return diurnal_results, diurnal_delays_map


def collect_provenance() -> Dict[str, object]:
    """Record the code state that produced this measurement.

    The table caption reads the commit back out of the artifact, so a table
    regenerated from a recorded artifact carries the commit that made the
    measurement rather than the commit that happens to be checked out.
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except OSError:
            return ""

    status = git("status", "--porcelain")
    return {
        "retrospective": False,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": DEVICE,
        "script": "scripts/analysis/benchmark_3_traffic_evasion.py",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "seed": SEED,
    }


def load_corpus_human_sessions(limit: int = 300) -> List[dict]:
    """Load held-out human sessions from the project's traffic corpus.

    These are the sessions the detector was *not* trained on. The detector
    checkpoint records its own holdout split, so the calibration uses
    exactly the rows the detector never saw rather than an arbitrary tail
    of the corpus, which may be mostly training rows.

    The corpus is the stricter human population for the soft-branch
    calibration: the benchmark's synthetic surrogate is a different
    distribution.
    """
    sessions: List[dict] = []
    for path in corpus_files():
        payload = json.loads(path.read_text())
        sessions.extend(payload)
    if not sessions:
        # The corpus is the human side of the calibration. An empty one
        # does not fail the run, it silently shrinks the calibration
        # population and moves the threshold, which is exactly the kind
        # of quiet wrong number this component must not produce.
        raise RuntimeError(
            "no human traffic corpus found under "
            f"{traffic_dir()}. The soft branch's threshold is calibrated "
            "against it, so the measurement cannot be made without it. "
            "Set DCASS_TRAFFIC_DIR to the corpus directory, or run "
            "scripts/analysis/generate_traffic_dataset.py to build one."
        )

    holdout_indices: Optional[List[int]] = None
    if DETECTOR_PATH.exists():
        try:
            ckpt = load_trained_checkpoint(DETECTOR_PATH, map_location="cpu",)
            if isinstance(ckpt, dict):
                holdout_indices = ckpt.get("holdout_human_indices")
        except Exception:  # pragma: no cover - a missing split is not fatal
            holdout_indices = None

    if holdout_indices:
        selected = [sessions[i] for i in holdout_indices if 0 <= i < len(sessions)]
        return selected[:limit]

    # No recorded split: fall back to the tail of the corpus.
    return sessions[-limit:]


def calibrate_soft_threshold(
    warden: DeepPacketInspectionWarden,
    human_delays: np.ndarray,
    corpus_sessions: Optional[List[dict]] = None,
    n_trials: int = 200,
    target_fpr: float = 0.05,
) -> Dict[str, float]:
    """Calibrate the soft-branch flag threshold on human traffic.

    The soft branch's score is the trained detector's bot probability.
    Its threshold has to be set somewhere, and setting it by hand would
    make the false-positive rate an assumption rather than a measurement.

    The calibration population is the union of two valid burst-level human
    populations, and the threshold is the one that holds the
    false-positive rate across both:

    * whole sessions from the traffic corpus in `storage/data/traffic/`,
      using the detector's own holdout split, so these are rows the
      detector never trained on;
    * bursts drawn from the benchmark's own surrogate reference, which is
      the burst-level distribution the recorded KS term is measured
      against.

    The surrogate was previously a session-level mixture carrying
    hour-scale session gaps, and it was dropped from the calibration for
    that reason: a 50-item burst drawn from a pool with 10 percent
    hour-scale gaps contains one with near certainty, 299 times out of
    300 in measurement, so essentially every surrogate burst was outside
    the detector's training distribution and the threshold came out at
    0.97, at which the detector flags almost nothing. The reference is
    now burst-level, so it is a valid false-positive population again.

    The calibration uses its own RNG stream, so the deterministic scheduler
    runs are not perturbed.

    Returns a dict with the threshold and the measured rates.
    """
    rng = np.random.default_rng(SEED + 7)

    def burst_scores() -> List[float]:
        out = []
        for _ in range(n_trials):
            delays = human_delays[
                rng.choice(len(human_delays), CARRIERS_PER_MESSAGE, replace=False)
            ]
            channels = rng.integers(0, NUM_CHANNELS, size=CARRIERS_PER_MESSAGE)
            res = evaluate_dpi_warden(delays, channels, warden, human_delays)
            if res["rule"] == "soft":
                out.append(float(res["score"]))
        return out

    surrogate_scores = burst_scores()

    corpus_scores: List[float] = []
    if corpus_sessions:
        for session in corpus_sessions:
            delays = np.asarray(session["delays"], dtype=np.float64)[:CARRIERS_PER_MESSAGE]
            channels = np.asarray(session["channels"], dtype=np.int64)[:CARRIERS_PER_MESSAGE]
            res = evaluate_dpi_warden(delays, channels, warden, human_delays)
            if res["rule"] == "soft":
                corpus_scores.append(float(res["score"]))

    calibration_scores = np.asarray(surrogate_scores + corpus_scores)
    if calibration_scores.size == 0:
        # Every calibration burst hit a hard rule, so the soft branch never
        # fired and its threshold cannot be calibrated from this draw.
        return {
            "threshold": SOFT_FLAG_THRESHOLD,
            "n_calibration": 0,
            "target_fpr": target_fpr,
            "measured_fpr": float("nan"),
            "note": "no calibration trial reached the soft branch; "
            "the default threshold was kept",
        }

    # The smallest threshold whose false-positive rate on human traffic is
    # at or below the target. See detector_scoring.py for why this is not
    # simply a quantile.
    threshold, measured = select_threshold(calibration_scores, target_fpr)

    def rate(values: List[float]) -> float:
        if not values:
            return float("nan")
        arr = np.asarray(values)
        return float((arr >= threshold).mean())

    print(
        f"  Soft-branch threshold calibrated on {len(calibration_scores)} human bursts "
        f"({len(surrogate_scores)} surrogate, {len(corpus_scores)} corpus): "
        f"{threshold:.4f} (target FPR {target_fpr:.0%}, measured {measured:.2%})"
    )
    return {
        "threshold": threshold,
        "n_calibration": int(calibration_scores.size),
        "n_surrogate": len(surrogate_scores),
        "n_corpus": len(corpus_scores),
        "target_fpr": target_fpr,
        "measured_fpr": measured,
        "measured_fpr_surrogate": rate(surrogate_scores),
        "measured_fpr_corpus": rate(corpus_scores),
        "human_score_mean": float(calibration_scores.mean()),
        "human_score_p95": float(np.percentile(calibration_scores, 95)),
        "human_score_max": float(calibration_scores.max()),
        "calibration_populations": "burst-level surrogate reference and the "
        "detector's held-out corpus sessions",
    }


def run_human_reference_self_test(
    warden: DeepPacketInspectionWarden,
    human_delays: np.ndarray,
    n_trials: int = 50,
    corpus_sessions: Optional[List[dict]] = None,
) -> HumanSelfTest:
    """Positive control for the warden decision function.

    Two populations are scored, because they answer different questions.

    The *corpus* self-test draws whole held-out sessions from the traffic
    corpus. This is the operating point's false-positive rate, measured on
    the population the detector classifies as human and on rows it never
    trained on. It is the number the interception rates should be read
    against.

    The *surrogate* self-test draws trial-sized samples from the synthetic
    burst-level reference used as the KS term. It is a second, independent
    population, since the reference is not the corpus the detector was
    trained on. A warden that flags its own ground truth at a high rate
    cannot be used to claim that a scheduler's smooth traffic is
    'undetectable', so the rate is recorded either way. Uses a dedicated
    RNG stream so the deterministic scheduler runs above are not perturbed.
    """
    print("  Running human-reference self-test (warden positive control)...")
    rng = np.random.default_rng(SEED + 1)
    scores = []
    rules = []
    flagged_count = 0
    for _ in range(n_trials):
        delays = human_delays[rng.choice(len(human_delays), CARRIERS_PER_MESSAGE, replace=False)]
        channels = rng.integers(0, NUM_CHANNELS, size=CARRIERS_PER_MESSAGE)
        eval_res = evaluate_dpi_warden(delays, channels, warden, human_delays)
        scores.append(eval_res["score"])
        rules.append(eval_res["rule"])
        if eval_res["flagged"]:
            flagged_count += 1
    int_rate = (flagged_count / n_trials) * 100.0
    print(f"    Surrogate reference flagged: {flagged_count}/{n_trials} ({int_rate:.1f}%) | rules: "
          f"periodic_cv={rules.count('periodic_cv')}, periodic_entropy={rules.count('periodic_entropy')}, "
          f"burst={rules.count('burst')}, soft={rules.count('soft')}")

    corpus_scores: List[float] = []
    corpus_rules: List[str] = []
    corpus_flagged = 0
    if corpus_sessions:
        for session in corpus_sessions[:n_trials]:
            delays = np.asarray(session["delays"], dtype=np.float64)[:CARRIERS_PER_MESSAGE]
            channels = np.asarray(session["channels"], dtype=np.int64)[:CARRIERS_PER_MESSAGE]
            eval_res = evaluate_dpi_warden(delays, channels, warden, human_delays)
            corpus_scores.append(eval_res["score"])
            corpus_rules.append(eval_res["rule"])
            if eval_res["flagged"]:
                corpus_flagged += 1
        n_corpus = len(corpus_scores)
        corpus_rate = (corpus_flagged / n_corpus) * 100.0
        print(f"    Held-out corpus sessions flagged: {corpus_flagged}/{n_corpus} "
              f"({corpus_rate:.1f}%) | rules: "
              f"burst={corpus_rules.count('burst')}, soft={corpus_rules.count('soft')}")
    else:
        n_corpus = 0
        corpus_rate = float("nan")

    return HumanSelfTest(
        n_trials=n_trials,
        interception_rate=int_rate,
        evasion_rate=100.0 - int_rate,
        scores=scores,
        rules=rules,
        corpus_n=n_corpus,
        corpus_interception_rate=corpus_rate,
        corpus_scores=corpus_scores,
        corpus_rules=corpus_rules,
    )


def create_cdf_plot(
    results_map: Dict[str, np.ndarray],
    human_delays: np.ndarray,
    output_path: Path
):
    """Create empirical CDF plot comparing actual delay distributions without synthetic mocks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Static Cadence": "#E85D5D",
        "Stochastic Poisson": "#F39C12",
        "DCASS Temporal WGAN": "#4A90D9",
        "DCASS Hybrid Engine": "#9B59B6",
        "DCASS PPO RL Agent": "#27AE60",
    }

    # 1. Human Baseline empirical CDF (synthetic surrogate, see docstring of
    #    generate_human_baseline_delays: 80% Exp(5s), 20% log-normal; the
    #    x-range must cover the log-normal tail or the curve plateaus below
    #    1.0 for purely visual reasons)
    x_human = np.sort(human_delays)
    y_human = np.arange(1, len(x_human) + 1) / len(x_human)
    ax.plot(x_human, y_human, "k--", linewidth=2.2, label="Human Surrogate Baseline", alpha=0.85)

    # 2. Schedulers CDF from actual recorded delay arrays
    for name, delays in results_map.items():
        color = colors.get(name, "#888888")
        x = np.sort(delays)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linewidth=2.0, label=name, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel(r"Inter-Arrival Time (seconds, log scale)", fontsize=12)
    ax.set_ylabel(r"Cumulative Distribution Function (CDF)", fontsize=12)
    ax.set_title(r"Transmission Cadence: Inter-Arrival Time CDF vs Human Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10.0, framealpha=0.92, loc="lower right")
    ax.grid(True, alpha=0.35, which="both")
    ax.set_xlim(0.05, 1.0e4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_circadian_plot(
    diurnal_map: Dict[str, np.ndarray],
    human_delays: np.ndarray,
    output_path: Path
):
    """Create dedicated Daytime vs Nighttime circadian modulation CDF plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Human Baseline
    x_human = np.sort(human_delays)
    y_human = np.arange(1, len(x_human) + 1) / len(x_human)
    ax.plot(x_human, y_human, "k--", linewidth=2.0, label="Human Surrogate Baseline", alpha=0.8)

    colors = {
        "Daytime Hour (14:00)": "#E67E22",
        "Night Hour (03:00)": "#2C3E50"
    }

    for name, delays in diurnal_map.items():
        x = np.sort(delays)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=colors.get(name, "#3498DB"), linewidth=2.2, label=name, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel(r"Inter-Arrival Time (seconds, log scale)", fontsize=11.5)
    ax.set_ylabel(r"Cumulative Distribution Function (CDF)", fontsize=11.5)
    ax.set_title(r"Circadian Conditioning Check: Hour 14:00 vs Hour 03:00, Identical Delay Mapping", fontsize=12.0, fontweight="bold")
    ax.legend(fontsize=10.5, framealpha=0.92, loc="lower right")
    ax.grid(True, alpha=0.35, which="both")
    ax.set_xlim(0.1, 200.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_bar_chart(results: List[SchedulerResult], output_path: Path):
    """Create bar chart of Interception and Evasion rates with proper raw strings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    names = [r.scheduler for r in results]
    colors = ["#E85D5D", "#F39C12", "#4A90D9", "#9B59B6", "#27AE60"]

    # 1. Interception Rate
    rates = [r.interception_rate for r in results]
    bars1 = ax1.bar(names, rates, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel(r"Interception Rate (%)", fontsize=11)
    ax1.set_title(r"DPI Warden Interception Rate ($P_{\mathrm{detect}}$)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=10.0, fontweight="bold")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    # 2. Not-flagged rate. The artifact field is still named evasion_rate,
    # but the quantity is 100 - P_detect, the share of trials the recorded
    # rule did not flag. It is not an evasion measurement.
    evasion = [r.evasion_rate for r in results]
    bars2 = ax2.bar(names, evasion, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel(r"Not-Flagged Rate (%)", fontsize=11)
    ax2.set_title(r"Not-Flagged Rate ($100 - P_{\mathrm{detect}}$)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars2, evasion):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=10.0, fontweight="bold")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _rule_summary(rules: List[str]) -> str:
    """Compact per-scheduler rule attribution, e.g. 'burst 30/50, soft 20/50'."""
    from collections import Counter

    counts = Counter(rules)
    return ", ".join(
        f"{rule.replace('_', ' ')} {count}/{len(rules)}"
        for rule, count in counts.most_common()
    )


def generate_latex_table(
    results: List[SchedulerResult],
    diurnal_results: List[DiurnalResult],
    human_entropy: float,
    output_path: Path,
    commit: str | None = None,
):
    """Generate LaTeX table matching paper requirements with raw strings."""
    provenance = caption_provenance(
        n=f"{len(results)} schedulers x 50 trials x 50 carriers",
        scope=(
            "recorded composite warden; the soft branch is a trained traffic "
            "detector with a threshold calibrated on held-out human traffic, "
            "so a not-flagged rate is a measurement and not an artifact"
        ),
        commit=commit
        or recorded_commit(BENCHMARK_DIR / "component_3_results.json"),
    )
    with open(output_path, "w") as f:
        f.write("% Component 3: Traffic Timing & DPI Evasion Benchmark Results\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(
            r"\caption{Traffic Timing: Warden Interception Rates and Timing Fidelity. "
            r"The second column is the not-flagged rate, not an evasion rate.%s}" % provenance
            + "\n"
        )
        f.write(r"\label{tab:traffic_evasion}" + "\n")
        f.write(r"\begin{tabular}{lccccccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Scheduler & Interception Rate (\%) & Not Flagged Rate (\%) & "
                r"Delay Entropy (bits) & KS Distance & KS $p$-value & Mean Latency (s) & "
                r"Cooldown Violations & Triggering Rule (trials) \\" + "\n")
        f.write(r"\midrule" + "\n")

        for res in results:
            f.write(f"{res.scheduler} & {res.interception_rate:.1f} & {res.evasion_rate:.1f} & "
                    f"{res.delay_entropy:.4f} & {res.ks_statistic:.4f} & {res.ks_pvalue:.2e} & "
                    f"{res.mean_latency:.2f} & {res.cooldown_violations} & "
                    f"{_rule_summary(res.rules)} \\\\\n")

        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{9}{l}{\textbf{Diurnal Conditioning Check (WGAN, identical mapping):}} \\" + "\n")
        for d in diurnal_results:
            f.write(fr"~~\textendash~{d.regime} & {d.interception_rate:.1f} & {d.evasion_rate:.1f} & "
                    f"{d.delay_entropy:.4f} & {d.ks_statistic:.4f} & -- & {d.mean_latency:.2f} & -- & "
                    f"{_rule_summary(d.rules)} \\\\\n")

        f.write(r"\midrule" + "\n")
        f.write(f"Human Baseline & -- & -- & {human_entropy:.4f} & -- & -- & -- & -- & -- \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    scale_table_to_textwidth(output_path)


def run_component3():
    print("=" * 80)
    print(" COMPONENT 3: TRAFFIC TIMING & TRAFFIC ANALYSIS EVASION BENCHMARK (ENHANCED)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Independent Trials: {NUM_TRIALS} per scheduler")
    print(f"Carriers per Message: {CARRIERS_PER_MESSAGE}")
    print()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the trained traffic detector.
    #
    # The neural term comes from the held-out detector, a supervised
    # detector trained with `compute_detector_loss_bce`. The WGAN-GP critic
    # in `warden.pt` is not usable here: its transformer collapsed to a
    # constant output and its anomaly head never received a gradient. See
    # docs/benchmarking/C3_WARDEN_DIAGNOSTIC.md.
    print("1. Loading the trained traffic detector...")
    warden = DeepPacketInspectionWarden(num_channels=NUM_CHANNELS).to(DEVICE)
    if DETECTOR_PATH.exists():
        det_ckpt = load_trained_checkpoint(DETECTOR_PATH, map_location=DEVICE,)
        if isinstance(det_ckpt, dict) and "state_dict" in det_ckpt:
            warden.load_state_dict(det_ckpt["state_dict"])
        else:
            warden.load_state_dict(det_ckpt)
        print(f"   Loaded trained detector from {DETECTOR_PATH}")
    else:
        print(f"   ERROR: {DETECTOR_PATH} not found.")
        print("   Train it first: .venv/bin/python scripts/analysis/train_traffic_detector.py")
        return None
    warden.eval()

    # 2. Generate Human Baseline
    print("\n2. Generating human baseline delays...")
    human_delays = generate_human_baseline_delays(5000)
    human_entropy = compute_delay_entropy(human_delays)
    human_entropy_log = compute_log_bin_entropy(human_delays)
    print(f"   Human delay baseline entropy: {human_entropy:.4f} bits (linear bins), "
          f"{human_entropy_log:.4f} bits (log bins, warden feature)")

    # 3. Initialize StealthScheduler
    print("\n3. Initializing StealthScheduler (GAN & RL agents)...")
    scheduler = StealthScheduler(
        num_channels=NUM_CHANNELS,
        device=DEVICE,
        profile="casual",
        seed=SEED
    )
    # Eagerly load generator and RL agent
    scheduler._load_generator(PROJECT_ROOT / "storage" / "models" / "gan_generator.pt")
    scheduler._load_rl_agent(PROJECT_ROOT / "storage" / "models" / "rl_agent.pt")
    print("   Models loaded successfully.")

    media_ids = [f"carrier_{i:04d}" for i in range(CARRIERS_PER_MESSAGE)]

    # 4. Calibrate the soft-branch threshold on human traffic BEFORE the
    #    schedulers are evaluated, so the recorded interception rates use the
    #    calibrated threshold rather than a placeholder.
    print("\n4. Calibrating the soft-branch threshold on human traffic...")
    corpus_sessions = load_corpus_human_sessions()
    soft_calibration = calibrate_soft_threshold(
        warden, human_delays, corpus_sessions=corpus_sessions, n_trials=200
    )
    global SOFT_FLAG_THRESHOLD
    SOFT_FLAG_THRESHOLD = float(soft_calibration["threshold"])
    print(f"   Soft-branch flag threshold: {SOFT_FLAG_THRESHOLD:.4f}")

    # 5. Evaluate Schedulers
    print(f"\n5. Running multi-trial evaluations ({NUM_TRIALS} trials each)...")
    results: List[SchedulerResult] = []
    delay_records: Dict[str, np.ndarray] = {}

    # 4.1 Static Cadence
    res_static, delays_static = run_scheduler_benchmark_multi(
        "Static Cadence", scheduler, "static", media_ids, BASE_DELAY, warden, human_delays
    )
    results.append(res_static)
    delay_records["Static Cadence"] = delays_static

    # 4.2 Stochastic Poisson
    res_poisson, delays_poisson = run_scheduler_benchmark_multi(
        "Stochastic Poisson", scheduler, "poisson", media_ids, BASE_DELAY, warden, human_delays
    )
    results.append(res_poisson)
    delay_records["Stochastic Poisson"] = delays_poisson

    # 4.3 DCASS Temporal WGAN
    res_gan, delays_gan = run_scheduler_benchmark_multi(
        "DCASS Temporal WGAN", scheduler, "gan", media_ids, BASE_DELAY, warden, human_delays
    )
    results.append(res_gan)
    delay_records["DCASS Temporal WGAN"] = delays_gan

    # 4.4 DCASS Hybrid Engine (WGAN + Channel Masking)
    res_hybrid, delays_hybrid = run_scheduler_benchmark_multi(
        "DCASS Hybrid Engine", scheduler, "hybrid", media_ids, BASE_DELAY, warden, human_delays
    )
    results.append(res_hybrid)
    delay_records["DCASS Hybrid Engine"] = delays_hybrid

    # 4.5 DCASS PPO RL Agent (Optimized)
    res_rl, delays_rl = run_scheduler_benchmark_multi(
        "DCASS PPO RL Agent", scheduler, "rl", media_ids, BASE_DELAY, warden, human_delays
    )
    results.append(res_rl)
    delay_records["DCASS PPO RL Agent"] = delays_rl

    # 6a. Diurnal Circadian Rhythm Analysis (identical mapping for both hours)
    print("\n6a. Running Diurnal Conditioning Check (Hour 14 vs Hour 3, identical mapping)...")
    diurnal_results, diurnal_delays = evaluate_diurnal_circadian_split(
        scheduler, warden, human_delays, n_trials=25
    )

    # 6b. Warden positive control on the human reference itself
    print("\n6b. Warden positive control (human reference self-test)...")
    human_self_test = run_human_reference_self_test(
        warden, human_delays, n_trials=50, corpus_sessions=corpus_sessions
    )

    # 6. Summary metrics
    human_ks = {
        res.scheduler: res.ks_statistic
        for res in results
    }

    # 7. Generate Visualizations & Tables
    print("\n6. Generating benchmark artifacts...")
    cdf_path = PLOTS_DIR / "fig_interarrival_cdf.png"
    circadian_path = PLOTS_DIR / "fig_circadian_diurnal.png"
    bar_path = PLOTS_DIR / "fig_interception_rates.png"
    tex_path = TABLES_DIR / "table3_traffic_evasion.tex"
    json_path = BENCHMARK_DIR / "component_3_results.json"

    create_cdf_plot(delay_records, human_delays, cdf_path)
    print(f"   CDF plot saved: {cdf_path}")

    create_circadian_plot(diurnal_delays, human_delays, circadian_path)
    print(f"   Circadian plot saved: {circadian_path}")

    create_bar_chart(results, bar_path)
    print(f"   Bar chart saved: {bar_path}")

    # The measurement is happening now, so HEAD is the commit that produced
    # these numbers.

    print(f"   LaTeX table saved: {tex_path}")

    output_result = Component3Result(
        schedulers=results,
        human_baseline_entropy=human_entropy,
        human_baseline_entropy_log=human_entropy_log,
        human_ks_distances=human_ks,
        diurnal_breakdown=diurnal_results,
        human_self_test=human_self_test,
        soft_branch_calibration=soft_calibration,
        detector={
            "path": "storage/models/traffic_detector_eval.pt",
            "kind": "supervised binary traffic detector, held out from RL training",
            "score": "bot_probability",
            "flag_threshold": SOFT_FLAG_THRESHOLD,
            "adversary_path": "storage/models/traffic_detector.pt",
            "note": (
                "The soft branch is the trained detector's bot probability. "
                "It is not the WGAN-GP critic: that critic's transformer "
                "collapsed to a constant output and its anomaly head received "
                "no gradient, so it could not separate anything. "
                "The detector used for the measurement is the HELD-OUT one, "
                "not the adversary the RL agent was trained against, so the "
                "result is not a measurement of how well the agent gamed "
                "its own opponent. The KS statistic is recorded per trial "
                "but is not part of the score, because its null against the "
                "human reference is about 0.12 for a burst and 0.28 for a "
                "whole session, and adding it to a calibrated detector "
                "dilutes the detector. "
                "See docs/benchmarking/C3_TRAFFIC_DETECTOR_EVALUATION.md."
            ),
        },
        provenance=collect_provenance(),
    )
    with open(json_path, "w") as f:
        json.dump(asdict(output_result), f, indent=2)
    print(f"   JSON results saved: {json_path}")

    # The table is generated after the artifact is written, so the commit its
    # caption carries is the one that produced the measurement in the
    # artifact rather than the previous run's.
    generate_latex_table(results, diurnal_results, human_entropy, tex_path)

    # Provenance manifest for this run. Records the command, git state,
    # seeds, package versions, input hashes, and output hashes.
    manifest = build_manifest(
        component="component_3_traffic_evasion",
        outputs=[json_path, tex_path, cdf_path, circadian_path, bar_path],
        inputs=[
            PROJECT_ROOT / "storage" / "models" / "gan_generator.pt",
            PROJECT_ROOT / "storage" / "models" / "rl_agent.pt",
            PROJECT_ROOT / "storage" / "models" / "warden.pt",
        ],
        seeds={
            "global_seed": SEED,
            "self_test_seed": SEED + 1,
            "torch_manual_seed": SEED,
            "numpy_seed": SEED,
        },
        extra={
            "num_trials": NUM_TRIALS,
            "carriers_per_message": CARRIERS_PER_MESSAGE,
            "base_delay_seconds": BASE_DELAY,
            "num_channels": NUM_CHANNELS,
            "device": DEVICE,
            "delay_mapping": "(raw - 0.5) * 1500 + 1.5, clipped to [0.5, 30.0] s, "
            "defined in src/stealth/delay_contract.py and shared by the "
            "deployed GAN path, the deployed hybrid path, and this benchmark",
            "warden_note": "storage/models/warden.pt is the historical collapsed "
            "critic, kept as the subject of the diagnostic, not as a usable "
            "model; see docs/benchmarking/C3_WARDEN_DIAGNOSTIC.md. This "
            "benchmark scores with the trained traffic detector.",
        },
    )
    manifest_path = write_manifest(
        manifest, BENCHMARK_DIR / "component_3_results_manifest.json"
    )
    print(f"   Run manifest saved: {manifest_path}")

    print("\n" + "=" * 80)
    print(" ENHANCED COMPONENT 3 BENCHMARK COMPLETE")
    print("=" * 80)
    return output_result


if __name__ == "__main__":
    run_component3()

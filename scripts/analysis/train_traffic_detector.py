#!/usr/bin/env python3
"""
Train a calibrated traffic detector for the Component 3 benchmark.

The warden shipped in `storage/models/warden.pt` is a WGAN-GP critic, not
a detector. Two facts make it unusable as a detection signal, both
established in `docs/benchmarking/C3_WARDEN_DIAGNOSTIC.md`:

  * its transformer collapsed to a constant-output function, so the anomaly
    score is 0.62493 on every input;
  * the WGAN-GP loss differentiates only `raw_critic_score`, so the
    anomaly head receives no gradient at all and stays at its
    initialization.

This script trains a *detector* instead: a supervised binary classifier
over delay sequences, trained with `compute_detector_loss_bce`, which
supervises both the classification head and the per-timestamp anomaly head.

The training corpus is the project's own traffic data:

  * `storage/data/traffic/human_traffic.json` — 2,000 human sessions;
  * `storage/data/traffic/real_human_traffic.json` — 10,000 further
    human sessions.

Bot sessions are synthesized as deviations from human behaviour: a
human-like base session is perturbed toward regularity, toward a fixed
cadence, or toward bursty delivery.

The detector is evaluated on a held-out split and the threshold is
calibrated on a separate calibration split, so the reported false-positive
rate is measured on data the detector never trained on.

Writes:
  storage/models/traffic_detector.pt          the trained detector
  storage/benchmarks/traffic_detector_results.json  metrics and provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.adversarial.warden import (  # noqa: E402
    DeepPacketInspectionWarden,
    compute_detector_loss_bce,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_scoring import select_threshold  # noqa: E402
from traffic_corpus import corpus_files, traffic_dir  # noqa: E402

TRAFFIC_DIR = traffic_dir()
MODEL_PATH = PROJECT_ROOT / "storage" / "models" / "traffic_detector.pt"
RESULTS_PATH = PROJECT_ROOT / "storage" / "benchmarks" / "traffic_detector_results.json"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 64
NUM_CHANNELS = 3


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def load_human_sessions() -> list[dict]:
    """Load every human session from the project's traffic corpus.

    The directory is resolved from `DCASS_TRAFFIC_DIR`, so a real capture
    can be substituted for the synthetic corpus without editing code. See
    `docs/benchmarking/REAL_TRAFFIC_CAPTURE.md`.
    """
    sessions: list[dict] = []
    for path in corpus_files():
        payload = json.loads(path.read_text())
        print(f"  {path.name}: {len(payload)} sessions")
        sessions.extend(payload)
    if not sessions:
        # A detector trained with no human traffic is not a detector, and
        # the failure is quiet: it trains, it scores, and it is wrong.
        raise RuntimeError(
            f"no human traffic corpus found under {TRAFFIC_DIR}. The "
            "detector needs human traffic to learn the human class. Set "
            "DCASS_TRAFFIC_DIR to the corpus directory, or run "
            "scripts/stealth/generate_traffic_dataset.py to build one."
        )
    return sessions


def make_bot_session(human: dict, rng: np.random.Generator) -> dict:
    """Perturb a human session into a bot-like one.

    Three families, matching the traffic patterns a timing-based detector is
    supposed to catch:

    * `cadence`   — a near-fixed inter-arrival interval;
    * `burst`     — delivery compressed into short bursts;
    * `jittered`  — human structure with its long pauses removed, so the
      delay distribution becomes a tight, low-entropy band.

    Each family carries a `strength` in [0, 1] that interpolates between
    the human session and the fully bot-like one. Without it the two
    classes separate trivially, the detector saturates, and the calibrated
    threshold degenerates. With it, some bots are only mildly
    machine-like, which is what makes the task and the threshold
    meaningful.
    """
    delays = np.asarray(human["delays"], dtype=np.float64)
    channels = np.asarray(human["channels"], dtype=np.int64)
    family = rng.choice(["cadence", "burst", "jittered"])
    strength = float(rng.uniform(0.35, 1.0))

    if family == "cadence":
        period = float(rng.uniform(1.0, 8.0))
        noise = rng.normal(0.0, period * 0.02, size=delays.shape)
        target = np.clip(period + noise, 0.5, None)
    elif family == "burst":
        # Half the steps become a tight burst; the rest become long pauses.
        burst_len = int(rng.integers(3, 8))
        short = rng.uniform(0.5, 1.5)
        long = rng.uniform(20.0, 60.0)
        target = np.where(
            (np.arange(len(delays)) % (2 * burst_len)) < burst_len, short, long
        )
    else:  # jittered
        # Keep the human ordering but compress the spread: same mean, far
        # smaller variance, so the delay distribution becomes a narrow band.
        mean = float(delays.mean())
        target = np.clip(
            mean + rng.normal(0.0, max(mean * 0.08, 0.15), size=delays.shape),
            0.5,
            None,
        )

    # Interpolate from the human session toward the bot-like one, so the
    # resulting session keeps some human structure.
    delays = (1.0 - strength) * delays + strength * target
    delays = np.clip(delays, 0.5, None)

    # A bot also tends to reuse one channel, or to alternate in a fixed
    # rotation, rather than switching as a person does. Applied with the
    # same strength, so a weak bot keeps most of the human channel pattern.
    if rng.random() < 0.5:
        bot_channels = np.full_like(channels, int(rng.integers(0, NUM_CHANNELS)))
    else:
        bot_channels = np.arange(len(channels)) % NUM_CHANNELS
    channels = np.where(
        rng.random(len(channels)) < strength, bot_channels, channels
    ).astype(np.int64)

    return {
        "delays": delays.tolist(),
        "channels": channels.tolist(),
        "time_of_day": int(human["time_of_day"]),
    }


class TrafficDataset(Dataset):
    """Padded delay/channel sequences with a bot label."""

    def __init__(self, sessions: list[dict], labels: list[int]):
        self.sessions = sessions
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int):
        session = self.sessions[index]
        delays = np.asarray(session["delays"], dtype=np.float32)
        channels = np.asarray(session["channels"], dtype=np.int64)
        n = len(delays)
        take = min(n, SEQ_LEN)

        padded_delays = np.zeros(SEQ_LEN, dtype=np.float32)
        padded_channels = np.zeros(SEQ_LEN, dtype=np.int64)
        mask = np.zeros(SEQ_LEN, dtype=np.float32)
        padded_delays[:take] = delays[:take]
        padded_channels[:take] = channels[:take]
        mask[:take] = 1.0

        return (
            torch.from_numpy(padded_delays),
            torch.from_numpy(padded_channels),
            torch.from_numpy(mask),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
        )


def split_indices(n: int, rng: np.random.Generator) -> tuple[list, list, list]:
    """Split into train / calibration / holdout, 70 / 15 / 15."""
    order = rng.permutation(n)
    n_train = int(0.70 * n)
    n_cal = int(0.15 * n)
    return (
        order[:n_train].tolist(),
        order[n_train : n_train + n_cal].tolist(),
        order[n_train + n_cal :].tolist(),
    )


def split_human_sessions(
    n: int, rng: np.random.Generator
) -> tuple[list, list, list]:
    """Split the *human* sessions into train / calibration / holdout.

    The split is over the human sessions alone, and the bot sessions are
    synthesized only from the training half. Splitting the combined set
    instead would put bot sessions derived from a held-out human session
    into the training set, and the held-out human sessions would then have
    their own perturbed twins in training. That leaks the holdout, so the
    split is done here, on the human sessions, before any bot session is
    synthesized.
    """
    order = rng.permutation(n)
    n_train = int(0.70 * n)
    n_cal = int(0.15 * n)
    return (
        order[:n_train].tolist(),
        order[n_train : n_train + n_cal].tolist(),
        order[n_train + n_cal :].tolist(),
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def evaluate(
    model: DeepPacketInspectionWarden, loader: DataLoader
) -> dict[str, float]:
    """Threshold-free metrics plus the best achievable accuracy."""
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    anomalies: list[np.ndarray] = []
    masks: list[torch.Tensor] = []
    with torch.no_grad():
        for delays, channels, mask, label in loader:
            verdict = model(
                delays.to(DEVICE),
                channels.to(DEVICE),
                mask=mask.to(DEVICE),
            )
            scores.append(verdict.bot_probability.cpu().numpy())
            anomalies.append(verdict.anomaly_scores.cpu().numpy())
            labels.append(label.numpy())
            masks.append(mask)

    scores_arr = np.concatenate(scores)
    labels_arr = np.concatenate(labels)
    mask_arr = np.concatenate([batch_mask.numpy() for batch_mask in masks])
    # Mean anomaly over the valid steps only.
    per_sample = (np.concatenate(anomalies) * mask_arr).sum(axis=1) / np.maximum(
        mask_arr.sum(axis=1), 1.0
    )

    from sklearn.metrics import roc_auc_score

    result = {
        "n": int(len(scores_arr)),
        "bot_fraction": float(labels_arr.mean()),
        "bot_probability_auc": float(roc_auc_score(labels_arr, scores_arr)),
        "anomaly_mean_auc": float(roc_auc_score(labels_arr, per_sample)),
    }
    for key, values in (("bot_probability", scores_arr), ("anomaly_mean", per_sample)):
        # Threshold-free separation and the best single-threshold accuracy.
        order = np.argsort(values)
        sorted_labels = labels_arr[order]
        n_pos = int(sorted_labels.sum())
        n_neg = len(sorted_labels) - n_pos
        if n_pos and n_neg:
            tp = np.cumsum(sorted_labels)
            fp = np.cumsum(1 - sorted_labels)
            best_acc = float(
                np.max(np.concatenate([[0.0], (tp / n_pos + (n_neg - fp) / n_neg) * 0.5]))
            )
        else:
            best_acc = float("nan")
        result[f"{key}_best_accuracy"] = best_acc
    return result


def calibrate_threshold(
    model: DeepPacketInspectionWarden, loader: DataLoader, target_fpr: float
) -> dict[str, float]:
    """Pick the threshold that holds the false-positive rate at the target.

    The threshold is set on the calibration split, at the quantile of the
    human-class scores that allows `target_fpr` of human traffic to be
    flagged. The score used is the mean anomaly over valid steps, because
    that is the quantity the Component 3 decision function consumes.
    """
    model.eval()
    human_scores: list[np.ndarray] = []
    bot_scores: list[np.ndarray] = []
    with torch.no_grad():
        for delays, channels, mask, label in loader:
            verdict = model(
                delays.to(DEVICE),
                channels.to(DEVICE),
                mask=mask.to(DEVICE),
            )
            m = mask.numpy()
            per_sample = (
                verdict.anomaly_scores.cpu().numpy() * m
            ).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
            is_bot = label.numpy() > 0.5
            if is_bot.any():
                bot_scores.append(per_sample[is_bot])
            else:
                human_scores.append(per_sample[~is_bot])

    human = np.concatenate(human_scores) if human_scores else np.array([0.0])
    bot = np.concatenate(bot_scores) if bot_scores else np.array([0.0])

    # The threshold is the smallest one whose false-positive rate on human
    # traffic is at or below the target. See detector_scoring.py for why
    # this is not simply a quantile.
    threshold, measured_fpr = select_threshold(human, target_fpr)

    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "calibration_fpr": measured_fpr,
        "calibration_tpr": float((bot >= threshold).mean()) if len(bot) else float("nan"),
        "calibration_human_n": int(len(human)),
        "calibration_bot_n": int(len(bot)),
        "calibration_human_max": float(human.max()),
        "calibration_human_p95": float(np.quantile(human, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument(
        "--bot-multiplier",
        type=int,
        default=1,
        help="How many bot sessions to synthesize per human session.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed. A different seed gives a different split, a "
        "different initialization, and a different set of synthesized bot "
        "sessions, which is what makes an independently trained detector a "
        "valid evaluator for an agent trained against another one.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(MODEL_PATH),
        help="Where to write the trained detector.",
    )
    parser.add_argument(
        "--results-out",
        type=str,
        default=str(RESULTS_PATH),
        help="Where to write the training summary.",
    )
    args = parser.parse_args()

    seed = args.seed
    model_path = Path(args.model_out)
    results_path = Path(args.results_out)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print("=" * 78)
    print(" COMPONENT 3: TRAFFIC DETECTOR TRAINING")
    print("=" * 78)
    print(f"Device: {DEVICE}")

    print("\n1. Loading human traffic corpus...")
    human = load_human_sessions()
    if not human:
        print("ERROR: no human sessions found; cannot train.")
        return 2
    print(f"  Total human sessions: {len(human)}")

    print("\n2. Synthesizing bot sessions...")
    # The split is over the *human* sessions, and each subset's bot
    # sessions are synthesized only from the human sessions in that same
    # subset. Splitting the combined set instead would put a bot session
    # derived from a held-out human session into the training set, and the
    # held-out human session would then have its own perturbed twin in
    # training. That leaks the holdout.
    #
    # Every subset therefore carries both classes: the training subset
    # learns the task, the calibration subset sets the threshold, and the
    # holdout subset reports the generalization numbers.
    train_h, cal_h, hold_h = split_human_sessions(len(human), rng)

    def with_bots(indices: list[int], multiplier: int) -> tuple[list[dict], list[int]]:
        """Human sessions from `indices` plus bots derived from them."""
        sessions = [human[i] for i in indices]
        labels = [0] * len(indices)
        for _ in range(multiplier):
            for idx in indices:
                sessions.append(make_bot_session(human[idx], rng))
                labels.append(1)
        return sessions, labels

    train_sessions, train_labels = with_bots(train_h, args.bot_multiplier)
    cal_sessions, cal_labels = with_bots(cal_h, 1)
    hold_sessions, hold_labels = with_bots(hold_h, 1)
    print(
        f"  Split: train {len(train_sessions)} / calibration {len(cal_sessions)} "
        f"/ holdout {len(hold_sessions)}"
    )
    print(
        f"  Human sessions: train {len(train_h)} / calibration {len(cal_h)} "
        f"/ holdout {len(hold_h)}"
    )

    def loader_for(
        sessions: list[dict], labels: list[int], shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            TrafficDataset(sessions, labels),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = loader_for(train_sessions, train_labels, shuffle=True)
    cal_loader = loader_for(cal_sessions, cal_labels, shuffle=False)
    hold_loader = loader_for(hold_sessions, hold_labels, shuffle=False)

    print("\n3. Training the detector...")
    model = DeepPacketInspectionWarden(num_channels=NUM_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        batches = 0
        for delays, channels, mask, label in train_loader:
            optimizer.zero_grad()
            verdict = model(
                delays.to(DEVICE),
                channels.to(DEVICE),
                mask=mask.to(DEVICE),
            )
            loss = compute_detector_loss_bce(
                verdict, label.to(DEVICE), mask=mask.to(DEVICE)
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        mean_loss = total / max(batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={mean_loss:.4f}")

    print("\n4. Calibrating the threshold on the calibration split...")
    calibration = calibrate_threshold(model, cal_loader, args.target_fpr)
    print(
        f"  threshold={calibration['threshold']:.4f} "
        f"(target FPR {args.target_fpr:.0%}, "
        f"measured {calibration['calibration_fpr']:.2%})"
    )

    print("\n5. Evaluating on the held-out split...")
    holdout = evaluate(model, hold_loader)
    print(
        f"  bot_probability AUC: {holdout['bot_probability_auc']:.4f} | "
        f"anomaly-mean AUC: {holdout['anomaly_mean_auc']:.4f}"
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "num_channels": NUM_CHANNELS,
                "seq_len": SEQ_LEN,
                "threshold": calibration["threshold"],
                "target_fpr": args.target_fpr,
                "seed": seed,
            },
            # The held-out human session indices, so a downstream benchmark
            # can calibrate on data the detector never saw instead of
            # re-deriving the split.
            "holdout_human_indices": hold_h,
            "calibration_human_indices": cal_h,
        },
        model_path,
    )
    print(f"\n  Detector saved: {model_path}")

    results = {
        "schema": "traffic-detector-v1",
        "seed": seed,
        "device": str(DEVICE),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "bot_multiplier": args.bot_multiplier,
        "corpus": {
            "human_sessions": len(human),
            "bot_sessions": sum(labels for labels in train_labels),
            "source": f"{TRAFFIC_DIR}/{{human_traffic,real_human_traffic}}.json "
            "plus synthesized bot sessions",
        },
        "split": {
            "train": len(train_sessions),
            "calibration": len(cal_sessions),
            "holdout": len(hold_sessions),
            "human_train": len(train_h),
            "human_calibration": len(cal_h),
            "human_holdout": len(hold_h),
            "note": "the split is over the human sessions; each subset's bot "
            "sessions are synthesized only from the human sessions in that "
            "same subset, so no held-out human session has a perturbed twin "
            "in the training set",
        },
        "calibration": calibration,
        "holdout": holdout,
        "history": history,
        "note": (
            "The human corpus is the project's own traffic data. Its time-of-day "
            "distribution is uniform and its delay values are bounded, so it is "
            "a synthetic surrogate rather than recorded human browsing. This "
            "detector therefore measures how well a supervised detector "
            "separates the two traffic families present in the repository, not "
            "how detectable DCASS traffic is to a real network observer."
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

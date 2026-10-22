#!/usr/bin/env python3
"""
Ingest a real human traffic capture into the corpus format Component 3 uses.

The corpus that ships with the repository is synthetic. Its time-of-day
distribution is uniform, its delays are bounded at 120 seconds, and its
channel runs are short. That is enough to build and test the pipeline
against, and not enough to support a claim about real human browsing.

This script takes a capture of real traffic and produces the corpus the
detector trainer and the Component 3 benchmark read, and reports on the
capture's properties so a reader can see whether it is usable.

INPUT FORMAT
------------
A JSON file holding a list of sessions. Each session is a list of
transmission events, in time order, and each event carries the channel it
went out on:

    [
      {"events": [
         {"t": 1712345678.12, "channel": 0},
         {"t": 1712345681.90, "channel": 1}
       ]},
      ...
    ]

`t` is a Unix timestamp in seconds and `channel` is an integer channel
index. The script derives the inter-arrival delays, the hour of day, and
the session boundaries.

A capture of per-request timestamps from a proxy, a browser extension,
or a platform API log is the expected input. See
`docs/benchmarking/REAL_TRAFFIC_CAPTURE.md`.

USAGE
-----
    .venv/bin/python scripts/analysis/ingest_real_traffic.py \
        --capture my_capture.json \
        --out storage/data/traffic_real/human_traffic.json

Then point the pipeline at the ingested corpus:

    export DCASS_TRAFFIC_DIR=storage/data/traffic_real
    .venv/bin/python scripts/analysis/train_traffic_detector.py
    .venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_capture(payload: object) -> list[dict]:
    """Turn a capture into corpus sessions.

    Accepts either the documented list-of-sessions form or a flat list of
    events, which is split into sessions on a gap of more than
    `session_gap` seconds.
    """
    if not isinstance(payload, list):
        raise ValueError("the capture must be a JSON list")

    # Flat list of events: split on long gaps.
    if payload and isinstance(payload[0], dict) and "t" in payload[0]:
        events = sorted(payload, key=lambda e: float(e["t"]))
        return []

    sessions = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("each session must be an object")
        events = raw.get("events")
        if not events:
            continue
        events = sorted(events, key=lambda e: float(e["t"]))
        sessions.append({"events": events})
    return sessions


def split_flat_events(events: list[dict], session_gap: float) -> list[dict]:
    """Split a flat event list into sessions on a gap of `session_gap` seconds."""
    sessions: list[dict] = []
    current: list[dict] = []
    previous: float | None = None
    for event in sorted(events, key=lambda e: float(e["t"])):
        t = float(event["t"])
        if previous is not None and (t - previous) > session_gap:
            sessions.append({"events": current})
            current = []
        current.append(event)
        previous = t
    if current:
        sessions.append({"events": current})
    return sessions


def to_corpus_sessions(sessions: list[dict], min_len: int, max_len: int) -> list[dict]:
    """Convert capture sessions into the corpus record format."""
    out = []
    for session in sessions:
        events = session["events"]
        times = np.array([float(e["t"]) for e in events])
        channels = np.array([int(e.get("channel", 0)) for e in events])
        if len(times) < 2:
            continue
        delays = np.diff(times)
        # A session's delays are the gaps between consecutive sends.
        delays = delays[(delays > 0) & (delays < 86400)]
        if len(delays) < min_len:
            continue
        if len(delays) > max_len:
            delays = delays[:max_len]
        hour = int(((times[0] % 86400) / 3600.0) % 24)
        out.append(
            {
                "delays": [round(float(d), 3) for d in delays],
                "channels": [int(c) for c in channels[: len(delays)]],
                "time_of_day": hour,
            }
        )
    return out


def describe(sessions: list[dict]) -> dict:
    """Report the properties that decide whether a capture is usable.

    The synthetic corpus fails several of these: its time-of-day
    distribution is uniform, which real browsing never is, and its delays
    are bounded. A capture that fails the same tests is not better than
    the synthetic corpus for the purposes of the claim.
    """
    if not sessions:
        return {"sessions": 0}

    all_delays = np.concatenate([np.asarray(s["delays"]) for s in sessions])
    hours = np.array([s["time_of_day"] for s in sessions])

    # The chi-square goodness of fit of the hour distribution against
    # uniform. Real browsing concentrates on waking hours.
    counts = np.bincount(hours, minlength=24).astype(float)
    expected = counts.sum() / 24.0
    chi2 = float(((counts - expected) ** 2 / expected).sum()) if expected > 0 else 0.0
    # Rough critical value at p = 0.001, 23 degrees of freedom.
    uniform_p = 1.0 if chi2 < 43.0 else 0.0

    # The same-channel reuse gap, and the channel run lengths.
    reuse_gaps = []
    run_lengths = []
    for s in sessions:
        delays = np.asarray(s["delays"], dtype=float)
        channels = np.asarray(s["channels"])
        for i in range(1, len(channels)):
            if channels[i] == channels[i - 1]:
                reuse_gaps.append(float(delays[i]))
        run = 1
        for i in range(1, len(channels)):
            if channels[i] == channels[i - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

    reuse = np.asarray(reuse_gaps) if reuse_gaps else np.array([0.0])
    runs = np.asarray(run_lengths) if run_lengths else np.array([0.0])

    return {
        "sessions": len(sessions),
        "events": int(all_delays.size),
        "delays": {
            "mean": round(float(all_delays.mean()), 4),
            "p50": round(float(np.percentile(all_delays, 50)), 4),
            "p95": round(float(np.percentile(all_delays, 95)), 4),
            "p99": round(float(np.percentile(all_delays, 99)), 4),
            "max": round(float(all_delays.max()), 4),
            "frac_below_1p5s": round(float((all_delays < 1.5).mean()), 4),
            "frac_above_120s": round(float((all_delays > 120).mean()), 6),
        },
        "time_of_day": {
            "chi2_vs_uniform": round(chi2, 2),
            "uniform_p_value_approx": uniform_p,
            "busiest_hours": [
                int(h) for h in np.argsort(-counts)[:3]
            ],
        },
        "channels": {
            "same_channel_reuse_gap_p50": round(float(np.percentile(reuse, 50)), 4),
            "same_channel_reuse_gap_p95": round(float(np.percentile(reuse, 95)), 4),
            "run_length_p50": round(float(np.percentile(runs, 50)), 2),
            "run_length_p95": round(float(np.percentile(runs, 95)), 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="The capture JSON file.")
    parser.add_argument("--out", required=True, help="Where to write the corpus file.")
    parser.add_argument(
        "--session-gap",
        type=float,
        default=900.0,
        help="Seconds of inactivity that end a session, when the capture is a flat event list.",
    )
    parser.add_argument("--min-len", type=int, default=15)
    parser.add_argument("--max-len", type=int, default=80)
    args = parser.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"ERROR: {capture_path} not found")
        return 1
    payload = json.loads(capture_path.read_text())

    sessions = parse_capture(payload)
    if not sessions and isinstance(payload, list) and payload and "t" in payload[0]:
        sessions = split_flat_events(payload, args.session_gap)

    corpus = to_corpus_sessions(sessions, args.min_len, args.max_len)
    if not corpus:
        print("ERROR: no usable sessions in the capture")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus))

    report = describe(corpus)
    report["capture"] = str(capture_path)
    report["out"] = str(out_path)
    report_path = out_path.with_suffix(".provenance.json")
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {len(corpus)} sessions to {out_path}")
    print(f"Wrote the capture report to {report_path}")
    print()
    print(json.dumps(report, indent=2))
    print()
    print("Next: point the pipeline at this corpus and retrain.")
    print(f"  export DCASS_TRAFFIC_DIR={out_path.parent}")
    print("  .venv/bin/python scripts/analysis/train_traffic_detector.py")
    print("  .venv/bin/python scripts/analysis/benchmark_3_traffic_evasion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

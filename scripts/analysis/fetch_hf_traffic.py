#!/usr/bin/env python3
"""Build an exploratory Reddit comment-timestamp proxy dataset.

Reddit comment timestamps are not observations of an individual user's
browsing or network traffic. The channel switching below is synthetic. The
output must not be called a human capture, Tor trace, or DCASS baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Exploratory subreddit-to-proxy-channel mapping. These are not observed
# browser destinations, Tor routes, or DCASS channels.
SUBREDDIT_CHANNEL_MAP = [
    ("programming", 0),
    ("technology", 0),
    ("buildapc", 0),
    ("gadgets", 0),
    ("gaming", 1),
    ("Games", 1),
    ("science", 1),
    ("space", 1),
    ("sports", 1),
    ("AskHistorians", 2),
    ("books", 2),
    ("explainlikeimfive", 2),
    ("todayilearned", 2),
    ("history", 2),
]


def fetch_reddit_activity_proxy(
    target_sessions: int = 1500,
    max_per_sub: int = 8000,
    min_session_len: int = 15,
    max_session_len: int = 80,
    out_capture_path: Path = Path("storage/data/traffic_proxy/reddit_comment_timestamp_proxy.json"),
) -> list[dict]:
    """Stream public comment timestamps and make a synthetic-channel proxy."""
    from datasets import load_dataset

    print(f"Streaming Reddit timestamps to construct >= {target_sessions} proxy sessions...")
    sessions: list[dict] = []

    for sub, channel_idx in SUBREDDIT_CHANNEL_MAP:
        if len(sessions) >= target_sessions:
            break

        print(f"  Streaming subreddit: r/{sub} (Channel {channel_idx})...")
        try:
            ds = load_dataset(
                "HuggingFaceGECLM/REDDIT_comments",
                split=sub,
                streaming=True,
            )
        except Exception as exc:
            print(f"    Warning: skipping r/{sub} due to error: {exc}")
            continue

        thread_events = defaultdict(list)
        count = 0
        for item in ds:
            count += 1
            if count > max_per_sub:
                break
            lid = item.get("link_id")
            created_utc = item.get("created_utc")
            if not lid or created_utc is None:
                continue

            try:
                t = float(created_utc)
            except (ValueError, TypeError):
                continue

            thread_events[lid].append({"t": t, "channel": channel_idx})

        # Convert threads with enough comments into conversation sessions.
        for lid, evs in thread_events.items():
            if len(evs) < min_session_len:
                continue

            evs.sort(key=lambda x: x["t"])

            # Split thread if there is an idle gap longer than 1 hour (3600s)
            curr_chunk = [evs[0]]
            for ev in evs[1:]:
                gap = ev["t"] - curr_chunk[-1]["t"]
                if gap > 3600 or len(curr_chunk) >= max_session_len:
                    if len(curr_chunk) >= min_session_len:
                        sessions.append({"events": curr_chunk})
                    curr_chunk = [ev]
                elif gap > 0:
                    curr_chunk.append(ev)

            if len(curr_chunk) >= min_session_len:
                sessions.append({"events": curr_chunk})

        print(f"    Accumulated {len(sessions)} valid sessions so far.")

    # Apply an explicitly synthetic switching heuristic for exploratory tests.
    channel_rng = random.Random(1337)
    multi_channel_sessions: list[dict] = []
    for s in sessions:
        evs = s["events"]
        base_ch = evs[0].get("channel", 0)
        curr_ch = base_ch
        new_evs = []
        for ev in evs:
            # 20% transition probability per action models human tab-switching
            if channel_rng.random() < 0.20:
                curr_ch = (curr_ch + channel_rng.choice([1, 2])) % 3
            new_evs.append({"t": ev["t"], "channel": int(curr_ch)})
        multi_channel_sessions.append({"events": new_evs})
    sessions = multi_channel_sessions

    print(f"Total Reddit timestamp proxy sessions: {len(sessions)}")
    out_capture_path.parent.mkdir(parents=True, exist_ok=True)
    out_capture_path.write_text(json.dumps(sessions, indent=2))
    provenance = {
        "source": "HuggingFaceGECLM/REDDIT_comments",
        "derived_from_comment_timestamps": True,
        "contributor_ids_present": False,
        "channel_switches_synthetic": True,
        "is_human_browsing_capture": False,
        "is_tor_packet_capture": False,
        "is_dcass_traffic": False,
    }
    out_capture_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Saved raw capture to: {out_capture_path} ({out_capture_path.stat().st_size / 1024:.1f} KB)")
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exploratory Reddit timestamp proxy")
    parser.add_argument("--target-sessions", type=int, default=1500)
    parser.add_argument("--max-per-sub", type=int, default=8000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("storage/data/traffic_proxy/reddit_comment_timestamp_proxy.json"),
    )
    args = parser.parse_args()

    fetch_reddit_activity_proxy(
        target_sessions=args.target_sessions,
        max_per_sub=args.max_per_sub,
        out_capture_path=args.out,
    )
    # Use os._exit to immediately release PyArrow streaming worker threads without Python thread state assert
    os._exit(0)


if __name__ == "__main__":
    main()

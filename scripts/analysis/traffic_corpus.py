"""
Resolve the traffic corpus directory.

The Component 3 work compares DCASS traffic against human traffic, and
the human traffic is the single most important input to that comparison.
The corpus that ships with the repository is synthetic: its time-of-day
distribution is uniform, its delays are bounded at 120 seconds, and it
has no channel runs longer than a handful of steps. A real capture
substituted for it changes what the whole component can claim, so the
directory is resolved from the environment rather than hardcoded.

`DCASS_TRAFFIC_DIR` points the detector trainer, the Component 3
benchmark and the measurement scripts at a different corpus. The directory
is expected to hold one or more JSON files, each a list of sessions of
the form:

    [{"delays": [...], "channels": [...], "time_of_day": 14}, ...]

`scripts/analysis/ingest_real_traffic.py` converts a capture into that
form. See `docs/benchmarking/REAL_TRAFFIC_CAPTURE.md`.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: The corpus files the loader reads, in order.
CORPUS_FILENAMES = ("human_traffic.json", "real_human_traffic.json")


def traffic_dir() -> Path:
    """The directory the human traffic corpus is read from."""
    override = os.environ.get("DCASS_TRAFFIC_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "storage" / "data" / "traffic"


def corpus_files() -> list[Path]:
    """The corpus files that exist, in the order they should be read."""
    directory = traffic_dir()
    return [directory / name for name in CORPUS_FILENAMES if (directory / name).exists()]

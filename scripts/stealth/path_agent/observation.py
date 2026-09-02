"""Shared conversion from raw path measurements to the normalized
observation format the network was trained on. Used by both the simulator
(indirectly, via network_env.py's own normalization) and by real integrations
that feed in live measurements (path_selector.py, choose_path.py).
"""

import numpy as np

from path_agent.config import Config
from path_agent.network_env import _LATENCY_NORM_MS, _UTIL_NORM

_CAPACITY_MAX = Config().capacity_range[1]

REQUIRED_FIELDS = ("latency_ms", "queue", "buffer", "loss_rate", "arrival_rate", "capacity")


def build_observation(paths_raw, capacity_max: float = _CAPACITY_MAX) -> np.ndarray:
    """paths_raw: list of dicts, one per path, each with:
        latency_ms    - measured latency in milliseconds
        queue         - current queue occupancy (same units as `buffer`)
        buffer        - max queue size for that path
        loss_rate     - recent packet loss fraction, 0..1
        arrival_rate  - current traffic rate on the path
        capacity      - the path's bandwidth/capacity (same units as arrival_rate)

    Returns an [N, 5] float32 array matching the network's training format.
    Raises KeyError if a path dict is missing a required field.
    """
    rows = []
    for p in paths_raw:
        missing = [f for f in REQUIRED_FIELDS if f not in p]
        if missing:
            raise KeyError(f"path measurement missing fields: {missing}")

        latency_norm = np.clip(p["latency_ms"] / _LATENCY_NORM_MS, 0.0, 1.0)
        queue_occupancy = np.clip(p["queue"] / p["buffer"], 0.0, 1.0)
        loss_rate = np.clip(p["loss_rate"], 0.0, 1.0)
        utilization = np.clip((p["arrival_rate"] / p["capacity"]) / _UTIL_NORM, 0.0, 1.0)
        capacity_norm = np.clip(p["capacity"] / capacity_max, 0.0, 1.0)
        rows.append([latency_norm, queue_occupancy, loss_rate, utilization, capacity_norm])

    return np.array(rows, dtype=np.float32)

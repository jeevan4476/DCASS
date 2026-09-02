"""Offline scheduling: given a set of timestamps (e.g. produced by a GAN
module) at which a packet must be sent, decide the best channel/path for
each one.

This is decision-only for now - the actual send mechanism isn't decided
yet, so `plan_sends` takes an optional `send_fn` callback you can wire up
later (a socket call, a queue publish, etc.) without changing this module.

Congestion state is supplied by a `congestion_provider`:
  - a plain callable `f(timestamp) -> list[dict]` for a real/live source
    (see observation.py for the required dict fields), or
  - an object exposing `.observe(timestamp) -> list[dict]` and an optional
    `.commit(channel)` (called after each decision, so a stateful source -
    like the simulator below - can advance as if that channel were used).

`SimulatedCongestionProvider` is provided so the whole pipeline (GAN
timestamps -> scheduler -> chosen channel) can be tested end-to-end before
a real congestion source or send mechanism exists.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from path_agent.config import Config
from path_agent.network_env import NetworkEnv
from path_agent.path_selector import PathSelector


@dataclass
class ScheduledSend:
    timestamp: float
    channel: int
    path_stats: list = field(default_factory=list)


class SimulatedCongestionProvider:
    """Congestion source backed by the simulator, for testing before a real
    measurement pipeline is wired in.

    Each `observe()` call returns the current simulated per-path stats;
    `commit(channel)` advances the simulation by one step as if that
    channel had just been used, so later timestamps see updated congestion.

    Note: each call currently advances the simulation by a fixed step
    (`config.dt`) regardless of the actual gap between consecutive
    timestamps - a simplification appropriate for offline testing. Once
    this moves to real-time operation, congestion should instead be read
    live at the moment of each decision, and `commit` can be dropped.
    """

    def __init__(self, config: Optional[Config] = None, seed: int = 0):
        self.cfg = config or Config()
        self.env = NetworkEnv(self.cfg)
        self.env.reset(seed=seed)

    def observe(self, timestamp: float):
        return self.env.raw_path_stats()

    def commit(self, channel: int):
        self.env.step(channel)


def plan_sends(
    timestamps,
    selector: PathSelector,
    congestion_provider,
    send_fn: Optional[Callable[[ScheduledSend], None]] = None,
) -> list[ScheduledSend]:
    """timestamps: iterable of times (e.g. from the GAN module), any order.
    selector: a loaded PathSelector.
    congestion_provider: see module docstring.
    send_fn: optional callback invoked with each ScheduledSend as it's
             decided; leave unset until the send mechanism is settled.

    Returns the schedule as a list of ScheduledSend, sorted by timestamp.
    """
    observe = getattr(congestion_provider, "observe", congestion_provider)
    commit = getattr(congestion_provider, "commit", None)

    schedule = []
    for t in sorted(timestamps):
        paths_raw = observe(t)
        channel = selector.choose(paths_raw)
        entry = ScheduledSend(timestamp=t, channel=channel, path_stats=paths_raw)
        schedule.append(entry)

        if commit is not None:
            commit(channel)
        if send_fn is not None:
            send_fn(entry)

    return schedule

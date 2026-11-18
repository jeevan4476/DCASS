"""Two-timescale empirical timing model for activity-derived proxy traffic.

The model bootstraps short contiguous active-gap blocks and samples longer
session pauses separately. It is not a model of network packets or DCASS
routes; callers provide the source corpus and must preserve that scope.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class TwoTimescaleActivityModel:
    """Bootstrap active timing blocks and long pauses by contributor.

    Gaps above ``max_active_gap_s`` are retained as session-level pauses and
    never emitted as active carrier delays. Contributors are sampled uniformly
    before selecting a block, preventing frequent contributors from dominating
    the empirical schedule.
    """

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        *,
        max_active_gap_s: float = 30.0,
        block_size: int = 5,
    ) -> None:
        if max_active_gap_s <= 0:
            raise ValueError("max_active_gap_s must be positive")
        if block_size < 1:
            raise ValueError("block_size must be at least one")

        active_runs: dict[str, list[list[tuple[float, int]]]] = defaultdict(list)
        pauses: dict[str, list[float]] = defaultdict(list)
        self.active_gap_count = 0
        self.pause_count = 0

        for session in sessions:
            contributor = session.get("contributor_id")
            if contributor is None:
                raise ValueError("every session requires contributor_id")
            delays = session.get("delays", [])
            channels = session.get("channels", [])
            if len(delays) != len(channels):
                raise ValueError("delays and channels must have the same length")

            run: list[tuple[float, int]] = []
            contributor = str(contributor)
            for raw_delay, raw_channel in zip(delays, channels):
                delay = float(raw_delay)
                channel = int(raw_channel)
                if not np.isfinite(delay) or delay <= 0:
                    continue
                if channel not in (0, 1, 2):
                    raise ValueError("channels must be integers in [0, 2]")
                if delay > max_active_gap_s:
                    if run:
                        active_runs[contributor].append(run)
                        run = []
                    pauses[contributor].append(delay)
                    self.pause_count += 1
                else:
                    run.append((delay, channel))
                    self.active_gap_count += 1
            if run:
                active_runs[contributor].append(run)

        self._blocks: dict[str, list[tuple[tuple[float, int], ...]]] = {}
        for contributor, runs in active_runs.items():
            blocks: list[tuple[tuple[float, int], ...]] = []
            for run in runs:
                if len(run) < block_size:
                    blocks.append(tuple(run))
                else:
                    blocks.extend(
                        tuple(run[start : start + block_size])
                        for start in range(len(run) - block_size + 1)
                    )
            if blocks:
                self._blocks[contributor] = blocks

        self._pauses = {key: values for key, values in pauses.items() if values}
        if not self._blocks:
            raise ValueError("no active timing blocks found in corpus")
        self.contributors = tuple(sorted(self._blocks))
        self.pause_contributors = tuple(sorted(self._pauses))
        self.max_active_gap_s = float(max_active_gap_s)
        self.block_size = int(block_size)

    def sample_active_burst(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, list[float] | list[int]]:
        """Return exactly ``n`` proxy delays/channels via contributor-balanced blocks."""
        if n < 1:
            raise ValueError("n must be at least one")
        contributor = self.contributors[int(rng.integers(len(self.contributors)))]
        blocks = self._blocks[contributor]
        sampled: list[tuple[float, int]] = []
        while len(sampled) < n:
            block = blocks[int(rng.integers(len(blocks)))]
            sampled.extend(block)
        sampled = sampled[:n]
        return {
            "delays": [delay for delay, _ in sampled],
            "channels": [channel for _, channel in sampled],
            "contributor_id": contributor,
        }

    def sample_pause(self, rng: np.random.Generator) -> float:
        """Sample one pause, uniformly choosing among contributors with pauses."""
        if not self.pause_contributors:
            raise ValueError("no session pauses above max_active_gap_s were found")
        contributor = self.pause_contributors[
            int(rng.integers(len(self.pause_contributors)))
        ]
        values = self._pauses[contributor]
        return float(values[int(rng.integers(len(values)))])

    def summary(self) -> dict[str, int | float]:
        return {
            "contributors_with_active_gaps": len(self.contributors),
            "contributors_with_pauses": len(self.pause_contributors),
            "active_gap_count": self.active_gap_count,
            "pause_count": self.pause_count,
            "max_active_gap_s": self.max_active_gap_s,
            "block_size": self.block_size,
        }


class ContiguousGapBlockModel:
    """Contributor-balanced moving-block bootstrap over complete gap series.

    Unlike ``TwoTimescaleActivityModel``, this baseline does not split or
    truncate long pauses. It samples contiguous gap blocks from complete
    sessions and ignores LSApp event-type channels, whose meaning is not
    equivalent to DCASS routes.
    """

    def __init__(self, sessions: list[dict[str, Any]], *, block_size: int = 5) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least one")
        sequences: dict[str, list[list[float]]] = defaultdict(list)
        for session in sessions:
            contributor = session.get("contributor_id")
            if contributor is None:
                raise ValueError("every session requires contributor_id")
            delays = [
                float(value)
                for value in session.get("delays", [])
                if np.isfinite(float(value)) and float(value) > 0
            ]
            if delays:
                sequences[str(contributor)].append(delays)

        self._blocks: dict[str, list[tuple[float, ...]]] = {}
        for contributor, contributor_sessions in sequences.items():
            blocks: list[tuple[float, ...]] = []
            for sequence in contributor_sessions:
                if len(sequence) <= block_size:
                    blocks.append(tuple(sequence))
                else:
                    blocks.extend(
                        tuple(sequence[start : start + block_size])
                        for start in range(len(sequence) - block_size + 1)
                    )
            if blocks:
                self._blocks[contributor] = blocks
        if not self._blocks:
            raise ValueError("no valid inter-event gaps found in corpus")
        self.contributors = tuple(sorted(self._blocks))
        self.block_size = int(block_size)

    def to_checkpoint(self) -> dict[str, Any]:
        """Return the fitted empirical model without event labels or session IDs."""
        return {
            "schema": "dcass-contiguous-gap-block-model-v1",
            "block_size": self.block_size,
            "contributor_blocks": {
                key: [list(block) for block in blocks]
                for key, blocks in self._blocks.items()
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "ContiguousGapBlockModel":
        """Load a fitted model checkpoint produced by :meth:`to_checkpoint`."""
        if checkpoint.get("schema") != "dcass-contiguous-gap-block-model-v1":
            raise ValueError("unsupported activity timing checkpoint schema")
        block_size = int(checkpoint.get("block_size", 0))
        if block_size < 1:
            raise ValueError("activity timing checkpoint block_size must be positive")
        contributor_blocks = checkpoint.get("contributor_blocks")
        if not isinstance(contributor_blocks, dict) or not contributor_blocks:
            raise ValueError("activity timing checkpoint has no contributor blocks")

        model = cls.__new__(cls)
        model.block_size = block_size
        model._blocks = {}
        for contributor, raw_blocks in contributor_blocks.items():
            blocks: list[tuple[float, ...]] = []
            for raw_block in raw_blocks:
                block = tuple(float(value) for value in raw_block)
                if not block or any(not np.isfinite(value) or value <= 0 for value in block):
                    raise ValueError("checkpoint blocks must contain finite positive gaps")
                blocks.append(block)
            if blocks:
                model._blocks[str(contributor)] = blocks
        if not model._blocks:
            raise ValueError("activity timing checkpoint has no valid blocks")
        model.contributors = tuple(sorted(model._blocks))
        return model

    @classmethod
    def load(cls, path: str | Path) -> "ContiguousGapBlockModel":
        """Load a JSON checkpoint saved from the training split."""
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_checkpoint(checkpoint)

    def save(self, path: str | Path) -> None:
        """Write the fitted timing model as a portable JSON checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_checkpoint(), separators=(",", ":")), encoding="utf-8")

    def sample_delays(self, n: int, rng: np.random.Generator) -> dict[str, Any]:
        """Sample exactly ``n`` gaps while retaining order within each block."""
        if n < 1:
            raise ValueError("n must be at least one")
        contributor = self.contributors[int(rng.integers(len(self.contributors)))]
        blocks = self._blocks[contributor]
        sampled: list[float] = []
        while len(sampled) < n:
            block = blocks[int(rng.integers(len(blocks)))]
            sampled.extend(block)
        return {"delays": sampled[:n], "contributor_id": contributor}


def route_gaps_with_cooldowns(
    delays: list[float] | tuple[float, ...],
    rng: np.random.Generator,
    cooldowns: dict[int, float],
) -> tuple[list[float], list[int]]:
    """Assign legal channels, extending waits only when every route is busy.

    This benchmark helper deliberately leaves long idle gaps unbounded. It
    models route legality separately from the LSApp event-type labels and
    does not apply the production scheduler's current 30-second cap.
    """
    if not cooldowns or any(value < 0 for value in cooldowns.values()):
        raise ValueError("cooldowns must contain non-negative values")
    if any(not np.isfinite(float(delay)) or float(delay) <= 0 for delay in delays):
        raise ValueError("delays must be finite and positive")

    channels_available = tuple(sorted(cooldowns))
    last_landing = {channel: float("-inf") for channel in channels_available}
    now = 0.0
    routed_delays: list[float] = []
    routed_channels: list[int] = []
    for delay in delays:
        desired_landing = now + float(delay)
        available = [
            channel
            for channel in channels_available
            if last_landing[channel] + float(cooldowns[channel]) <= desired_landing
        ]
        if not available:
            desired_landing = min(
                last_landing[channel] + float(cooldowns[channel])
                for channel in channels_available
            )
            available = [
                channel
                for channel in channels_available
                if last_landing[channel] + float(cooldowns[channel]) <= desired_landing
            ]
        channel = int(available[int(rng.integers(len(available)))])
        routed_delays.append(desired_landing - now)
        routed_channels.append(channel)
        now = desired_landing
        last_landing[channel] = now
    return routed_delays, routed_channels

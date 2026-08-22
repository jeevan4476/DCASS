# src/distribution/scheduler.py

import time
from datetime import datetime, timezone
from typing import List
from .dispatcher import Dispatcher


class Scheduler:
    """
    Dispatches items with per-item delays.

    Delay semantics (matching NoiseController and the API transmitter):
    `delays[i]` is the pause AFTER dispatching item i. The first item is
    sent immediately; the final delay is still honoured so the schedule's
    total duration matches the API path.
    """

    def __init__(self, dispatcher: Dispatcher, delays: List[int]):
        """
        delays: list of seconds to wait AFTER each dispatch
        len(delays) must be >= len(image_sequence)
        """
        self.dispatcher = dispatcher
        self.delays = delays

    def run(self, image_sequence: List[str]) -> List[dict]:
        logs = []

        for idx, image_id in enumerate(image_sequence):
            # Stamp the actual send time.
            log = self.dispatcher.dispatch_one(
                image_id, idx, timestamp=datetime.now(timezone.utc).isoformat()
            )
            logs.append(log)

            # Honour the post-item pause (skippable for tests/demos).
            delay = self.delays[idx]
            if delay > 0 and not getattr(self, "skip_delays", False):
                time.sleep(delay)

        return logs

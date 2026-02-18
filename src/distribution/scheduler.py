# src/distribution/scheduler.py

import time
from datetime import datetime, timedelta
from typing import List
from .dispatcher import Dispatcher

class Scheduler:
    def __init__(
        self,
        dispatcher: Dispatcher,
        delays: List[int]
    ):
        """
        delays: list of seconds to wait BEFORE each dispatch
        len(delays) must be >= len(image_sequence)
        """
        self.dispatcher = dispatcher
        self.delays = delays

    def run(self, image_sequence: List[str]) -> List[dict]:
        logs = []

        scheduled_time = datetime.utcnow()

        for idx, image_id in enumerate(image_sequence):
            delay = self.delays[idx]
            scheduled_time = scheduled_time + timedelta(seconds=delay)
            time.sleep(delay)
            log = self.dispatcher.dispatch_one(image_id, idx, timestamp=scheduled_time.isoformat())
            logs.append(log)

        return logs

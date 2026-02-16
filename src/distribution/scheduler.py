# src/distribution/scheduler.py

import time
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

        for idx, image_id in enumerate(image_sequence):
            delay = self.delays[idx]
            time.sleep(delay)
            log = self.dispatcher.dispatch_one(image_id, idx)
            logs.append(log)

        return logs

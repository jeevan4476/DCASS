# src/distribution/dispatcher.py

from typing import List, Dict, Optional
from .base_channel import BaseChannel

class Dispatcher:
    def __init__(
        self,
        channels: Dict[str, BaseChannel],
        policy: str = "round_robin"
    ):
        if not channels:
            raise ValueError("No channels provided to dispatcher")

        self.channels = channels
        self.policy = policy
        self._channel_names = list(channels.keys())

    def dispatch(
        self,
        image_sequence: List[str],
        timestamps: Optional[List[str]] = None,
    ) -> List[dict]:
        logs = []

        for idx, image_id in enumerate(image_sequence):
            channel = self._select_channel(idx)
            metadata: dict | None = None
            if timestamps is not None and idx < len(timestamps):
                metadata = {"timestamp": timestamps[idx], "sequence_index": idx}
            log = channel.send(image_id, metadata=metadata)
            logs.append(log)

        return logs

    def _select_channel(self, index: int) -> BaseChannel:
        if self.policy == "round_robin":
            name = self._channel_names[index % len(self._channel_names)]
            return self.channels[name]

        if self.policy == "fixed":
            return self.channels[self._channel_names[0]]

        if self.policy == "alternating":
            name = self._channel_names[index % 2]
            return self.channels[name]

        raise ValueError(f"Unknown dispatch policy: {self.policy}")
    
    def dispatch_one(
        self,
        image_id: str,
        index: int,
        timestamp: str | None = None,
    ) -> dict:
        """
        Dispatch a single image to a selected channel.
        Called by the Scheduler.
        """
        channel = self._select_channel(index)
        metadata = {"timestamp": timestamp, "sequence_index": index} if timestamp is not None else None
        return channel.send(image_id, metadata=metadata)


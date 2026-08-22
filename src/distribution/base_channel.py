# src/distribution/base_channel.py

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class BaseChannel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def send(self, image_id: str, metadata: dict | None = None) -> dict:
        """
        Send an image identifier to this channel.

        Returns a log dictionary with:
        - channel
        - image_id
        - timestamp
        """
        pass

    def _base_log(self, image_id: str) -> dict:
        return {
            "channel": self.name,
            "image_id": image_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _base_log_with_timestamp(
        self,
        image_id: str,
        timestamp: str | None,
        metadata: dict | None = None,
    ) -> dict:
        if timestamp is None:
            log = self._base_log(image_id)
        else:
            log = {
                "channel": self.name,
                "image_id": image_id,
                "timestamp": timestamp,
            }

        if metadata:
            for k, v in metadata.items():
                if k == "timestamp":
                    continue
                if k not in log:
                    log[k] = v

        return log

# src/distribution/base_channel.py

from abc import ABC, abstractmethod
from datetime import datetime

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
            "timestamp": datetime.utcnow().isoformat()
        }

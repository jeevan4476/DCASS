# src/distribution/console_channel.py

from .base_channel import BaseChannel

class ConsoleChannel(BaseChannel):
    def __init__(self):
        super().__init__(name="console")

    def send(self, image_id: str, metadata: dict | None = None) -> dict:
        timestamp = None
        if metadata is not None:
            timestamp = metadata.get("timestamp")

        log = self._base_log_with_timestamp(image_id, timestamp, metadata=metadata)
        print(f"[CONSOLE] {log['timestamp']} | {image_id}")
        return log

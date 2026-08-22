# src/distribution/local_folder_channel.py

import os
from .base_channel import BaseChannel

class LocalFolderChannel(BaseChannel):
    def __init__(self, output_dir: str = "phase3_out"):
        super().__init__(name="local_folder")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def send(self, image_id: str, metadata: dict | None = None) -> dict:
        timestamp = None
        if metadata is not None:
            timestamp = metadata.get("timestamp")

        log = self._base_log_with_timestamp(image_id, timestamp, metadata=metadata)

        filename = f"{log['timestamp'].replace(':', '_')}_{image_id}.txt"
        path = os.path.join(self.output_dir, filename)

        print(f"[LOCAL] {log['timestamp']} | {image_id}")


        with open(path, "w", encoding="utf-8") as f:
            f.write(f"IMAGE_ID={image_id}\n")
            if metadata:
                for k, v in metadata.items():
                    f.write(f"{k}={v}\n")

        return log

#!/usr/bin/env python3
# scripts/run_receiver.py
"""
Bob (Receiver) Daemon for DCASS Dockerized Simulation.

Monitors a shared directory for incoming media files, reassembles sequences,
and decodes the secret message using the SemanticDecoder.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.decoder import SemanticDecoder
from src.corpus.index.unified_index import UnifiedSemanticIndex


@dataclass
class ReceivedPacket:
    """Represents a received media packet."""
    media_id: str
    channel_id: int
    sequence_number: int
    timestamp: float
    file_path: Path

    def __lt__(self, other: 'ReceivedPacket') -> bool:
        """Sort by sequence number."""
        return self.sequence_number < other.sequence_number


@dataclass
class ReassemblyBuffer:
    """
    Buffer for collecting and reassembling out-of-order packets.

    Since the RL agent might send packets out of order to maximize stealth,
    we need to collect them, wait for a silence threshold, then reassemble.
    """
    packets: List[ReceivedPacket] = field(default_factory=list)
    last_packet_time: float = field(default_factory=time.time)
    silence_threshold: float = 10.0  # Seconds of silence before reassembly

    def add_packet(self, packet: ReceivedPacket):
        """Add a packet to the buffer."""
        self.packets.append(packet)
        self.last_packet_time = time.time()

    def is_complete(self) -> bool:
        """Check if we should reassemble (silence threshold reached)."""
        if not self.packets:
            return False

        time_since_last = time.time() - self.last_packet_time
        return time_since_last >= self.silence_threshold

    def reassemble(self) -> List[str]:
        """
        Reassemble packets into ordered media ID sequence.

        Sorts by sequence_number (from metadata) or timestamp if no sequence number.
        """
        if not self.packets:
            return []

        # Sort packets
        sorted_packets = sorted(self.packets)

        # Extract media IDs in order
        media_ids = [p.media_id for p in sorted_packets]

        # Clear buffer
        self.packets.clear()

        return media_ids

    def __len__(self) -> int:
        return len(self.packets)


class ReceiverDaemon:
    """
    Asynchronous receiver daemon for DCASS.

    Watches a shared directory for incoming media files, buffers them,
    reassembles sequences, and decodes messages.
    """

    def __init__(
        self,
        watch_directory: Path,
        silence_threshold: float = 10.0,
        poll_interval: float = 1.0,
        decoder: Optional[SemanticDecoder] = None,
        decode_enabled: bool = True,
        exit_after_decode: bool = False,
    ):
        """
        Initialize receiver daemon.

        Args:
            watch_directory: Directory to monitor for incoming files
            silence_threshold: Seconds of silence before reassembly
            poll_interval: Seconds between directory polls
            decoder: Pre-initialized SemanticDecoder (created if None)
            decode_enabled: If True, load SemanticDecoder and decode on
                reassembly. Disable for packet-only sniff tests when the
                FAISS indices are not available in this container.
        """
        self.watch_directory = Path(watch_directory)
        self.watch_directory.mkdir(parents=True, exist_ok=True)

        self.poll_interval = poll_interval
        self.buffer = ReassemblyBuffer(silence_threshold=silence_threshold)

        # Decoder
        self._decoder = decoder
        self._decoder_loaded = False
        self._decode_enabled = decode_enabled
        self._exit_after_decode = exit_after_decode
        self._done = False

        # Tracking
        self.processed_files: set[str] = set()
        self.decoded_messages: List[str] = []

        print(f"[Receiver] Initialized")
        print(f"  Watch directory: {self.watch_directory}")
        print(f"  Silence threshold: {silence_threshold}s")
        print(f"  Decoding:          {'enabled' if decode_enabled else 'disabled (sniff-only)'}")

    @property
    def decoder(self) -> SemanticDecoder:
        """Get decoder (lazy initialization)."""
        if self._decoder is None:
            print("[Receiver] Loading SemanticDecoder (this may take a few seconds)...")
            self._decoder = SemanticDecoder()
            self._decoder.load()
            self._decoder_loaded = True
            print("[Receiver] SemanticDecoder ready.")
        return self._decoder

    def parse_packet_metadata(self, file_path: Path) -> Optional[ReceivedPacket]:
        """
        Parse metadata from a received file.

        Expected filename format: {media_id}_{channel}_{sequence}.json

        Args:
            file_path: Path to received file

        Returns:
            ReceivedPacket or None if parsing fails
        """
        try:
            # Read metadata file
            with open(file_path, 'r') as f:
                metadata = json.load(f)

            packet = ReceivedPacket(
                media_id=metadata.get("media_id", "unknown"),
                channel_id=metadata.get("channel_id", 0),
                sequence_number=metadata.get("sequence_number", 0),
                timestamp=metadata.get("timestamp", time.time()),
                file_path=file_path
            )

            return packet

        except Exception as e:
            print(f"[Receiver] Error parsing {file_path}: {e}")
            return None

    async def _watch_directory(self):
        """
        Asynchronously watch directory for new files.

        Monitors for .json files (packet metadata), parses them,
        and adds to reassembly buffer.
        """
        print(f"[Receiver] Watching {self.watch_directory}...")

        while True:
            # Scan for new files
            for file_path in self.watch_directory.glob("*.json"):
                # Skip sender-side control files (e.g. _manifest.json)
                if file_path.name.startswith("_"):
                    continue
                # Skip if already processed
                if file_path.name in self.processed_files:
                    continue

                # Parse packet
                packet = self.parse_packet_metadata(file_path)
                if packet is not None:
                    print(
                        f"[Receiver] Received packet: {packet.media_id} "
                        f"(seq={packet.sequence_number}, channel={packet.channel_id})"
                    )
                    self.buffer.add_packet(packet)

                # Mark as processed
                self.processed_files.add(file_path.name)

            # Check if buffer is ready for reassembly
            if self.buffer.is_complete():
                await self.reassemble_and_decode()
                if self._exit_after_decode and self._done:
                    print("[Receiver] exit-after-decode set; shutting down.")
                    return

            # Sleep before next poll
            await asyncio.sleep(self.poll_interval)

    async def reassemble_and_decode(self):
        """Reassemble buffered packets and decode message."""
        print(f"[Receiver] Silence threshold reached. Reassembling {len(self.buffer)} packets...")

        # Reassemble media ID sequence
        media_ids = self.buffer.reassemble()

        if not media_ids:
            print("[Receiver] No packets to reassemble")
            return

        print(f"[Receiver] Reassembled sequence ({len(media_ids)} items): {media_ids}")

        if not self._decode_enabled:
            print("[Receiver] Decoding disabled (sniff-only mode); skipping decode.")
            print("-" * 60)
            return

        # Decode media sequence back to semantic meaning
        try:
            result = self.decoder.decode(media_ids)
        except Exception as e:
            print(f"[Receiver] Decoding error: {e}")
            print("-" * 60)
            return

        print("")
        print("=" * 60)
        print("[Receiver] DECODED MESSAGE")
        print("=" * 60)
        for i, item in enumerate(result.decoded, 1):
            status = "OK " if item.verified else "MISS"
            snippet = item.content if len(item.content) <= 80 else item.content[:77] + "..."
            print(f"  {i:2d}. [{status}] [{item.modality}] {item.media_id}")
            print(f"       \"{snippet}\"")
        print("-" * 60)
        print(f"  Verification rate : {result.verification_rate:.1%}")
        print(f"  All verified      : {result.all_verified}")
        print(f"  Reconstructed     : \"{result.reconstructed_meaning}\"")
        print("=" * 60)
        self.decoded_messages.append(result.reconstructed_meaning)
        print("-" * 60)
        self._done = True

    async def run(self):
        """Run the receiver daemon."""
        print("[Receiver] Starting daemon...")
        try:
            await self._watch_directory()
        except KeyboardInterrupt:
            print("\n[Receiver] Shutting down...")
        except Exception as e:
            print(f"[Receiver] Fatal error: {e}")
            raise

    def run_sync(self):
        """Run the daemon synchronously (for compatibility)."""
        asyncio.run(self.run())


def main():
    """Main entry point for receiver daemon."""
    parser = argparse.ArgumentParser(
        description="DCASS Receiver Daemon (Bob)"
    )
    parser.add_argument(
        "--watch",
        type=str,
        default="/app/shared_channel",
        help="Directory to watch for incoming packets"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Silence threshold (seconds) before reassembly"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between directory polls"
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Sniff-only mode: log reassembled media IDs but skip decoding "
             "(use when FAISS indices are not available in the container)"
    )
    parser.add_argument(
        "--exit-after-decode",
        action="store_true",
        help="Exit after the first successful reassembly+decode (one-shot demo mode)"
    )

    args = parser.parse_args()

    # Create receiver
    receiver = ReceiverDaemon(
        watch_directory=Path(args.watch),
        silence_threshold=args.timeout,
        poll_interval=args.poll_interval,
        decode_enabled=not args.no_decode,
        exit_after_decode=args.exit_after_decode,
    )

    # Run daemon
    print("=" * 60)
    print("DCASS Receiver Daemon (Bob)")
    print("=" * 60)
    receiver.run_sync()


if __name__ == "__main__":
    main()

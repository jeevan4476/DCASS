# src/stealth/pipeline.py
"""
Phase F — Unified end-to-end pipeline.

A single programmatic entry point that proves the full DCASS round-trip:

    message str
       ↓   SemanticEncoder.encode(message)
    media_ids
       ↓   StealthScheduler.schedule(media_ids, mode)
    (delays, channels)
       ↓   transmitter writes JSON packets to
       │      storage/shared_channel/<recipient_id>/<session_id>__*.json
       │      with real time.sleep() between packets (scaled by
       │      speed_multiplier so tests don't take real wall-clock minutes)
       ↓
    reassembler reads packets sorted by sequence_number
       ↓   SemanticDecoder.decode(media_ids)
    reconstructed message

See docs/GAN_RL_INTEGRATION_PLAN.md Phase F.3. The pipeline here is
"same-process, shared-channel-faithful": sender and reassembly run as
two phases of ONE Python process, but still go through the real
file-based transport layer so what we test is what gets deployed.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PipelineResult:
    """Result of one end-to-end round-trip through the full DCASS stack."""

    original_message: str
    decoded_message: str
    sender: str
    recipient: str
    mode_used: str
    round_trip_seconds: float
    decoded_ok: bool
    verification_rate: float
    schedule: dict
    ecc_success: bool
    ecc_errors_fixed: list[int] = field(default_factory=list)
    session_id: str = ""
    error: Optional[str] = None

    def summary(self) -> str:
        tick = "✓" if self.decoded_ok else "✗"
        lines = [
            "=" * 64,
            f"  DCASS pipeline round-trip  [{tick}]",
            "=" * 64,
            f"  Sender         : {self.sender}",
            f"  Recipient      : {self.recipient}",
            f"  Mode used      : {self.mode_used}",
            f"  Session ID     : {self.session_id}",
            f"  Round-trip     : {self.round_trip_seconds:.3f}s",
            f"  Items in sched : {len(self.schedule.get('delays', []))}",
            f"  Total delay    : {sum(self.schedule.get('delays', [])):.1f}s"
            f"  (sender time; speed_multiplier scales this down for tests)",
            f"  Verification   : {self.verification_rate:.1%}",
            f"  ECC            : {'OK' if self.ecc_success else 'FAILED'}"
            f"  (errors fixed: {len(self.ecc_errors_fixed)})",
            "-" * 64,
            f"  Original       : {self.original_message!r}",
            f"  Decoded        : {self.decoded_message!r}",
            f"  Match          : {self.decoded_ok}",
        ]
        if self.error:
            lines += ["-" * 64, f"  Error          : {self.error}"]
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lazy engine singleton (cached across calls so test runs don't repay cost)
# ---------------------------------------------------------------------------
_engine_singleton = None
_engine_lock = threading.Lock()


def _get_engine():
    """
    Cache the SemanticEngine across pipeline calls. First call pays the
    FAISS + CLIP + codebook load cost (seconds); later calls are free.
    """
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton
    with _engine_lock:
        if _engine_singleton is None:
            from src.engine.semantic_engine import SemanticEngine

            engine = SemanticEngine()
            engine.load()
            _engine_singleton = engine
    return _engine_singleton


# ---------------------------------------------------------------------------
# Transmitter + reassembler
# ---------------------------------------------------------------------------


def _resolve_user_id(username: str) -> Optional[int]:
    """
    Look up a user id by username, if the Phase F auth DB exists.
    Falls back to None (programmatic mode without auth DB) — the pipeline
    then uses `storage/shared_channel/_pipeline_<username>/` as the
    recipient's subdirectory, keeping programmatic runs isolated from
    real users.
    """
    try:
        from src.api import auth as _auth

        _auth.init_db()  # idempotent
        user = _auth.get_user_by_username(username)
        return user.id if user else None
    except Exception:
        return None


def _recipient_dir(shared_dir: Path, recipient_username: str) -> Path:
    """Resolve the recipient's subdirectory under `shared_dir`."""
    user_id = _resolve_user_id(recipient_username)
    if user_id is not None:
        sub = shared_dir / str(user_id)
    else:
        # No auth DB entry — isolate programmatic runs from real inboxes.
        sub = shared_dir / f"_pipeline_{recipient_username}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def _write_packet(
    recipient_dir: Path,
    session_id: str,
    media_id: str,
    channel: int,
    sequence_number: int,
    delay: float,
    mode_used: str,
) -> None:
    """Write one packet JSON into the recipient's shared-channel subdir."""
    # Use opaque filename without sequence_number or channel_id in the filename
    # to prevent wire metadata leakage. All metadata is inside the JSON content.
    filename = f"{session_id}__{media_id}.json"
    packet = {
        "media_id": media_id,
        "channel_id": channel,
        "sequence_number": sequence_number,
        "delay_seconds": round(delay, 3),
        "timestamp": time.time(),
        "mode_used": mode_used,
        "session_id": session_id,
    }
    with open(recipient_dir / filename, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)


def _write_manifest(
    recipient_dir: Path,
    session_id: str,
    message: str,
    schedule: dict,
    sender: str,
    sender_id: Optional[int],
    recipient: str,
    recipient_id: Optional[int],
) -> None:
    manifest = {
        "message": message,
        "mode_used": schedule["mode_used"],
        "total_items": len(schedule["items"]),
        "total_delay_seconds": round(sum(schedule["delays"]), 2),
        "timestamp": time.time(),
        "sender_id": sender_id,
        "sender_username": sender,
        "recipient_id": recipient_id,
        "recipient_username": recipient,
        "session_id": session_id,
    }
    with open(recipient_dir / f"_manifest_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _transmit(
    schedule: dict,
    recipient_dir: Path,
    session_id: str,
    speed_multiplier: float,
) -> None:
    """
    Dispatch the schedule's items through the recipient directory with real
    sleeps. Dispatch-then-wait semantics (matching the delay convention
    documented in src/stealth/stealth_scheduler.py and src/distribution/noise.py).
    """
    items = schedule["items"]
    delays = schedule["delays"]
    channels = schedule["channels"]
    mode_used = schedule["mode_used"]

    for idx, (media_id, delay, channel) in enumerate(zip(items, delays, channels)):
        if media_id is None:
            # Noise gap — just wait, don't write.
            if delay > 0 and speed_multiplier > 0:
                time.sleep(delay / speed_multiplier)
            continue

        _write_packet(
            recipient_dir=recipient_dir,
            session_id=session_id,
            media_id=media_id,
            channel=int(channel),
            sequence_number=idx,
            delay=float(delay),
            mode_used=mode_used,
        )

        # Sleep AFTER the write (delay-after-send convention). Skip the
        # last-item delay since nothing follows it.
        if delay > 0 and speed_multiplier > 0 and idx < len(items) - 1:
            time.sleep(delay / speed_multiplier)


def _reassemble(recipient_dir: Path, session_id: str) -> list[str]:
    """Read all packets for `session_id`, sort by sequence_number."""
    packets: list[dict] = []
    for pkt_path in recipient_dir.glob(f"{session_id}__*.json"):
        try:
            with open(pkt_path, "r", encoding="utf-8") as f:
                packets.append(json.load(f))
        except Exception:
            continue
    packets.sort(key=lambda p: p.get("sequence_number", 0))
    return [p["media_id"] for p in packets]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    message: str,
    sender: str,
    recipient: str,
    mode: Literal["static", "gan", "rl", "auto"] = "auto",
    speed_multiplier: float = 100.0,
    shared_dir: Optional[Path] = None,
    gan_checkpoint: Optional[Path] = None,
    rl_checkpoint: Optional[Path] = None,
    num_channels: int = 3,
    base_delay: float = 3.0,
    cleanup: bool = True,
    codec_mode: Literal["dssc", "exact_vcp"] = "dssc",
    session_key: Optional[bytes] = None,
) -> PipelineResult:
    """
    Run one message end-to-end through the real transport.

    Args:
        message: the plaintext secret message to send.
        sender, recipient: usernames. Used for the manifest / inbox routing.
            If the Phase F auth DB has entries for them, their integer
            user_ids drive the subdirectory choice; otherwise programmatic
            runs land in `_pipeline_<username>/` to stay out of real inboxes.
        mode: scheduling mode. "auto" tries rl → gan → static.
        speed_multiplier: real wall-clock delays are divided by this factor.
            100.0 means a 5s delay sleeps 50ms. Set to 1.0 to run against
            real time (matches deployment).
        shared_dir: override the default `storage/shared_channel/` root.
            Pass a tmp path from pytest so parallel pipeline runs don't
            collide.
        gan_checkpoint, rl_checkpoint: scheduler checkpoint overrides.
            Default is to use the project-root paths StealthScheduler
            already looks for.
        num_channels: channel count passed to the StealthScheduler.
        base_delay: fallback delay for the static NoiseController path.
        cleanup: if True (default), remove the session's packets and
            manifest after the pipeline returns, so repeated test runs
            don't accumulate state. The decoded sidecar (if any) is NOT
            written in this programmatic path — this function returns the
            decoded message directly in PipelineResult.
        codec_mode: encoding mode. "dssc" (default) uses session-keyed DSSC;
            "exact_vcp" uses the baseline Voronoi codebook codec.
        session_key: 32-byte session key for DSSC mode. If None and codec_mode="dssc",
            a random key is generated. Must be provided for decoding to succeed.

    Returns:
        PipelineResult. decoded_ok is True iff decoded == original message.
    """
    from src.stealth.stealth_scheduler import StealthScheduler
    import os

    t0 = time.perf_counter()

    if shared_dir is None:
        shared_dir = _PROJECT_ROOT / "storage" / "shared_channel"
    shared_dir.mkdir(parents=True, exist_ok=True)

    recipient_dir = _recipient_dir(shared_dir, recipient)
    sender_id = _resolve_user_id(sender)
    recipient_id = _resolve_user_id(recipient)
    session_id = uuid.uuid4().hex[:16]

    # Generate session key for DSSC if needed
    if codec_mode == "dssc" and session_key is None:
        session_key = os.urandom(32)

    # 1. Encode message -> media_ids
    engine = _get_engine()
    try:
        if codec_mode == "dssc":
            encode_result = engine.encode(
                message=message,
                mode=codec_mode,
                session_key=session_key,
                use_ecc=True,
            )
        else:
            encode_result = engine.encode(message, use_ecc=True)
    except Exception as e:
        return PipelineResult(
            original_message=message,
            decoded_message="",
            sender=sender,
            recipient=recipient,
            mode_used="(encode failed)",
            round_trip_seconds=time.perf_counter() - t0,
            decoded_ok=False,
            verification_rate=0.0,
            schedule={},
            ecc_success=False,
            session_id=session_id,
            error=f"encode: {e}",
        )

    # The EncodeResult may wrap exact_vcp inside .exact_vcp_result;
    # unify to a flat media_ids list.
    if hasattr(encode_result, "media_ids") and encode_result.media_ids:
        media_ids = list(encode_result.media_ids)
    elif getattr(encode_result, "exact_vcp_result", None) is not None:
        media_ids = list(encode_result.exact_vcp_result.media_ids)
    else:
        return PipelineResult(
            original_message=message,
            decoded_message="",
            sender=sender,
            recipient=recipient,
            mode_used="(encode produced no media_ids)",
            round_trip_seconds=time.perf_counter() - t0,
            decoded_ok=False,
            verification_rate=0.0,
            schedule={},
            ecc_success=False,
            session_id=session_id,
            error="encoder returned no media_ids",
        )

    # 2. Schedule
    scheduler = StealthScheduler(num_channels=num_channels, device="cpu", profile="casual")
    try:
        schedule = scheduler.schedule(
            media_ids=media_ids,
            mode=mode,
            base_delay=base_delay,
            gan_checkpoint=gan_checkpoint,
            rl_checkpoint=rl_checkpoint,
        )
    except Exception as e:
        return PipelineResult(
            original_message=message,
            decoded_message="",
            sender=sender,
            recipient=recipient,
            mode_used="(schedule failed)",
            round_trip_seconds=time.perf_counter() - t0,
            decoded_ok=False,
            verification_rate=0.0,
            schedule={},
            ecc_success=False,
            session_id=session_id,
            error=f"schedule: {e}",
        )

    # 3. Write manifest, then transmit with real (scaled) delays
    _write_manifest(
        recipient_dir=recipient_dir,
        session_id=session_id,
        message=message,
        schedule=schedule,
        sender=sender,
        sender_id=sender_id,
        recipient=recipient,
        recipient_id=recipient_id,
    )
    _transmit(
        schedule=schedule,
        recipient_dir=recipient_dir,
        session_id=session_id,
        speed_multiplier=speed_multiplier,
    )

    # 4. Reassemble on the receiver side (same process, real file I/O)
    received_media_ids = _reassemble(recipient_dir, session_id)

    # 5. Decode
    try:
        if codec_mode == "dssc":
            # session_key is guaranteed to be non-None here (generated above if needed)
            assert session_key is not None
            decode_result = engine.decode(
                media_ids=received_media_ids, mode=codec_mode, session_key=session_key, use_ecc=True
            )
        else:
            decode_result = engine.decode(
                media_ids=received_media_ids, mode=codec_mode, use_ecc=True
            )
    except Exception as e:
        decoded_message = ""
        verification_rate = 0.0
        ecc_success = False
        ecc_errors_fixed: list[int] = []
        error = f"decode: {e}"
    else:
        decoded_message = decode_result.reconstructed_message or ""
        verification_rate = decode_result.verification_rate
        if decode_result.exact_vcp_result is not None:
            ecc_success = decode_result.exact_vcp_result.ecc_success
            ecc_errors_fixed = list(decode_result.exact_vcp_result.ecc_errors_fixed)
        else:
            ecc_success = decode_result.success
            ecc_errors_fixed = list(decode_result.ecc_fixed_errors)
        error = None

    # 6. Optional cleanup so repeated test runs don't accumulate
    if cleanup:
        try:
            for p in list(recipient_dir.glob(f"{session_id}__*.json")) + [
                recipient_dir / f"_manifest_{session_id}.json"
            ]:
                if p.exists():
                    p.unlink()
        except Exception:
            pass

    return PipelineResult(
        original_message=message,
        decoded_message=decoded_message,
        sender=sender,
        recipient=recipient,
        mode_used=schedule["mode_used"],
        round_trip_seconds=time.perf_counter() - t0,
        decoded_ok=(decoded_message == message),
        verification_rate=verification_rate,
        schedule=schedule,
        ecc_success=ecc_success,
        ecc_errors_fixed=ecc_errors_fixed,
        session_id=session_id,
        error=error,
    )

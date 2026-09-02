# src/api/server.py
"""
DCASS FastAPI Backend.

Exposes the DCASS engine (encode, decode, search, status, benchmark)
as a REST API for the Next.js frontend.

Usage:
    uvicorn src.api.server:app --reload --port 8000
"""

from __future__ import annotations

import os
import time
import json
import threading
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.corpus.index.unified_index import resolve_indices_base_path


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
_DEFAULT_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:3001,http://127.0.0.1:3001,"
    "http://localhost:8000,http://127.0.0.1:8000"
)
app = FastAPI(
    title="DCASS API",
    description="Dynamic Context-Aware Semantic Steganography",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Allow known frontend origins, and regex-match any localhost/127.0.0.1 dev port
    allow_origins=[
        o.strip()
        for o in os.environ.get("DCASS_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
        if o.strip()
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth (Bearer token when DCASS_API_TOKEN is set)
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


def require_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """
    Gate sensitive routes when DCASS_API_TOKEN is configured.

    When the env var is unset, auth is skipped (local/dev). Production
    deployments must set DCASS_API_TOKEN.
    """
    expected = os.environ.get("DCASS_API_TOKEN")
    if not expected:
        return
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or credentials.credentials != expected
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _context_secret_from_env() -> Optional[bytes]:
    raw = os.environ.get("DCASS_CONTEXT_SECRET")
    if not raw:
        return None
    return raw.encode("utf-8")


def _build_context_manager(bucket_seconds: int):
    """Wire API dynamic context to optional DCASS_CONTEXT_SECRET."""
    from src.engine.context import ContextKeyManager

    secret = _context_secret_from_env()
    return ContextKeyManager(bucket_seconds=bucket_seconds, secret=secret)


def sanitize_packet_filename(shared_dir: Path, media_id: str, channel: int, idx: int) -> Path:
    """
    Build a packet path under *shared_dir*, rejecting path traversal.

    Raises ValueError when media_id contains separators, `..`, or would
    resolve outside shared_dir.
    """
    if not media_id or not isinstance(media_id, str):
        raise ValueError("media_id must be a non-empty string")
    if media_id in (".", "..") or ".." in media_id:
        raise ValueError(f"invalid media_id: {media_id!r}")
    if "/" in media_id or "\\" in media_id or media_id != Path(media_id).name:
        raise ValueError(f"invalid media_id (path characters): {media_id!r}")

    shared_resolved = shared_dir.resolve()
    filename = f"{media_id}_{channel}_{idx:04d}.json"
    path = (shared_dir / filename).resolve()
    if not path.is_relative_to(shared_resolved):
        raise ValueError(f"packet path escapes shared_channel: {media_id!r}")
    return path


def validate_transmit_media_ids(media_ids: list[str]) -> None:
    """Reject unsafe media_ids before scheduling transmission."""
    for mid in media_ids:
        if not mid or not isinstance(mid, str):
            raise ValueError("media_ids entries must be non-empty strings")
        if mid in (".", "..") or ".." in mid:
            raise ValueError(f"invalid media_id: {mid!r}")
        if "/" in mid or "\\" in mid or mid != Path(mid).name:
            raise ValueError(f"invalid media_id (path characters): {mid!r}")


# ---------------------------------------------------------------------------
# Lazy engine singletons
# ---------------------------------------------------------------------------
_engine = None
_engine_lock = threading.Lock()
_initializing = False
_ready = False

# Transmission state
_transmission_active = False
_transmission_stop_requested = False
_transmission_progress = {"current": 0, "total": 0, "status": "idle"}
_transmission_lock = threading.Lock()

# Cached index counts for /api/status (avoids re-reading FAISS files per call)
_index_counts_cache: Optional[dict] = None


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from src.engine.semantic_engine import SemanticEngine

                engine = SemanticEngine()
                engine.load()
                _engine = engine
    return _engine


def _get_encoder():
    return _get_engine()._exact_encoder


def _get_decoder():
    return _get_engine()._exact_decoder


def warmup():
    """Pre-load unified SemanticEngine on startup."""
    global _initializing, _ready
    _initializing = True
    print("\n" + "=" * 70)
    print("🔥 Warming up DCASS engine...")
    print("=" * 70)
    try:
        print("\n📦 Loading SemanticEngine and indices...")
        engine = _get_engine()
        print(f"✅ SemanticEngine ready: {engine}")

        _ready = True
        print("\n" + "=" * 70)
        print("✅ DCASS engine ready!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Warmup failed: {e}")
        print("=" * 70 + "\n")
        _ready = False
    finally:
        _initializing = False


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class EncodeRequest(BaseModel):
    message: str
    mode: Optional[str] = "exact_vcp"
    session_key_hex: Optional[str] = None
    diversity_mode: Literal["best", "round_robin", "balanced"] = "best"
    modalities: list[str] = Field(default=["image", "text", "audio"])
    use_ecc: bool = True
    ecc_parity_bytes: int = Field(default=8, ge=1, le=64)
    use_dynamic_context: bool = False
    context_bucket_seconds: int = Field(default=3600, ge=1)


class EncodeResponse(BaseModel):
    mode: str = "exact_vcp"
    media_ids: list[str]
    carrier_count: int = 0
    chunks: list[str] = Field(default_factory=list)
    encoded: list[dict] = Field(default_factory=list)
    media_sequence: list[dict] = Field(default_factory=list)
    modality_breakdown: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: float
    bits_per_carrier: float = 8.0
    ecc_parity_bytes: int = 0
    payload_bytes: list[int] = Field(default_factory=list)
    context_info: dict = Field(default_factory=dict)


class DecodeRequest(BaseModel):
    media_ids: list[str]
    mode: Optional[Literal["exact_vcp", "dssc"]] = "exact_vcp"
    session_key_hex: Optional[str] = None
    modalities: Optional[list[str]] = None
    use_ecc: bool = True
    ecc_parity_bytes: int = Field(default=8, ge=1, le=64)
    use_dynamic_context: bool = False
    context_bucket_seconds: int = Field(default=3600, ge=1)
    context_epoch_hint: Optional[str] = None


class DecodeResponse(BaseModel):
    mode: str = "exact_vcp"
    reconstructed_meaning: str
    items: list[dict]
    decoded: list[dict] = Field(default_factory=list)
    verification_rate: float
    all_verified: bool
    elapsed_ms: float
    ecc_success: bool = True
    ecc_errors_fixed: list[int] = Field(default_factory=list)
    payload_bytes: list[int] = Field(default_factory=list)
    context_epoch_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    modalities: list[str] = Field(default=["image", "text", "audio"])


class SearchResponse(BaseModel):
    results: list[dict]
    elapsed_ms: float


class StatusResponse(BaseModel):
    indices: dict
    total_items: int
    device: str
    stealth_models: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest, _: None = Depends(require_api_token)):
    t0 = time.perf_counter()

    # Determine mode and diversity_mode
    engine_mode = "exact_vcp"
    diversity_mode = req.diversity_mode
    if req.mode in ("best", "round_robin", "balanced"):
        diversity_mode = req.mode
        engine_mode = "exact_vcp"
    elif req.mode in ("exact_vcp", "dssc"):
        engine_mode = req.mode
    else:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")

    session_key: Optional[bytes] = None
    if engine_mode == "dssc":
        if not req.session_key_hex:
            raise HTTPException(
                status_code=400,
                detail="session_key is required for DSSC mode. Provide session_key_hex as a hex string.",
            )
        try:
            session_key = bytes.fromhex(req.session_key_hex)
        except ValueError:
            raise HTTPException(status_code=400, detail="session_key_hex is not valid hex.")

    context_manager = None
    if req.use_dynamic_context:
        context_manager = _build_context_manager(req.context_bucket_seconds)

    try:
        engine = _get_engine()
        result = engine.encode(
            message=req.message,
            mode=engine_mode,
            session_key=session_key,
            modalities=req.modalities,
            use_ecc=req.use_ecc,
            ecc_parity_bytes=req.ecc_parity_bytes,
            diversity_mode=diversity_mode,
            context_manager=context_manager,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    encoded_items: list[dict] = []
    media_seq_items: list[dict] = []
    chunks: list[str] = []
    modality_breakdown: dict[str, int] = {}

    if result.exact_vcp_result is not None:
        vcp = result.exact_vcp_result
        chunks = [c.original for c in vcp.chunks]
        modality_breakdown = vcp.modality_breakdown
        for enc in vcp.encoded:
            fpath = enc.file_path or (enc.media.file_path if enc.media else "")
            encoded_items.append(
                {
                    "media_id": enc.media.id,
                    "modality": enc.media.modality,
                    "score": round(enc.media.normalized_score, 4),
                    "content": enc.media.content[:120],
                    "file_path": fpath,
                    "gdrive_path": enc.media.gdrive_path if enc.media else "",
                    "gdrive_url": enc.media.gdrive_url if enc.media else "",
                    "payload_byte": enc.payload_byte,
                    "cluster_id": enc.cluster_id,
                }
            )
        for item in vcp.media_sequence:
            media_seq_items.append(
                {
                    "id": item.id,
                    "media_id": item.id,
                    "modality": item.modality,
                    "content": item.content[:120],
                    "score": round(item.score, 4),
                    "file_path": item.file_path or "",
                    "gdrive_path": item.gdrive_path or "",
                    "gdrive_url": item.gdrive_url or "",
                }
            )
        context_info = dict(vcp.context_info)
    elif result.dssc_result is not None:
        dssc = result.dssc_result
        modality_breakdown = {}
        for carrier in dssc.encoded_carriers:
            item = engine.index.get_by_id(carrier.media_id)
            mod = item.modality if item else "unknown"
            modality_breakdown[mod] = modality_breakdown.get(mod, 0) + 1
            encoded_items.append(
                {
                    "media_id": carrier.media_id,
                    "modality": mod,
                    "score": 1.0,
                    "content": (item.content[:120] if item else ""),
                    "file_path": (item.file_path or "" if item else ""),
                    "gdrive_path": (item.gdrive_path or "" if item else ""),
                    "gdrive_url": (item.gdrive_url or "" if item else ""),
                    "payload_byte": carrier.symbol,
                    "cluster_id": None,
                }
            )
            media_seq_items.append(
                {
                    "id": carrier.media_id,
                    "media_id": carrier.media_id,
                    "modality": mod,
                    "content": (item.content[:120] if item else ""),
                    "score": 1.0,
                    "file_path": (item.file_path or "" if item else ""),
                    "gdrive_path": (item.gdrive_path or "" if item else ""),
                    "gdrive_url": (item.gdrive_url or "" if item else ""),
                }
            )
        context_info = result.context_info
    else:
        context_info = result.context_info

    if req.use_dynamic_context and "context_mode" not in context_info:
        context_info["context_mode"] = (
            "keyed" if _context_secret_from_env() else "obfuscation"
        )

    return EncodeResponse(
        mode=result.mode,
        media_ids=result.media_ids,
        carrier_count=result.carrier_count,
        chunks=chunks,
        encoded=encoded_items,
        media_sequence=media_seq_items,
        modality_breakdown=modality_breakdown,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        bits_per_carrier=result.bits_per_carrier,
        ecc_parity_bytes=result.ecc_parity_bytes,
        payload_bytes=result.payload_bytes,
        context_info=context_info,
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest, _: None = Depends(require_api_token)):
    t0 = time.perf_counter()

    engine_mode = req.mode or "exact_vcp"
    session_key: Optional[bytes] = None
    if engine_mode == "dssc":
        if not req.session_key_hex:
            raise HTTPException(
                status_code=400,
                detail="session_key is required for DSSC mode. Provide session_key_hex as a hex string.",
            )
        try:
            session_key = bytes.fromhex(req.session_key_hex)
        except ValueError:
            raise HTTPException(status_code=400, detail="session_key_hex is not valid hex.")

    context_manager = None
    if req.use_dynamic_context:
        context_manager = _build_context_manager(req.context_bucket_seconds)

    try:
        engine = _get_engine()
        result = engine.decode(
            media_ids=req.media_ids,
            mode=engine_mode,
            session_key=session_key,
            modalities=req.modalities,
            use_ecc=req.use_ecc,
            ecc_parity_bytes=req.ecc_parity_bytes,
            context_manager=context_manager,
            context_epoch_hint=req.context_epoch_hint,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items: list[dict] = []
    if result.exact_vcp_result is not None:
        vcp = result.exact_vcp_result
        for d in vcp.decoded:
            file_path = d.file_path or ""
            item = engine.index.get_by_id(d.media_id)
            if not file_path and d.verified and item:
                file_path = item.file_path or ""
            items.append(
                {
                    "media_id": d.media_id,
                    "modality": d.modality,
                    "content": d.content[:200],
                    "file_path": file_path,
                    "gdrive_path": item.gdrive_path if item else "",
                    "gdrive_url": item.gdrive_url if item else "",
                    "verified": d.verified,
                    "payload_byte": d.payload_byte,
                    "cluster_id": d.cluster_id,
                }
            )
        ecc_success = vcp.ecc_success
        ecc_fixed = vcp.ecc_errors_fixed
        epoch_id = vcp.context_epoch_id
        payload_bytes = vcp.payload_symbols
    elif result.dssc_result is not None:
        dssc = result.dssc_result
        for mid in dssc.media_ids:
            item = engine.index.get_by_id(mid)
            items.append(
                {
                    "media_id": mid,
                    "modality": item.modality if item else "unknown",
                    "content": item.content[:200] if item else "",
                    "file_path": item.file_path or "" if item else "",
                    "gdrive_path": item.gdrive_path or "" if item else "",
                    "gdrive_url": item.gdrive_url or "" if item else "",
                    "verified": item is not None,
                    "payload_byte": None,
                    "cluster_id": None,
                }
            )
        ecc_success = dssc.success
        ecc_fixed = dssc.ecc_fixed_errors
        epoch_id = None
        payload_bytes = []
    else:
        ecc_success = result.success
        ecc_fixed = result.ecc_fixed_errors
        epoch_id = None
        payload_bytes = []

    return DecodeResponse(
        mode=result.mode,
        reconstructed_meaning=result.reconstructed_message or "",
        items=items,
        decoded=items,
        verification_rate=result.verification_rate,
        all_verified=all(i["verified"] for i in items) if items else True,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        ecc_success=ecc_success,
        ecc_errors_fixed=ecc_fixed,
        payload_bytes=payload_bytes,
        context_epoch_id=epoch_id,
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    encoder = _get_encoder()
    results = encoder.index.search(req.query, k=req.k, modalities=req.modalities)

    items = []
    for r in results:
        items.append(
            {
                "id": r.id,
                "modality": r.modality,
                "score": round(r.normalized_score, 4),
                "content": r.content[:200],
            }
        )

    return SearchResponse(
        results=items,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/api/doctor")
def doctor(_: None = Depends(require_api_token)):
    """Full runtime diagnostics (same as `dcass doctor` CLI)."""
    from src.diagnostics import run_doctor

    report = run_doctor()
    project_root = Path(__file__).resolve().parent.parent.parent
    return report.to_public_dict(project_root)


@app.get("/api/status", response_model=StatusResponse)
def status():
    import torch

    global _index_counts_cache
    indices_path = resolve_indices_base_path()
    models_path = Path(__file__).parent.parent.parent / "storage" / "models"

    index_info = {}
    total = 0

    # Cache counts keyed on (path, size, mtime) so repeated status calls do
    # not re-read every FAISS index from disk.
    cache_key = {}
    for mod in ["image", "text", "audio"]:
        idx_file = indices_path / f"{mod}.index"
        if idx_file.exists():
            st = idx_file.stat()
            cache_key[mod] = (str(idx_file), st.st_size, st.st_mtime)
        else:
            cache_key[mod] = None

    if _index_counts_cache is None or _index_counts_cache.get("key") != cache_key:
        counts = {}
        for mod in ["image", "text", "audio"]:
            if cache_key[mod] is None:
                counts[mod] = None
                continue
            try:
                import faiss

                idx = faiss.read_index(cache_key[mod][0])
                counts[mod] = idx.ntotal
            except Exception as e:
                counts[mod] = f"error: {e}"
        _index_counts_cache = {"key": cache_key, "counts": counts}
    else:
        counts = _index_counts_cache["counts"]

    for mod in ["image", "text", "audio"]:
        c = counts.get(mod)
        if c is None:
            index_info[mod] = {"status": "missing"}
        elif isinstance(c, int):
            total += c
            index_info[mod] = {"status": "ok", "count": c}
        else:
            index_info[mod] = {"status": "error", "error": str(c)}

    stealth = {
        "gan_checkpoint": (models_path / "gan_generator.pt").exists()
        or (models_path / "gan" / "final.pt").exists(),
        "rl_checkpoint": (models_path / "rl_agent.pt").exists()
        or (models_path / "rl" / "ppo_agent_final.pt").exists(),
        "voronoi_codebook": (indices_path / "voronoi_codebook.npz").exists(),
    }

    return StatusResponse(
        indices=index_info,
        total_items=total,
        device="cuda" if torch.cuda.is_available() else "cpu",
        stealth_models=stealth,
    )


@app.get("/api/ready")
def ready():
    """Check if the server is ready to process requests."""
    is_ready = _ready or (_engine is not None and _engine.is_loaded())
    return {
        "ready": is_ready,
        "initializing": _initializing,
        "encoder_loaded": _engine is not None and _engine.is_loaded(),
        "decoder_loaded": _engine is not None and _engine.is_loaded(),
    }


@app.get("/api/benchmark/latest")
def benchmark_latest():
    results_dir = (
        Path(__file__).parent.parent.parent / "storage" / "data" / "benchmarks" / "results"
    )
    if not results_dir.exists():
        return {"available": False}

    files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not files:
        return {"available": False}

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"available": True, "filename": files[0].name, "data": data}


@app.get("/api/wire/packets")
def get_wire_packets():
    """List all packets in shared_channel directory."""
    shared_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"

    if not shared_dir.exists():
        return {
            "packets": [],
            "count": 0,
            "error": "shared_channel directory not found",
        }

    packets = []
    try:
        for f in sorted(shared_dir.glob("*.json")):
            # Skip manifest file
            if f.name.startswith("_"):
                continue

            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    data["filename"] = f.name
                    packets.append(data)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
    except Exception as e:
        return {"packets": [], "count": 0, "error": str(e)}

    return {"packets": packets, "count": len(packets)}


@app.delete("/api/wire/packets")
def clear_wire_packets(_: None = Depends(require_api_token)):
    """Clear all packets from shared_channel directory."""
    shared_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    shared_dir.mkdir(parents=True, exist_ok=True)

    deleted_count = 0
    try:
        for f in shared_dir.glob("*.json"):
            # Keep sender-side control files (e.g. _manifest.json), consistent
            # with the GET endpoint which skips them.
            if f.name.startswith("_"):
                continue
            f.unlink()
            deleted_count += 1
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "deleted": deleted_count}


class TransmitRequest(BaseModel):
    media_ids: list[str]
    mode: Literal["static", "rl", "gan", "auto"] = "static"
    base_delay: float = 1.0  # Reduced default for faster demo
    num_channels: int = 3
    message: str = ""
    speed_multiplier: float = 1.0  # 1.0 = real-time, 2.0 = 2x faster, etc.


def _transmit_packets_sync(
    schedule: dict,
    shared_dir: Path,
    message: str,
    speed_multiplier: float = 1.0,
):
    """
    Synchronous function to transmit packets with real delays.
    Runs in a background thread.
    """
    global _transmission_active, _transmission_progress, _transmission_stop_requested

    items = schedule["items"]
    delays = schedule["delays"]
    channels = schedule["channels"]
    mode_used = schedule["mode_used"]

    with _transmission_lock:
        _transmission_active = True
        _transmission_stop_requested = False
        _transmission_progress = {
            "current": 0,
            "total": len([i for i in items if i is not None]),
            "status": "transmitting",
        }

    try:
        # Write manifest
        manifest = {
            "message": message,
            "mode_requested": schedule.get("mode_requested", mode_used),
            "mode_used": mode_used,
            "total_items": len(items),
            "total_delay_seconds": round(sum(delays), 2),
            "timestamp": time.time(),
        }
        with open(shared_dir / "_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Transmit packets with real delays
        packet_count = 0
        for idx, (media_id, delay, channel) in enumerate(zip(items, delays, channels)):
            with _transmission_lock:
                if _transmission_stop_requested:
                    break

            if media_id is None:
                # Noise gap - just wait
                actual_delay = delay / speed_multiplier
                if actual_delay > 0:
                    time.sleep(actual_delay)
                continue

            # Write packet (media_id sanitized against path traversal)
            packet = {
                "media_id": media_id,
                "channel_id": channel,
                "sequence_number": idx,
                "delay_seconds": round(delay, 3),
                "timestamp": time.time(),
                "mode_used": mode_used,
            }

            try:
                path = sanitize_packet_filename(shared_dir, media_id, channel, idx)
            except ValueError as e:
                raise RuntimeError(f"refusing unsafe packet write: {e}") from e
            with open(path, "w") as f:
                json.dump(packet, f, indent=2)

            packet_count += 1

            # Update progress
            with _transmission_lock:
                _transmission_progress["current"] = packet_count

            # Apply delay before next packet (scaled by speed_multiplier),
            # in short slices so a stop request is honored promptly.
            actual_delay = delay / speed_multiplier
            if actual_delay > 0 and idx < len(items) - 1:
                remaining = actual_delay
                while remaining > 0:
                    with _transmission_lock:
                        if _transmission_stop_requested:
                            break
                    slice_sleep = min(0.25, remaining)
                    time.sleep(slice_sleep)
                    remaining -= slice_sleep

        # Mark as complete
        with _transmission_lock:
            _transmission_progress["status"] = (
                "stopped" if _transmission_stop_requested else "complete"
            )
            _transmission_active = False

    except Exception as e:
        with _transmission_lock:
            _transmission_progress["status"] = f"error: {str(e)}"
            _transmission_active = False
        raise


@app.post("/api/transmit")
def transmit_sequence(
    req: TransmitRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_token),
):
    """
    Start transmitting a media sequence through the shared channel.

    This starts a background task that writes packets with real delays,
    simulating realistic transmission timing.
    """
    global _transmission_active, _transmission_progress

    try:
        validate_transmit_media_ids(req.media_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Atomically claim the transmitter so concurrent requests cannot double-start
    with _transmission_lock:
        if _transmission_active:
            raise HTTPException(
                status_code=409,
                detail="Transmission already in progress. Wait for it to complete or check /api/transmit/status",
            )
        _transmission_active = True
        _transmission_progress = {"current": 0, "total": 0, "status": "starting"}

    try:
        from src.stealth.stealth_scheduler import StealthScheduler

        # Initialize scheduler
        scheduler = StealthScheduler(
            num_channels=req.num_channels,
            device="cpu",
            profile="casual",
        )

        # Get schedule
        schedule = scheduler.schedule(
            media_ids=req.media_ids,
            mode=req.mode,
            base_delay=req.base_delay,
        )

        # Prepare shared_channel directory
        shared_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
        shared_dir.mkdir(parents=True, exist_ok=True)

        # Calculate estimated time
        total_delay = sum(schedule["delays"]) / req.speed_multiplier

        # Start background transmission in a thread
        thread = threading.Thread(
            target=_transmit_packets_sync,
            args=(schedule, shared_dir, req.message, req.speed_multiplier),
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "status": "started",
            "total_packets": len([i for i in schedule["items"] if i is not None]),
            "mode_used": schedule["mode_used"],
            "estimated_duration_seconds": round(total_delay, 2),
            "speed_multiplier": req.speed_multiplier,
            "message": "Transmission started in background. Poll /api/transmit/status for progress.",
        }

    except Exception as e:
        # Release the claim if scheduling failed before the worker took over.
        with _transmission_lock:
            _transmission_active = False
            _transmission_progress = {"current": 0, "total": 0, "status": "idle"}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transmit/status")
def get_transmission_status():
    """Get the current transmission status."""
    return {
        "active": _transmission_active,
        **_transmission_progress,
    }


@app.post("/api/transmit/stop")
def stop_transmission(_: None = Depends(require_api_token)):
    """Stop the current transmission (best effort)."""
    global _transmission_active, _transmission_progress, _transmission_stop_requested

    with _transmission_lock:
        if _transmission_active:
            _transmission_stop_requested = True
            _transmission_progress["status"] = "stopping"
            return {"success": True, "message": "Transmission stop requested"}
        else:
            return {"success": False, "message": "No active transmission"}

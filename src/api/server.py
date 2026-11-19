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
import uuid
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.corpus.index.unified_index import resolve_indices_base_path
from src.api import auth as _auth
from src.api.auth import AuthenticatedUser, get_current_user


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SHARED_CHANNEL_ROOT = _PROJECT_ROOT / "storage" / "shared_channel"
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
    # Allow known frontend origins, and regex-match localhost, 127.0.0.1, and private LAN IPs
    allow_origins=[
        o.strip()
        for o in os.environ.get("DCASS_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
        if o.strip()
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:[0-9]+)?$",
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

    When the env var is unset in development, auth is skipped (local/dev).
    Under DCASS_ENV=production an unset DCASS_API_TOKEN is a configuration
    error, not a decision: the request fails closed with 503, because the
    alternative is serving sensitive routes to anonymous callers silently.
    """
    expected = os.environ.get("DCASS_API_TOKEN")
    if expected:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or credentials.credentials != expected
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return
    if _auth._is_production():
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured: set DCASS_API_TOKEN (or "
                "stop claiming DCASS_ENV=production). Refusing to serve open."
            ),
        )


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


def sanitize_packet_filename(
    shared_dir: Path,
    media_id: str,
    channel: int,
    idx: int,
    session_id: Optional[str] = None,
) -> Path:
    """
    Build a packet path under *shared_dir*, rejecting path traversal.

    When `session_id` is given (Phase F), it is prepended to the filename
    so parallel sessions to the same recipient don't collide. The
    session_id itself is validated the same way media_id is — no path
    separators, no `..` — before being used in the filename.

    Raises ValueError when media_id (or session_id) contains separators,
    `..`, or would resolve outside shared_dir.
    """
    if not media_id or not isinstance(media_id, str):
        raise ValueError("media_id must be a non-empty string")
    if media_id in (".", "..") or ".." in media_id:
        raise ValueError(f"invalid media_id: {media_id!r}")
    if "/" in media_id or "\\" in media_id or media_id != Path(media_id).name:
        raise ValueError(f"invalid media_id (path characters): {media_id!r}")

    if session_id is not None:
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id in (".", "..")
            or ".." in session_id
            or "/" in session_id
            or "\\" in session_id
            or session_id != Path(session_id).name
        ):
            raise ValueError(f"invalid session_id: {session_id!r}")

    shared_resolved = shared_dir.resolve()
    # Use opaque filename without sequence_number or channel_id in the filename
    # to prevent wire metadata leakage. All metadata is inside the JSON content.
    if session_id:
        filename = f"{session_id}__{media_id}.json"
    else:
        filename = f"{media_id}.json"
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
# Per-session tracking instead of global lock
_transmission_sessions: dict[str, dict] = {}  # session_id -> {active, progress, stop_requested}
_transmission_sessions_lock = threading.Lock()

# Maximum number of transmissions admitted at once. The shared engine and
# the shared_channel directory are both shared resources, so admission is
# bounded rather than unlimited: the sessions themselves are independent
# (per-session packet filenames and per-session manifests), but each one
# holds a scheduler and writes into a common directory.
MAX_CONCURRENT_TRANSMISSIONS = int(
    os.environ.get("DCASS_MAX_CONCURRENT_TRANSMISSIONS", "4")
)

# Serializes engine encode/decode across threads. The engine is a
# singleton, and the encoder/decoder hold mutable scratch state, so
# concurrent calls would interleave. The transmission threads themselves
# only sleep and write files, so this lock is held briefly.
_engine_work_lock = threading.Lock()

def _prune_transmission_sessions(max_age_seconds: float = 300.0, max_entries: int = 256) -> None:
    """Evict old/excess INACTIVE transmission sessions.

    Caller must hold `_transmission_sessions_lock`. Entries with
    `active=True` are never removed. Inactive entries are evicted once their
    `finished_at` is older than *max_age_seconds*, and — if the map still
    exceeds *max_entries* — the oldest inactive entries are evicted first,
    so a long-running server cannot grow the map without bound.
    """
    now = time.time()
    for sid, state in [
        (sid, state)
        for sid, state in _transmission_sessions.items()
        if not state.get("active")
    ]:
        finished_at = state.get("finished_at")
        if finished_at is not None and (now - float(finished_at)) > max_age_seconds:
            _transmission_sessions.pop(sid, None)
    if len(_transmission_sessions) > max_entries:
        inactive = sorted(
            (
                sid
                for sid, state in _transmission_sessions.items()
                if not state.get("active")
            ),
            key=lambda sid: float(_transmission_sessions[sid].get("finished_at") or 0.0),
        )
        while inactive and len(_transmission_sessions) > max_entries:
            _transmission_sessions.pop(inactive.pop(0), None)


def _get_session_state(session_id: Optional[str]) -> dict:
    """Get or create transmission session state."""
    if session_id is None:
        session_id = "default"
    with _transmission_sessions_lock:
        _prune_transmission_sessions()
        if session_id not in _transmission_sessions:
            _transmission_sessions[session_id] = {
                "active": False,
                "stop_requested": False,
                "progress": {"current": 0, "total": 0, "status": "idle"},
                "owner": None,
            }
        return _transmission_sessions[session_id]


def _cleanup_session(session_id: str) -> None:
    """Remove session state after completion."""
    with _transmission_sessions_lock:
        _transmission_sessions.pop(session_id, None)


def _claim_transmission(session_id: str, owner: Optional[int] = None) -> bool:
    """Atomically claim a transmission slot for the given session.

    Sessions are independent: each writes packets under its own
    `<session_id>__<media_id>.json` name and its own
    `_manifest_<session_id>.json`, so several can run against the same
    recipient directory without colliding.

    Admission is nevertheless bounded by MAX_CONCURRENT_TRANSMISSIONS.
    Beyond the bound the claim fails and the endpoint answers 429, so a
    burst of clients cannot spawn an unbounded number of scheduler threads
    writing into one directory.

    `owner` stamps the authenticated user so /api/transmit/stop can scope
    its effects to the caller's own sessions.

    Returns True if claimed, False if the concurrency bound is reached.
    """
    with _transmission_sessions_lock:
        # Evict stale/finished sessions before counting admissions so the map
        # stays bounded and finished entries do not count against the cap.
        _prune_transmission_sessions()
        active = sum(1 for s in _transmission_sessions.values() if s["active"])
        if active >= MAX_CONCURRENT_TRANSMISSIONS:
            return False
        # Create or update our session state
        if session_id not in _transmission_sessions:
            _transmission_sessions[session_id] = {
                "active": False,
                "stop_requested": False,
                "progress": {"current": 0, "total": 0, "status": "idle"},
                "owner": owner,
            }
        _transmission_sessions[session_id]["active"] = True
        _transmission_sessions[session_id]["owner"] = owner
        return True


def _active_transmission_count() -> int:
    """Number of transmissions currently running."""
    with _transmission_sessions_lock:
        return sum(1 for s in _transmission_sessions.values() if s["active"])

# Cached index counts for /api/status (avoids re-reading FAISS files per call)
_index_counts_cache: Optional[dict] = None


def _get_engine():
    global _engine, _ready
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from src.engine.semantic_engine import SemanticEngine

                engine = SemanticEngine()
                engine.load()
                _engine = engine
                _ready = True
    return _engine


def _get_encoder():
    return _get_engine()._exact_encoder


def _get_decoder():
    return _get_engine()._exact_decoder


@app.on_event("startup")
def _startup_init_auth() -> None:
    """Initialize auth DB, seed demo users, and trigger engine warmup on API startup."""
    _auth.init_db()
    # Demo accounts are a local-dev convenience only: never seed known
    # username/password pairs when DCASS_ENV normalizes to "production".
    if not _auth._is_production():
        _auth.seed_demo_users()
    threading.Thread(target=warmup, daemon=True).start()


def _recipient_shared_dir(recipient_user_id: int) -> Path:
    """Per-recipient subdirectory of the shared channel."""
    base = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    sub = base / str(recipient_user_id)
    sub.mkdir(parents=True, exist_ok=True)
    return sub


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
    cover_story: Optional[str] = None
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
    session_key_hex: Optional[str] = Field(default=None, exclude=True)


class DecodeRequest(BaseModel):
    media_ids: list[Optional[str]]
    mode: Optional[str] = "exact_vcp"
    session_key_hex: Optional[str] = None
    modalities: Optional[list[str]] = None
    use_ecc: bool = True
    ecc_parity_bytes: int = Field(default=8, ge=1, le=64)
    use_dynamic_context: bool = False
    context_bucket_seconds: int = Field(default=3600, ge=1)
    context_epoch_hint: Optional[str] = None
    diversity_mode: Literal["best", "round_robin", "balanced"] = "best"
    erasure_positions: Optional[list[int]] = None


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
    erasure_positions: list[int] = Field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Auth routes (Phase F) — minimal demo auth, see src/api/auth.py and
# docs/GAN_RL_INTEGRATION_PLAN.md Phase F.6 for the deployment caveats.
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthUserInfo(BaseModel):
    id: int
    username: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserInfo


class UsersListResponse(BaseModel):
    users: list[AuthUserInfo]


@app.post("/api/auth/register", response_model=LoginResponse, status_code=201)
def auth_register(req: RegisterRequest):
    try:
        user = _auth.create_user(req.username, req.password)
    except ValueError as e:
        # Covers both the length check and the uniqueness collision.
        raise HTTPException(status_code=409, detail=str(e))
    token = _auth.create_access_token(user.id, user.username)
    return LoginResponse(
        access_token=token,
        user=AuthUserInfo(id=user.id, username=user.username),
    )


@app.post("/api/auth/login", response_model=LoginResponse)
def auth_login(req: LoginRequest):
    user = _auth.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _auth.create_access_token(user.id, user.username)
    return LoginResponse(
        access_token=token,
        user=AuthUserInfo(id=user.id, username=user.username),
    )


@app.get("/api/auth/me", response_model=AuthUserInfo)
def auth_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    return AuthUserInfo(id=current_user.id, username=current_user.username)


@app.get("/api/users", response_model=UsersListResponse)
def list_users(
    exclude_current: bool = False,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """List users (authentication required).

    Usernames are account identifiers; an unauthenticated caller must not
    be able to enumerate the membership of the deployment, in either env.
    """
    if exclude_current:
        users = _auth.list_users_excluding(current_user.id)
    else:
        users = _auth.list_all_users()
    return UsersListResponse(
        users=[AuthUserInfo(id=u.id, username=u.username) for u in users]
    )


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
                detail="session_key_hex is required for DSSC mode. Provide session_key_hex as a hex string.",
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
        # The engine is a singleton and its encoder holds mutable scratch
        # state, so concurrent calls are serialized here.
        with _engine_work_lock:
            result = engine.encode(
                message=req.message,
                mode=engine_mode,
                session_key=session_key,
                modalities=req.modalities,
                use_ecc=req.use_ecc,
                ecc_parity_bytes=req.ecc_parity_bytes,
                diversity_mode=diversity_mode,
                context_manager=context_manager,
                cover_story=req.cover_story,
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
            # Ensure file_path is always a string (empty string if not available)
            fpath = (enc.file_path or "") or (enc.media.file_path if enc.media and enc.media.file_path else "")
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

    # Include session_key in response for DSSC mode only if server generated it
    session_key_hex_out = session_key.hex() if (session_key and not req.session_key_hex) else None

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
        session_key_hex=session_key_hex_out,
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest, _: None = Depends(require_api_token)):
    t0 = time.perf_counter()

    # Determine mode and diversity_mode — mirrors the /api/encode
    # normalization so "best"/"round_robin"/"balanced" select exact_vcp with
    # that diversity mode instead of failing as an unknown engine mode.
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

    effective_erasures = sorted(
        {
            *(int(p) for p in (req.erasure_positions or []) if 0 <= int(p) < len(req.media_ids)),
            *(i for i, mid in enumerate(req.media_ids) if mid is None or mid == ""),
        }
    )

    try:
        engine = _get_engine()
        # Serialized for the same reason as encode: the decoder is shared
        # mutable state.
        with _engine_work_lock:
            result = engine.decode(
                media_ids=req.media_ids,
                mode=engine_mode,
                session_key=session_key,
                modalities=req.modalities,
                use_ecc=req.use_ecc,
                ecc_parity_bytes=req.ecc_parity_bytes,
                context_manager=context_manager,
                context_epoch_hint=req.context_epoch_hint,
                diversity_mode=diversity_mode,
                erasure_positions=effective_erasures or None,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items: list[dict] = []
    if result.exact_vcp_result is not None:
        vcp = result.exact_vcp_result
        for d in vcp.decoded:
            file_path = d.file_path or ""
            item = engine.index.get_by_id(d.media_id) if d.verified else None
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
        erased_set = set(effective_erasures)
        for idx, mid in enumerate(dssc.media_ids):
            if idx in erased_set or mid == "[DROPPED]":
                items.append(
                    {
                        "media_id": "[DROPPED]",
                        "modality": "text",
                        "content": (
                            "[DROPPED PACKET — RECOVERED BY RS-ECC]"
                            if dssc.success
                            else "[DROPPED PACKET]"
                        ),
                        "file_path": "",
                        "gdrive_path": "",
                        "gdrive_url": "",
                        "verified": False,
                        "payload_byte": None,
                        "cluster_id": None,
                    }
                )
                continue
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
        payload_bytes = dssc.raw_payload_bytes
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
        erasure_positions=effective_erasures,
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
    global _engine
    if _engine is None and not _initializing and not _ready:
        threading.Thread(target=warmup, daemon=True).start()

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
def get_wire_packets(
    recipient_username: Optional[str] = None,
    session_id: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """List packets routed to an inbox.

    If recipient_username is specified, returns that recipient's inbox
    (allowing the transmitter to monitor the live channel wire).
    """
    base_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    target_id = current_user.id
    if recipient_username:
        recip = _auth.get_user_by_username(recipient_username)
        if recip is not None:
            target_id = recip.id

    shared_dir = base_dir / str(target_id)

    if not base_dir.exists() or not shared_dir.exists():
        return {
            "packets": [],
            "count": 0,
        }

    packets = []
    try:
        json_files = [f for f in shared_dir.glob("*.json") if not f.name.startswith("_")]
        if session_id:
            json_files = [f for f in json_files if f.name.startswith(f"{session_id}__")]
        json_files.sort(key=lambda p: p.stat().st_mtime)
        # When the caller asks to monitor ANOTHER user's inbox (transmitter
        # view), only packets this caller actually sent may be returned —
        # otherwise any user could snoop every session in someone else's
        # channel directory.
        cross_user_snoop = target_id != current_user.id
        for f in json_files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    data["filename"] = f.name
                    data.setdefault("mode_used", "exact_vcp")
                    data.setdefault("delay_seconds", 0.0)
                    data.setdefault("sequence_number", 1)
                    data.setdefault("channel_id", 0)
                    data.setdefault("media_id", "carrier")
                    data.setdefault("timestamp", f.stat().st_mtime)
                    if session_id and data.get("session_id") and data.get("session_id") != session_id:
                        continue
                    if cross_user_snoop and data.get("sender_username") != current_user.username:
                        continue
                    packets.append(data)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
        # Sort by sequence_number if present
        packets.sort(key=lambda p: p.get("sequence_number", 0))
    except Exception as e:
        return {"packets": [], "count": 0, "error": str(e)}

    return {"packets": packets, "count": len(packets)}


@app.delete("/api/wire/packets")
def clear_wire_packets(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Clear the authenticated user's own inbox.

    Deletion is scoped to `storage/shared_channel/<current_user.id>/`;
    other users' packets are never touched.
    """
    base_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    shared_dir = base_dir / str(current_user.id)

    if not base_dir.exists() or not shared_dir.exists():
        return {"success": True, "deleted": 0}

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
    engine_mode: Literal["exact_vcp", "dssc"] = "exact_vcp"
    diversity_mode: Literal["best", "round_robin", "balanced"] = "best"
    session_key_hex: Optional[str] = None
    base_delay: float = 1.0  # Reduced default for faster demo
    num_channels: int = 3
    message: str = ""
    speed_multiplier: float = Field(default=1.0, gt=0.0)  # 1.0 = real-time, 2.0 = 2x faster, etc.
    # RS parity bytes recorded in the session manifest (consumed by decoders).
    ecc_parity_bytes: int = Field(default=8, ge=1, le=64)
    # Phase F — 1-to-1 routing: packets land in storage/shared_channel/<recipient_id>/
    recipient_username: Optional[str] = Field(default=None, min_length=1)


def _transmit_packets_sync(
    schedule: dict,
    shared_dir: Path,
    message: str,
    speed_multiplier: float = 1.0,
    sender_id: Optional[int] = None,
    sender_username: Optional[str] = None,
    recipient_id: Optional[int] = None,
    recipient_username: Optional[str] = None,
    session_id: Optional[str] = None,
    engine_mode: str = "exact_vcp",
    diversity_mode: str = "round_robin",
    session_key_hex: Optional[str] = None,
    ecc_parity_bytes: int = 8,
):
    """
    Synchronous function to transmit packets with real delays.
    Runs in a background thread.

    When sender_id/recipient_id/session_id are given (Phase F routed path),
    the manifest is written as `_manifest_<session_id>.json` so multiple
    sessions can coexist in the same recipient directory, and the packet
    filenames are prefixed with the session_id for the same reason.
    """
    session_state = _get_session_state(session_id)

    items = schedule["items"]
    delays = schedule["delays"]
    channels = schedule["channels"]
    mode_used = schedule["mode_used"]

    with _transmission_sessions_lock:
        session_state["active"] = True
        session_state["stop_requested"] = False
        session_state["progress"] = {
            "current": 0,
            "total": len([i for i in items if i is not None]),
            "status": "transmitting",
        }

    try:
        # Write manifest. Phase F adds sender/recipient metadata and a
        # session-specific filename so one recipient directory can hold
        # many messages without clobbering a shared _manifest.json.
        manifest = {
            "message": message,
            "mode_requested": schedule.get("mode_requested", mode_used),
            "mode_used": mode_used,
            "engine_mode": engine_mode,
            "diversity_mode": diversity_mode,
            "session_key_hex": session_key_hex,
            "ecc_parity_bytes": ecc_parity_bytes,
            "sequence_base": 0,
            "total_items": len(items),
            "total_delay_seconds": round(sum(delays), 2),
            "timestamp": time.time(),
            "sender_id": sender_id,
            "sender_username": sender_username,
            "recipient_id": recipient_id,
            "recipient_username": recipient_username,
            "session_id": session_id,
        }
        manifest_name = (
            f"_manifest_{session_id}.json" if session_id else "_manifest.json"
        )
        # Atomic write (temp file + os.replace) so a concurrent reader never
        # sees a torn manifest.
        _atomic_write_json(shared_dir / manifest_name, manifest)

        # Transmit packets with real delays
        packet_count = 0
        for idx, (media_id, delay, channel) in enumerate(zip(items, delays, channels)):
            with _transmission_sessions_lock:
                if session_state["stop_requested"]:
                    break

            if media_id is None:
                # Noise gap - just wait
                actual_delay = delay / speed_multiplier
                if actual_delay > 0:
                    time.sleep(actual_delay)
                continue

            # Write packet (media_id sanitized against path traversal).
            # Phase F: tag the packet with its session_id and prefix the
            # filename with it so two sessions to the same recipient don't
            # overwrite each other's packets.
            packet = {
                "media_id": media_id,
                "channel_id": channel,
                "sequence_number": idx,
                "delay_seconds": round(delay, 3),
                "timestamp": time.time(),
                "mode_used": mode_used,
                "session_id": session_id,
                # Sender/recipient tagging (same fields as chat_send/wire_send)
                # so the sender can list/inspect their own transmitted packets
                # via /api/wire/packets and /api/media/packet.
                "sender_username": sender_username,
                "recipient_username": recipient_username,
            }

            try:
                path = sanitize_packet_filename(
                    shared_dir, media_id, channel, idx, session_id=session_id
                )
            except ValueError as e:
                raise RuntimeError(f"refusing unsafe packet write: {e}") from e
            _atomic_write_json(path, packet)

            packet_count += 1

            # Update progress
            with _transmission_sessions_lock:
                session_state["progress"]["current"] = packet_count

            # Apply delay before next packet (scaled by speed_multiplier),
            # in short slices so a stop request is honored promptly.
            actual_delay = delay / speed_multiplier
            if actual_delay > 0 and idx < len(items) - 1:
                remaining = actual_delay
                while remaining > 0:
                    with _transmission_sessions_lock:
                        if session_state["stop_requested"]:
                            break
                    slice_sleep = min(0.25, remaining)
                    time.sleep(slice_sleep)
                    remaining -= slice_sleep

        # Mark as complete
        with _transmission_sessions_lock:
            session_state["progress"]["status"] = (
                "stopped" if session_state["stop_requested"] else "complete"
            )
            session_state["active"] = False
            session_state["finished_at"] = time.time()

    except Exception as e:
        with _transmission_sessions_lock:
            session_state["progress"]["status"] = f"error: {str(e)}"
            session_state["active"] = False
            session_state["finished_at"] = time.time()
        raise


@app.post("/api/transmit")
def transmit_sequence(
    req: TransmitRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Start transmitting a media sequence through the shared channel.

    Phase F: the sender identity is the authenticated user; the recipient
    is `req.recipient_username`. Packets are written to
    `storage/shared_channel/<recipient_user_id>/` with a session_id
    prefix so parallel sessions to the same recipient don't collide.
    """
    try:
        validate_transmit_media_ids(req.media_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Resolve recipient. If omitted, default to another active peer.
    recipient_username = req.recipient_username
    if not recipient_username:
        default_peer = "alice" if current_user.username == "bob" else "bob"
        recipient = _auth.get_user_by_username(default_peer)
        if recipient is None:
            peers = _auth.list_users_excluding(current_user.id)
            recipient = peers[0] if peers else current_user
    else:
        recipient = _auth.get_user_by_username(recipient_username)
        if recipient is None:
            raise HTTPException(status_code=404, detail=f"recipient {recipient_username!r} does not exist")

    # Loopback transmission (recipient == current_user) is fully supported
    # for single-host demos, automated verification, and self-testing.

    session_id = uuid.uuid4().hex[:16]

    # Atomically claim a transmission slot for this session. Several
    # sessions may run at once, up to the concurrency bound.
    if not _claim_transmission(session_id, current_user.id):
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many transmissions in progress "
                f"({_active_transmission_count()} active, limit "
                f"{MAX_CONCURRENT_TRANSMISSIONS}). Retry shortly or check "
                "/api/transmit/status"
            ),
        )

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

        # Prepare per-recipient shared_channel subdirectory (Phase F routing).
        shared_dir = _recipient_shared_dir(recipient.id)

        # Calculate estimated time
        total_delay = sum(schedule["delays"]) / req.speed_multiplier

        # Start background transmission in a thread
        thread = threading.Thread(
            target=_transmit_packets_sync,
            kwargs=dict(
                schedule=schedule,
                shared_dir=shared_dir,
                message=req.message,
                speed_multiplier=req.speed_multiplier,
                sender_id=current_user.id,
                sender_username=current_user.username,
                recipient_id=recipient.id,
                recipient_username=recipient.username,
                session_id=session_id,
                engine_mode=req.engine_mode,
                diversity_mode=req.diversity_mode,
                session_key_hex=req.session_key_hex,
                ecc_parity_bytes=req.ecc_parity_bytes,
            ),
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "status": "started",
            "session_id": session_id,
            "sender": current_user.username,
            "recipient": recipient.username,
            "total_packets": len([i for i in schedule["items"] if i is not None]),
            "mode_used": schedule["mode_used"],
            "estimated_duration_seconds": round(total_delay, 2),
            "speed_multiplier": req.speed_multiplier,
            "message": "Transmission started in background. Poll /api/transmit/status for progress.",
        }

    except Exception as e:
        # Release the claim if scheduling failed before the worker took over.
        _cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transmit/status")
def get_transmission_status(
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Get the current transmission status for the caller's own sessions.

    The top-level keys are the first active session, kept for backward
    compatibility with clients that read `active`/`current`/`total`/
    `status` directly. `sessions` lists the caller's active sessions, and
    `active_count` and `max_concurrent` expose the admission bound.
    """
    with _transmission_sessions_lock:
        sessions = {
            session_id: {"active": state["active"], **state["progress"]}
            for session_id, state in _transmission_sessions.items()
            if state["active"] and state.get("owner") == current_user.id
        }
        if not sessions:
            return {
                "active": False,
                "current": 0,
                "total": 0,
                "status": "idle",
                "sessions": {},
                "active_count": 0,
                "max_concurrent": MAX_CONCURRENT_TRANSMISSIONS,
            }
        # Return the first active session at the top level (backward
        # compatibility) and list them all under `sessions`.
        first_session = next(iter(sessions.values()))
        return {
            "active": True,
            **first_session,
            "sessions": sessions,
            "active_count": len(sessions),
            "max_concurrent": MAX_CONCURRENT_TRANSMISSIONS,
        }


@app.post("/api/transmit/stop")
def stop_transmission(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Stop the caller's own active transmissions (best effort).

    `session_id` in the query string stops one session — the caller's
    only; another user's session id is indistinguishable from an unknown
    one (404). Without `session_id`, the caller's active sessions stop.
    """
    session_id = request.query_params.get("session_id")
    with _transmission_sessions_lock:
        if session_id:
            state = _transmission_sessions.get(session_id)
            if state is None or state.get("owner") != current_user.id:
                raise HTTPException(status_code=404, detail="unknown session")
            targets = [session_id]
        else:
            targets = [
                sid
                for sid, s in _transmission_sessions.items()
                if s["active"] and s.get("owner") == current_user.id
            ]
        for sid in targets:
            state = _transmission_sessions.get(sid)
            if state is not None and state["active"]:
                state["stop_requested"] = True
                state["progress"]["status"] = "stopping"
        if targets:
            return {
                "success": True,
                "message": f"Transmission stop requested for {len(targets)} session(s)",
            }
        return {"success": False, "message": "No active transmission"}


# ---------------------------------------------------------------------------
# Inbox (Phase F) — per-recipient message list + on-demand decode.
#
# Session_id is the primary key for a sent message: one session = one
# transmission = one manifest + N packets. Listing the inbox returns the
# manifests under storage/shared_channel/<current_user_id>/; decoding a
# session reassembles its packets (sorted by sequence_number) and runs
# the SemanticDecoder, caching the result in a sidecar JSON so repeat
# reads don't repay the decoder cost.
# ---------------------------------------------------------------------------


def _inbox_dir(user_id: int) -> Path:
    base = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    sub = base / str(user_id)
    sub.mkdir(parents=True, exist_ok=True)
    return sub


class InboxItem(BaseModel):
    session_id: str
    sender_username: Optional[str] = None
    sender_id: Optional[int] = None
    recipient_username: Optional[str] = None
    timestamp: float
    mode_used: Optional[str] = None
    total_items: int = 0
    received_packets: int = 0
    decoded: bool = False


class InboxResponse(BaseModel):
    items: list[InboxItem]


@app.get("/api/inbox", response_model=InboxResponse)
def inbox_list(current_user: AuthenticatedUser = Depends(get_current_user)):
    """List all sessions addressed to the current user, newest first."""
    inbox = _inbox_dir(current_user.id)
    items: list[InboxItem] = []
    for manifest_path in inbox.glob("_manifest_*.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        session_id = manifest.get("session_id") or manifest_path.stem.removeprefix("_manifest_")
        packet_count = sum(1 for _ in inbox.glob(f"{session_id}__*.json"))
        decoded_sidecar = inbox / f"_decoded_{session_id}.json"
        items.append(
            InboxItem(
                session_id=session_id,
                sender_username=manifest.get("sender_username"),
                sender_id=manifest.get("sender_id"),
                recipient_username=manifest.get("recipient_username"),
                timestamp=manifest.get("timestamp", 0.0),
                mode_used=manifest.get("mode_used"),
                total_items=manifest.get("total_items", 0),
                received_packets=packet_count,
                decoded=decoded_sidecar.exists(),
            )
        )
    items.sort(key=lambda i: i.timestamp, reverse=True)
    return InboxResponse(items=items)


class InboxDecodeResponse(BaseModel):
    session_id: str
    sender_username: Optional[str] = None
    recipient_username: Optional[str] = None
    mode_used: Optional[str] = None
    reconstructed_meaning: str = ""
    verification_rate: float = 1.0
    all_verified: bool = True
    ecc_success: bool = True
    ecc_errors_fixed: list[int] = Field(default_factory=list)
    items: list[dict] = Field(default_factory=list)
    cached: bool = False
    elapsed_ms: Optional[float] = None


def _validate_session_id(session_id: str) -> None:
    if (
        not session_id
        or not isinstance(session_id, str)
        or session_id in (".", "..")
        or ".." in session_id
        or "/" in session_id
        or "\\" in session_id
        or session_id != Path(session_id).name
    ):
        raise HTTPException(status_code=400, detail=f"invalid session_id: {session_id!r}")


@app.post("/api/inbox/{session_id}/decode", response_model=InboxDecodeResponse)
def inbox_decode(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Reassemble the packets for `session_id` (sorted by sequence_number)
    and run SemanticDecoder on them. First call decodes-and-caches; later
    calls return the cached result from a `_decoded_<session_id>.json`
    sidecar in the recipient's inbox directory.
    """
    _validate_session_id(session_id)
    inbox = _inbox_dir(current_user.id)
    manifest_path = inbox / f"_manifest_{session_id}.json"
    sidecar = inbox / f"_decoded_{session_id}.json"
    with _get_session_lock(session_id):
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="session not found in inbox")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if sidecar.exists():
            with open(sidecar, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["cached"] = True
            if "elapsed_ms" not in cached:
                cached["elapsed_ms"] = 0.0
            return InboxDecodeResponse(**cached)

        # Gather surviving packet files currently on disk and align by sequence_number.
        packets = []
        for pkt_path in inbox.glob(f"{session_id}__*.json"):
            try:
                with open(pkt_path, "r", encoding="utf-8") as f:
                    packets.append(json.load(f))
            except Exception:
                continue
        packets.sort(key=lambda p: p.get("sequence_number", 0))

        if not packets:
            raise HTTPException(status_code=409, detail="no packets available yet for this session")

        # Pre-decode disk fingerprint: if the packets on disk change between
        # this read and the sidecar writeback below, the stale decode result
        # must NOT be cached.
        disk_snapshot = (
            manifest_path.exists(),
            tuple(
                sorted(
                    ((p.get("sequence_number"), p.get("media_id")) for p in packets),
                    key=lambda t: (str(t[0]), str(t[1])),
                )
            ),
        )

    expected_total = int(manifest.get("total_items") or len(packets))
    seq_base = manifest.get("sequence_base")
    if seq_base is None:
        seqs = [int(p.get("sequence_number", 0)) for p in packets]
        if session_id.startswith("chat_") or (min(seqs) >= 1 and max(seqs) <= expected_total):
            seq_base = 1
        else:
            seq_base = 0

    slots: list[Optional[str]] = [None] * expected_total
    for pkt in packets:
        slot_idx = int(pkt.get("sequence_number", 0)) - int(seq_base)
        if 0 <= slot_idx < expected_total:
            slots[slot_idx] = pkt.get("media_id")

    erasure_positions = [i for i, mid in enumerate(slots) if not mid]
    ecc_parity_bytes = int(manifest.get("ecc_parity_bytes") or 8)

    engine_mode = manifest.get("engine_mode") or "exact_vcp"
    diversity_mode = manifest.get("diversity_mode") or "round_robin"
    session_key: Optional[bytes] = None
    if engine_mode == "dssc":
        session_key_hex = manifest.get("session_key_hex")
        if session_key_hex:
            try:
                session_key = bytes.fromhex(session_key_hex)
            except ValueError:
                pass

    t0 = time.perf_counter()
    try:
        engine = _get_engine()
        with _engine_work_lock:
            decode_result = engine.decode(
                media_ids=slots,
                mode=engine_mode,
                session_key=session_key,
                diversity_mode=diversity_mode,
                use_ecc=True,
                ecc_parity_bytes=ecc_parity_bytes,
                erasure_positions=erasure_positions or None,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"decode failed: {e}")
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    items_out: list[dict] = []
    if decode_result.exact_vcp_result is not None:
        vcp = decode_result.exact_vcp_result
        for d in vcp.decoded:
            items_out.append(
                {
                    "media_id": d.media_id,
                    "modality": d.modality,
                    "content": d.content[:200],
                    "verified": d.verified,
                }
            )
        ecc_success = vcp.ecc_success
        ecc_fixed = vcp.ecc_errors_fixed
    else:
        ecc_success = decode_result.success
        ecc_fixed = decode_result.ecc_fixed_errors
        for idx, mid in enumerate(slots):
            if mid is None:
                items_out.append(
                    {
                        "media_id": "[DROPPED]",
                        "modality": "text",
                        "content": "[DROPPED PACKET — RECOVERED BY RS-ECC]" if ecc_success else "[DROPPED PACKET]",
                        "verified": False,
                    }
                )
            else:
                item = engine.index.get_by_id(mid)
                items_out.append(
                    {
                        "media_id": mid,
                        "modality": item.modality if item else "unknown",
                        "content": (item.content[:200] if item else ""),
                        "verified": item is not None,
                    }
                )

    body = InboxDecodeResponse(
        session_id=session_id,
        sender_username=manifest.get("sender_username"),
        recipient_username=manifest.get("recipient_username"),
        mode_used=manifest.get("mode_used"),
        reconstructed_meaning=(decode_result.reconstructed_message or "") if ecc_success else "",
        verification_rate=decode_result.verification_rate,
        all_verified=all(i["verified"] for i in items_out) if items_out else True,
        ecc_success=ecc_success,
        ecc_errors_fixed=ecc_fixed,
        items=items_out,
        cached=False,
        elapsed_ms=elapsed_ms,
    )

    # Cache the result only when decoding succeeded AND the on-disk packets
    # are unchanged since the pre-decode snapshot (no concurrent drop/tamper/
    # delete happened mid-decode). The sidecar is written atomically under
    # the session lock.
    if ecc_success:
        with _get_session_lock(session_id):
            current_packets = []
            for pkt_path in inbox.glob(f"{session_id}__*.json"):
                try:
                    with open(pkt_path, "r", encoding="utf-8") as f:
                        current_packets.append(json.load(f))
                except Exception:
                    continue
            current_snapshot = (
                manifest_path.exists(),
                tuple(
                    sorted(
                        ((p.get("sequence_number"), p.get("media_id")) for p in current_packets),
                        key=lambda t: (str(t[0]), str(t[1])),
                    )
                ),
            )
            if current_snapshot == disk_snapshot:
                _atomic_write_json(sidecar, body.model_dump())

    return body


@app.delete("/api/inbox/{session_id}")
def inbox_delete(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Purge one session's manifest, packets, decoded sidecar, and archive."""
    import shutil

    _validate_session_id(session_id)
    inbox = _inbox_dir(current_user.id)
    with _get_session_lock(session_id):
        deleted = 0
        for candidate in (
            [inbox / f"_manifest_{session_id}.json", inbox / f"_decoded_{session_id}.json"]
            + list(inbox.glob(f"{session_id}__*.json"))
        ):
            try:
                if candidate.exists():
                    # missing_ok: a concurrent delete of the same file must
                    # never raise FileNotFoundError.
                    candidate.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                continue
        # Also purge the dropped-packet archive for this session, if any.
        shutil.rmtree(_dropped_archive_dir(inbox, session_id), ignore_errors=True)
    # The session is gone: release its lock-map entry (only if unlocked).
    _release_session_lock_entry(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="session not found")
    return {"success": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Direct Real-Time Steganographic Messenger (WhatsApp / Telegram Experience)
# ---------------------------------------------------------------------------

class ChatSendRequest(BaseModel):
    recipient_username: str
    message: str
    mode: Optional[str] = "dssc"


@app.get("/api/chat/contacts")
def chat_contacts(current_user: AuthenticatedUser = Depends(get_current_user)):
    """List available peers with real-time last message previews."""
    all_users = _auth.list_users_excluding(current_user.id)
    my_inbox = _inbox_dir(current_user.id)

    contacts = []
    for u in all_users:
        last_msg = ""
        last_time = 0.0

        # Check sent records
        for sent_file in my_inbox.glob("_sent_*.json"):
            try:
                with open(sent_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("recipient_username") == u.username and d.get("timestamp", 0) > last_time:
                    last_time = d.get("timestamp", 0)
                    last_msg = d.get("text", "")
            except Exception:
                pass

        # Check received records
        for mf in my_inbox.glob("_manifest_*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("sender_username") == u.username and d.get("timestamp", 0) > last_time:
                    dec_f = my_inbox / f"_decoded_{d.get('session_id')}.json"
                    if dec_f.exists():
                        with open(dec_f, "r", encoding="utf-8") as df:
                            t = json.load(df).get("reconstructed_meaning", "")
                            if t and not t.startswith("[UNVERIFIED:"):
                                last_time = d.get("timestamp", 0)
                                last_msg = t
            except Exception:
                pass

        # Truncate preview
        if len(last_msg) > 40:
            last_msg = last_msg[:40] + "..."

        contacts.append({
            "id": u.id,
            "username": u.username,
            "last_message": last_msg or "Tap to start secure chat",
            "last_timestamp": last_time,
            "online": True,
        })

    # Stable sort by most recent activity with username tiebreaker
    contacts.sort(key=lambda x: (-x["last_timestamp"], x["username"]))
    return {"contacts": contacts}


def _load_session_carriers(session_id: str, inbox_dirs: list[Path]) -> list[dict]:
    """Inspect and load actual on-disk steganographic carriers for a session."""
    engine = _get_engine()
    conduits = ["Channel 1", "Channel 2", "Channel 3"]
    carriers = []

    for inbox in inbox_dirs:
        pkt_files = sorted(inbox.glob(f"{session_id}__*.json"))
        if pkt_files:
            for pfile in pkt_files:
                try:
                    with open(pfile, "r", encoding="utf-8") as f:
                        pkt = json.load(f)
                    mid = pkt.get("media_id", "")
                    ch_id = pkt.get("channel_id", 0)
                    seq = pkt.get("sequence_number", len(carriers) + 1)

                    media_item = engine.index.get_by_id(mid) if mid else None
                    modality = media_item.modality if media_item else ("image" if mid.endswith(".jpg") or mid.endswith(".png") else "text")
                    content = media_item.content[:140] if media_item and media_item.content else mid
                    file_path = media_item.file_path if media_item and media_item.file_path else ""

                    try:
                        rel_pkt = str(pfile.relative_to(_PROJECT_ROOT))
                    except Exception:
                        rel_pkt = str(pfile)

                    carriers.append({
                        "seq": seq,
                        "media_id": mid,
                        "modality": modality,
                        "channel_id": ch_id,
                        "channel_name": conduits[ch_id % len(conduits)],
                        "content": content,
                        "file_path": file_path,
                        "packet_file": rel_pkt,
                        "verified": True,
                    })
                except Exception:
                    continue
            break
    return carriers


@app.get("/api/chat/messages")
def chat_messages(
    peer: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Retrieve two-way real-time conversation between current_user and peer."""
    peer_user = _auth.get_user_by_username(peer)
    if not peer_user:
        raise HTTPException(status_code=404, detail="Peer user not found")

    my_inbox = _inbox_dir(current_user.id)
    peer_inbox = _inbox_dir(peer_user.id)
    messages = []
    seen_ids = set()

    # 1. Sent by me to peer
    for sent_file in my_inbox.glob("_sent_*.json"):
        try:
            with open(sent_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            s_id = d.get("session_id")
            if s_id and s_id not in seen_ids and d.get("recipient_username") == peer_user.username:
                seen_ids.add(s_id)
                carriers = d.get("carriers") or _load_session_carriers(s_id, [peer_inbox, my_inbox])
                messages.append({
                    "id": s_id,
                    "sender": current_user.username,
                    "recipient": peer_user.username,
                    "text": d.get("text", ""),
                    "timestamp": d.get("timestamp", 0.0),
                    "is_mine": True,
                    "carrier_count": len(carriers) if carriers else d.get("carrier_count", 3),
                    "conduits": d.get("conduits", ["Channel 1", "Channel 2", "Channel 3"]),
                    "carriers": carriers,
                    "status": "delivered",
                    "verified": True,
                })
        except Exception:
            continue

    # 2. Received by me from peer
    for manifest_path in my_inbox.glob("_manifest_*.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            s_id = m.get("session_id")
            if s_id and s_id not in seen_ids and m.get("sender_username") == peer_user.username:
                decoded_path = my_inbox / f"_decoded_{s_id}.json"
                text = ""
                dec_carriers = []
                if decoded_path.exists():
                    try:
                        with open(decoded_path, "r", encoding="utf-8") as df:
                            dec = json.load(df)
                        text = dec.get("reconstructed_meaning", "")
                        dec_carriers = dec.get("carriers", [])
                    except Exception:
                        pass
                else:
                    try:
                        dec_result = inbox_decode(s_id, current_user)
                        text = dec_result.reconstructed_meaning
                    except Exception:
                        pass

                if text and not text.startswith("[UNVERIFIED:"):
                    seen_ids.add(s_id)
                    carriers = dec_carriers or _load_session_carriers(s_id, [my_inbox, peer_inbox])
                    messages.append({
                        "id": s_id,
                        "sender": peer_user.username,
                        "recipient": current_user.username,
                        "text": text,
                        "timestamp": m.get("timestamp", 0.0),
                        "is_mine": False,
                        "carrier_count": len(carriers) if carriers else m.get("total_items", 3),
                        "conduits": ["Channel 1", "Channel 2", "Channel 3"],
                        "carriers": carriers,
                        "status": "verified",
                        "verified": True,
                    })
        except Exception:
            continue

    # Sort ascending by timestamp and return most recent 50
    messages.sort(key=lambda x: x["timestamp"])
    return {"messages": messages[-50:], "peer": peer}


@app.get("/api/chat/carriers")
def chat_carriers(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Retrieve the full covert carrier breakdown for a session the caller sent or received."""
    _validate_session_id(session_id)
    if any(c in session_id for c in "*?[]"):
        raise HTTPException(
            status_code=400, detail="glob characters are not allowed in session_id"
        )
    # Resolve only the authorized directory (sender or recipient side of
    # this session) instead of globbing across every user's inbox.
    user_dir, _manifest = _find_wire_session(session_id, current_user)
    carriers = _load_session_carriers(session_id, [user_dir])
    conduits = ["Channel 1", "Channel 2", "Channel 3"]
    return {
        "session_id": session_id,
        "carrier_count": len(carriers),
        "conduits": conduits,
        "carriers": carriers,
    }


@app.get("/api/media/packet")
def media_packet(
    path: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Serve a specific JSON packet file safely from storage/shared_channel.

    Authenticated: the caller must own the channel directory (be the
    recipient) or be the recorded sender of the packet. Internal control
    files (`_manifest_*`, `_decoded_*`, `_sent_*`) and anything inside
    `.dropped_archive` can never be read through this endpoint.
    """
    candidate = Path(path)
    # Reject symlinks BEFORE resolving: resolve() would silently follow the
    # link, so a symlink planted inside shared_channel must never be served.
    candidate_abs = candidate if candidate.is_absolute() else (_PROJECT_ROOT / path)
    if candidate_abs.is_symlink():
        raise HTTPException(status_code=403, detail="Access denied: symlinks are not allowed")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (_PROJECT_ROOT / path).resolve()

    # Packet reads must stay strictly inside storage/shared_channel (the
    # startswith() string check was bypassable via sibling-prefix dirs).
    shared_channel_root = (_PROJECT_ROOT / "storage" / "shared_channel").resolve()
    if not resolved.is_relative_to(shared_channel_root):
        raise HTTPException(
            status_code=403, detail="Access denied: outside storage/shared_channel"
        )

    # Internal control files and the dropped-packet archive are never served.
    if resolved.name.startswith("_"):
        raise HTTPException(
            status_code=403,
            detail="Access denied: internal packet control files are not readable",
        )
    if ".dropped_archive" in resolved.relative_to(shared_channel_root).parts:
        raise HTTPException(
            status_code=403,
            detail="Access denied: archived dropped packets are not readable",
        )

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Packet file not found on disk")

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read packet file: {str(e)}")

    # Authorization: the caller must own this channel directory (recipient
    # side) or be the recorded sender of this packet.
    if (
        resolved.parent.name != str(current_user.id)
        and data.get("sender_username") != current_user.username
    ):
        raise HTTPException(
            status_code=403,
            detail="Access denied: packet belongs to another user's session",
        )

    return JSONResponse(content=data, media_type="application/json")


# Corpus assets served to the browser: the ONLY filesystem root that
# /api/media/file and /api/media/info may resolve into. Anything outside
# storage/data (auth.db, .env, source files, models, ...) is refused.
_MEDIA_CORPUS_ROOT = (_PROJECT_ROOT / "storage" / "data").resolve()
# Explicit allowlist of corpus asset extensions. Every other suffix
# (.db, .env, .py, .pt, ...) is rejected with 403 — never served as
# application/octet-stream.
_MEDIA_ALLOWED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".wav", ".mp3", ".ogg", ".flac", ".m4a",
    ".json", ".arrow",
}


def _resolve_media_target(path: Optional[str], item) -> Optional[Path]:
    """Resolve a requested media path strictly inside the corpus root.

    Raises 403 for anything outside storage/data or with a non-corpus
    extension, BEFORE any existence check, so the endpoint cannot be used
    as an existence oracle for arbitrary files.
    """
    if path:
        p = Path(path)
        target = p.resolve() if p.is_absolute() else (_PROJECT_ROOT / path).resolve()
    elif item is not None and item.file_path:
        p = Path(item.file_path)
        target = p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()
    else:
        return None
    if not target.is_relative_to(_MEDIA_CORPUS_ROOT):
        raise HTTPException(
            status_code=403,
            detail="Access denied: outside the corpus root (storage/data)",
        )
    if target.suffix.lower() not in _MEDIA_ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=403,
            detail=(
                "Access denied: file type "
                f"'{target.suffix.lower() or '(none)'}' is not a servable corpus asset"
            ),
        )
    return target


@app.get("/api/media/file")
def media_file(
    path: Optional[str] = None,
    media_id: Optional[str] = None,
):
    """Serve an actual carrier asset (image, audio, or text item) directly to the browser.

    NOTE: no Bearer header is required here because browser <img>/<audio>
    tags load this URL directly without headers; safety comes from locking
    path resolution to the storage/data corpus root + extension allowlist.
    """
    # Validate the explicit path BEFORE touching the engine so hostile
    # paths fail fast with 403.
    target_path = _resolve_media_target(path, None) if path else None

    engine = _get_engine()
    item = None
    if media_id:
        item = engine.index.get_by_id(media_id)

    if target_path is None:
        target_path = _resolve_media_target(None, item)

    if not target_path or not target_path.exists() or not target_path.is_file():
        if item and item.content:
            mod_val = item.modality.value if hasattr(item.modality, "value") else str(item.modality)
            return JSONResponse(
                content={
                    "media_id": item.id,
                    "modality": mod_val,
                    "content": item.content,
                    "file_path": item.file_path or "",
                    "metadata": item.metadata,
                },
                media_type="application/json",
            )
        raise HTTPException(status_code=404, detail="Media file not found")

    suffix = target_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return FileResponse(str(target_path), media_type="image/jpeg", filename=target_path.name)
    elif suffix == ".png":
        return FileResponse(str(target_path), media_type="image/png", filename=target_path.name)
    elif suffix == ".webp":
        return FileResponse(str(target_path), media_type="image/webp", filename=target_path.name)
    elif suffix == ".gif":
        return FileResponse(str(target_path), media_type="image/gif", filename=target_path.name)
    elif suffix == ".wav":
        return FileResponse(str(target_path), media_type="audio/wav", filename=target_path.name)
    elif suffix == ".mp3":
        return FileResponse(str(target_path), media_type="audio/mpeg", filename=target_path.name)
    elif suffix == ".ogg":
        return FileResponse(str(target_path), media_type="audio/ogg", filename=target_path.name)
    elif suffix == ".flac":
        return FileResponse(str(target_path), media_type="audio/flac", filename=target_path.name)
    elif suffix == ".m4a":
        return FileResponse(str(target_path), media_type="audio/mp4", filename=target_path.name)
    elif suffix == ".json":
        if item and item.content:
            mod_val = item.modality.value if hasattr(item.modality, "value") else str(item.modality)
            return JSONResponse(
                content={
                    "media_id": item.id,
                    "modality": mod_val,
                    "content": item.content,
                    "file_path": str(target_path),
                    "metadata": item.metadata,
                },
                media_type="application/json",
            )
        return FileResponse(str(target_path), media_type="application/json", filename=target_path.name)
    elif suffix == ".arrow":
        mod_val = item.modality.value if hasattr(item.modality, "value") else str(item.modality) if item else "audio"
        return JSONResponse(
            content={
                "media_id": item.id if item else target_path.stem,
                "modality": mod_val,
                "content": item.content if item else "Audio carrier asset",
                "file_path": str(target_path),
                "metadata": item.metadata if item else {},
            },
            media_type="application/json",
        )
    else:
        # Unreachable in practice: the allowlist is enforced in
        # _resolve_media_target. Kept defensive — never fall through to
        # serving arbitrary bytes as application/octet-stream.
        raise HTTPException(status_code=403, detail="Access denied: unsupported file type")


@app.get("/api/media/info")
def media_info(
    media_id: str,
    path: Optional[str] = None,
):
    """Return structured carrier metadata, resolved disk path, content, and live SHA-256."""
    import hashlib
    from urllib.parse import urlencode

    engine = _get_engine()
    item = engine.index.get_by_id(media_id)
    if item is None and not path:
        raise HTTPException(status_code=404, detail=f"carrier not found in corpus: {media_id}")

    mod_val = (
        item.modality.value
        if item and hasattr(item.modality, "value")
        else (str(item.modality) if item else "unknown")
    )
    meta = dict(item.metadata) if item and isinstance(item.metadata, dict) else {}

    raw_path = path or (item.file_path if item else "") or ""
    target_path: Optional[Path] = None
    if raw_path:
        p = Path(raw_path)
        candidate = p.resolve() if p.is_absolute() else (_PROJECT_ROOT / raw_path).resolve()
        # Locked to the corpus root: paths outside storage/data (auth.db,
        # .env, source, ...) are refused with 403, never inspected.
        if not candidate.is_relative_to(_MEDIA_CORPUS_ROOT):
            raise HTTPException(
                status_code=403,
                detail="Access denied: outside the corpus root (storage/data)",
            )
        if candidate.exists() and candidate.is_file():
            target_path = candidate

    # Fallback to the real on-disk metadata catalog if the synthesized cache path isn't a standalone file
    if target_path is None and mod_val in ("text", "audio", "image"):
        catalog_candidate = (_PROJECT_ROOT / "storage" / "data" / "indices" / f"{mod_val}_metadata.json").resolve()
        if (
            catalog_candidate.is_relative_to(_MEDIA_CORPUS_ROOT)
            and catalog_candidate.exists()
            and catalog_candidate.is_file()
        ):
            target_path = catalog_candidate

    exists = bool(target_path and target_path.exists() and target_path.is_file())
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    rel_path = raw_path
    suffix = ""
    if exists and target_path is not None:
        size_bytes = target_path.stat().st_size
        suffix = target_path.suffix.lower()
        try:
            rel_path = str(target_path.relative_to(_PROJECT_ROOT.resolve()))
        except Exception:
            rel_path = str(target_path)
        # Compute live SHA-256 for files up to 80 MiB (covers all corpus images + text/audio catalogs)
        if size_bytes <= 80 * 1024 * 1024:
            h = hashlib.sha256()
            with open(target_path, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            sha256 = h.hexdigest()

    full_text = (
        meta.get("text")
        or meta.get("caption")
        or meta.get("transcription")
        or (item.content if item else "")
        or ""
    )

    is_image = mod_val == "image" or suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif")
    is_audio_playable = suffix in (".wav", ".mp3", ".ogg", ".flac", ".m4a")
    query_params = {"media_id": media_id}
    if rel_path:
        query_params["path"] = rel_path

    return {
        "media_id": media_id,
        "modality": mod_val,
        "content": full_text,
        "file_path": rel_path,
        "exists_on_disk": exists,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "bytes_modified": 0,
        "is_image": is_image,
        "is_audio_playable": is_audio_playable,
        "file_url": f"/api/media/file?{urlencode(query_params)}",
        "metadata": {
            k: v
            for k, v in meta.items()
            if isinstance(v, (str, int, float, bool)) and k not in ("text", "caption")
        },
    }


@app.post("/api/chat/send")
def chat_send(
    req: ChatSendRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Seamless real-time steganographic chat transmission."""
    recipient = _auth.get_user_by_username(req.recipient_username)
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient not found")
    if recipient.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot chat with yourself")

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    engine_mode = req.mode or "dssc"
    session_key: Optional[bytes] = None
    session_key_hex: Optional[str] = None
    if engine_mode == "dssc":
        session_key = os.urandom(32)
        session_key_hex = session_key.hex()

    engine = _get_engine()
    with _engine_work_lock:
        enc_res = engine.encode(
            message=text,
            mode=engine_mode,
            session_key=session_key,
            diversity_mode="round_robin",
        )

    session_id = f"chat_{uuid.uuid4().hex[:10]}"
    t_now = time.time()

    # 1. Write to recipient inbox
    r_inbox = _inbox_dir(recipient.id)
    slot_roles: list[str] = []
    slot_bytes_hex: list[list[str]] = []
    slot_byte_indices: list[list[int]] = []
    framed_len = 0
    try:
        from src.engine.payload_framing import frame_payload
        from src.engine.ecc import RSErrorCorrection
        from src.api.demo import _carrier_byte_spans_dssc, _role_for_bytes

        framed = frame_payload(text)
        framed_len = len(framed)
        codeword = RSErrorCorrection(parity_bytes=8).encode(framed)
        if engine_mode == "exact_vcp":
            for i in range(enc_res.carrier_count):
                slot_roles.append(_role_for_bytes([i], framed_len))
                slot_bytes_hex.append([f"{codeword[i]:02x}"] if i < len(codeword) else [])
                slot_byte_indices.append([i] if i < len(codeword) else [])
        elif session_key is not None:
            # The span computation reads shared engine state (index/mapper),
            # so it must run under the engine work lock like the encode above.
            with _engine_work_lock:
                spans = _carrier_byte_spans_dssc(
                    engine, enc_res.carrier_count, session_key, "round_robin", len(codeword)
                )
            for _, _, b_indices in spans:
                slot_roles.append(_role_for_bytes(b_indices, framed_len))
                slot_bytes_hex.append([f"{codeword[b]:02x}" for b in b_indices if b < len(codeword)])
                slot_byte_indices.append([b for b in b_indices if b < len(codeword)])
    except Exception:
        pass

    manifest = {
        "session_id": session_id,
        "sender_username": current_user.username,
        "sender_id": current_user.id,
        "recipient_username": recipient.username,
        "recipient_id": recipient.id,
        "timestamp": t_now,
        "mode_used": engine_mode,
        "engine_mode": engine_mode,
        "diversity_mode": "round_robin",
        "ecc_parity_bytes": 8,
        "sequence_base": 1,
        "session_key_hex": session_key_hex,
        "bits_per_carrier": enc_res.bits_per_carrier,
        "total_items": enc_res.carrier_count,
        "framed_bytes_len": framed_len,
        "slot_roles": slot_roles,
        "slot_bytes_hex": slot_bytes_hex,
        "slot_byte_indices": slot_byte_indices,
    }
    with open(r_inbox / f"_manifest_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    conduits = ["Channel 1", "Channel 2", "Channel 3"]
    items = []
    carriers_details = []
    for idx, mid in enumerate(enc_res.media_ids):
        ch_id = idx % 3
        media_item = engine.index.get_by_id(mid)
        modality = media_item.modality if media_item else ("image" if mid.startswith("flickr") or mid.endswith(".jpg") or mid.endswith(".png") else "audio" if mid.startswith("audio_") else "text")
        pkt = {
            "session_id": session_id,
            "sequence_number": idx + 1,
            "media_id": mid,
            "modality": modality,
            "channel_id": ch_id,
            "timestamp": t_now,
            "sender_username": current_user.username,
            "recipient_username": recipient.username,
            "mode_used": engine_mode,
            "delay_seconds": 0.0,
        }
        pkt_file = r_inbox / f"{session_id}__{idx + 1:04d}.json"
        with open(pkt_file, "w", encoding="utf-8") as f:
            json.dump(pkt, f, indent=2)

        content = media_item.content[:140] if media_item and media_item.content else mid
        file_path = media_item.file_path if media_item and media_item.file_path else ""

        try:
            rel_pkt = str(pkt_file.relative_to(_PROJECT_ROOT))
        except Exception:
            rel_pkt = str(pkt_file)

        items.append({
            "media_id": mid,
            "modality": modality,
            "content": content,
            "verified": True,
        })
        carriers_details.append({
            "seq": idx + 1,
            "media_id": mid,
            "modality": modality,
            "channel_id": ch_id,
            "channel_name": conduits[ch_id % len(conduits)],
            "content": content,
            "file_path": file_path,
            "packet_file": rel_pkt,
            "verified": True,
        })

    # Pre-cache decode in recipient inbox for instant display
    decoded_body = {
        "session_id": session_id,
        "sender_username": current_user.username,
        "recipient_username": recipient.username,
        "mode_used": engine_mode,
        "engine_mode": engine_mode,
        "session_key_hex": session_key_hex,
        "bits_per_carrier": enc_res.bits_per_carrier,
        "reconstructed_meaning": text,
        "verification_rate": 1.0,
        "all_verified": True,
        "ecc_success": True,
        "items": items,
        "carriers": carriers_details,
        "cached": True,
    }
    with open(r_inbox / f"_decoded_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(decoded_body, f, indent=2)

    # 2. Write sent record in current user inbox
    s_inbox = _inbox_dir(current_user.id)
    sent_record = {
        "session_id": session_id,
        "sender_username": current_user.username,
        "recipient_username": recipient.username,
        "text": text,
        "timestamp": t_now,
        "carrier_count": enc_res.carrier_count,
        "conduits": conduits[:min(enc_res.carrier_count, 3)],
        "carriers": carriers_details,
        "mode": engine_mode,
        "bits_per_carrier": enc_res.bits_per_carrier,
        "verified": True,
    }
    with open(s_inbox / f"_sent_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(sent_record, f, indent=2)

    return {
        "success": True,
        "id": session_id,
        "sender": current_user.username,
        "recipient": recipient.username,
        "text": text,
        "timestamp": t_now,
        "is_mine": True,
        "carrier_count": enc_res.carrier_count,
        "conduits": conduits[:min(enc_res.carrier_count, 3)],
        "carriers": carriers_details,
        "status": "delivered",
        "verified": True,
    }


# ---------------------------------------------------------------------------
# Real On-Disk Wire Session Management (Transmit, Drop/Delete, Tamper, Decode)
# ---------------------------------------------------------------------------


def _find_wire_session(
    session_id: str,
    current_user: AuthenticatedUser,
) -> tuple[Path, dict]:
    """Locate a session manifest on disk owned by or sent by current_user."""
    _validate_session_id(session_id)
    shared_root = _PROJECT_ROOT / "storage" / "shared_channel"
    if not shared_root.exists():
        raise HTTPException(status_code=404, detail="session not found on wire")

    for user_dir in sorted(shared_root.iterdir()):
        if not user_dir.is_dir():
            continue
        mf_path = user_dir / f"_manifest_{session_id}.json"
        if not mf_path.exists():
            continue
        try:
            with open(mf_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        is_recipient = (
            str(user_dir.name) == str(current_user.id)
            or manifest.get("recipient_id") == current_user.id
            or manifest.get("recipient_username") == current_user.username
        )
        is_sender = (
            manifest.get("sender_id") == current_user.id
            or manifest.get("sender_username") == current_user.username
        )
        if is_recipient or is_sender:
            return user_dir, manifest

    raise HTTPException(status_code=404, detail="session not found on wire")


def _infer_seq_base(session_id: str, manifest: dict, packets: list[dict], total_items: int) -> int:
    seq_base = manifest.get("sequence_base")
    if seq_base is not None:
        return int(seq_base)
    if session_id.startswith("chat_"):
        return 1
    if packets:
        seqs = [int(p.get("sequence_number", 0)) for p in packets]
        if min(seqs) >= 1 and max(seqs) <= total_items:
            return 1
    return 0


def _inspect_wire_session(user_dir: Path, session_id: str, manifest: dict) -> dict:
    """Build a truthful on-disk snapshot of one wire session."""
    total_items = int(manifest.get("total_items") or 0)
    pkt_files = sorted(user_dir.glob(f"{session_id}__*.json"))
    raw_packets: list[tuple[Path, dict]] = []
    for pfile in pkt_files:
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                raw_packets.append((pfile, json.load(f)))
        except Exception:
            continue

    if total_items <= 0:
        total_items = len(raw_packets)

    seq_base = _infer_seq_base(session_id, manifest, [d for _, d in raw_packets], total_items)

    slot_map: dict[int, tuple[Path, dict]] = {}
    for pfile, pdata in raw_packets:
        slot = int(pdata.get("sequence_number", 0)) - seq_base
        if 0 <= slot < total_items:
            slot_map[slot] = (pfile, pdata)

    slot_roles: list[str] = manifest.get("slot_roles") or []
    slot_bytes_hex: list[list[str]] = manifest.get("slot_bytes_hex") or []
    slot_byte_indices: list[list[int]] = manifest.get("slot_byte_indices") or []
    engine_mode = manifest.get("engine_mode") or "exact_vcp"

    if not slot_byte_indices and total_items > 0:
        if engine_mode == "exact_vcp":
            slot_byte_indices = [[i] for i in range(total_items)]
        elif manifest.get("session_key_hex") and _engine is not None and _engine.is_loaded():
            try:
                from src.api.demo import _carrier_byte_spans_dssc

                sk = bytes.fromhex(manifest["session_key_hex"])
                est_cw_len = int(manifest.get("framed_bytes_len") or 0) + int(manifest.get("ecc_parity_bytes") or 8)
                if est_cw_len <= 8:
                    est_cw_len = max(16, int(round(total_items * float(manifest.get("bits_per_carrier") or 13.5) / 8.0)) + 4)
                spans = _carrier_byte_spans_dssc(
                    _engine,
                    total_items,
                    sk,
                    manifest.get("diversity_mode") or "round_robin",
                    est_cw_len,
                )
                slot_byte_indices = [list(b_indices) for _, _, b_indices in spans]
            except Exception:
                slot_byte_indices = []

    slots_out: list[dict] = []
    dropped_sequences: list[int] = []
    tampered_sequences: list[int] = []
    channel_counts: dict[str, int] = {"0": 0, "1": 0, "2": 0}

    for slot in range(total_items):
        role = slot_roles[slot] if slot < len(slot_roles) else "payload"
        b_hex = slot_bytes_hex[slot] if slot < len(slot_bytes_hex) else []
        b_idx = slot_byte_indices[slot] if slot < len(slot_byte_indices) else []
        if slot not in slot_map:
            dropped_sequences.append(slot)
            slots_out.append(
                {
                    "seq": slot,
                    "status": "deleted",
                    "filename": None,
                    "rel_path": None,
                    "media_id": None,
                    "modality": "dropped",
                    "channel_id": slot % 3,
                    "delay_seconds": 0.0,
                    "timestamp": None,
                    "role": role,
                    "bytes_hex": b_hex,
                    "byte_indices": b_idx,
                }
            )
            continue

        pfile, pdata = slot_map[slot]
        is_tampered = bool(pdata.get("tampered", False))
        if is_tampered:
            tampered_sequences.append(slot)
        ch = int(pdata.get("channel_id", 0))
        ch_key = str(ch % 3)
        channel_counts[ch_key] = channel_counts.get(ch_key, 0) + 1
        try:
            rel_path = str(pfile.relative_to(_PROJECT_ROOT))
        except Exception:
            rel_path = str(pfile)
        raw_mod = pdata.get("modality")
        mid_val = pdata.get("media_id") or ""
        if not raw_mod or raw_mod == "carrier":
            m_item = _engine.index.get_by_id(mid_val) if (_engine is not None and _engine.is_loaded() and mid_val) else None
            if m_item is not None:
                raw_mod = m_item.modality
            elif mid_val.startswith("flickr") or mid_val.endswith((".jpg", ".png")):
                raw_mod = "image"
            elif mid_val.startswith("audio_"):
                raw_mod = "audio"
            else:
                raw_mod = "text"
        slots_out.append(
            {
                "seq": slot,
                "status": "tampered" if is_tampered else "on_disk",
                "filename": pfile.name,
                "rel_path": rel_path,
                "media_id": mid_val or None,
                "original_media_id": pdata.get("original_media_id"),
                "modality": raw_mod,
                "channel_id": ch,
                "delay_seconds": float(pdata.get("delay_seconds", 0.0)),
                "timestamp": pdata.get("timestamp"),
                "role": role,
                "bytes_hex": b_hex,
                "byte_indices": b_idx,
            }
        )

    erased_bytes_set: set[int] = set()
    for s in dropped_sequences:
        if s < len(slot_byte_indices):
            erased_bytes_set.update(slot_byte_indices[s])
    tampered_bytes_set: set[int] = set()
    for s in tampered_sequences:
        if s < len(slot_byte_indices):
            for b in slot_byte_indices[s]:
                if b not in erased_bytes_set:
                    tampered_bytes_set.add(b)
    has_exact_spans = bool(slot_byte_indices)
    exact_fault_weight = (
        len(erased_bytes_set) + 2 * len(tampered_bytes_set) if has_exact_spans else None
    )

    decoded_path = user_dir / f"_decoded_{session_id}.json"
    last_decoded = None
    if decoded_path.exists():
        try:
            with open(decoded_path, "r", encoding="utf-8") as f:
                last_decoded = json.load(f)
        except Exception:
            last_decoded = None

    try:
        channel_dir_rel = str(user_dir.relative_to(_PROJECT_ROOT))
    except Exception:
        channel_dir_rel = str(user_dir)

    ecc_parity = int(manifest.get("ecc_parity_bytes") or 8)

    return {
        "session_id": session_id,
        "sender_username": manifest.get("sender_username") or "unknown",
        "recipient_username": manifest.get("recipient_username") or "unknown",
        "channel_dir": channel_dir_rel,
        "engine_mode": manifest.get("engine_mode") or "exact_vcp",
        "diversity_mode": manifest.get("diversity_mode") or "round_robin",
        "mode_used": manifest.get("mode_used") or "static",
        "ecc_parity_bytes": ecc_parity,
        "bits_per_carrier": manifest.get("bits_per_carrier") or (8.0 if (manifest.get("engine_mode") or "exact_vcp") == "exact_vcp" else 13.5),
        "total_items": total_items,
        "surviving_count": total_items - len(dropped_sequences),
        "dropped_count": len(dropped_sequences),
        "tampered_count": len(tampered_sequences),
        "dropped_sequences": dropped_sequences,
        "tampered_sequences": tampered_sequences,
        "erased_bytes_count": len(erased_bytes_set) if has_exact_spans else None,
        "tampered_bytes_count": len(tampered_bytes_set) if has_exact_spans else None,
        "exact_fault_weight": exact_fault_weight,
        "channel_counts": channel_counts,
        "timestamp": manifest.get("timestamp", 0.0),
        "decoded": decoded_path.exists(),
        "last_decoded_message": (
            last_decoded.get("reconstructed_meaning")
            if isinstance(last_decoded, dict) and last_decoded.get("ecc_success")
            else None
        ),
        "packets": slots_out,
    }


class WireSendRequest(BaseModel):
    message: str = Field(min_length=1, max_length=220)
    recipient_username: Optional[str] = None
    engine_mode: Literal["exact_vcp", "dssc"] = "exact_vcp"
    diversity_mode: Literal["best", "round_robin", "balanced"] = "round_robin"
    ecc_parity_bytes: int = Field(default=8, ge=4, le=32)
    stealth_mode: Literal["static", "rl", "gan", "auto"] = "auto"


@app.post("/api/wire/send")
def wire_send(
    req: WireSendRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Encode a real message and write every carrier packet as an actual JSON file on disk."""
    from src.engine.payload_framing import frame_payload
    from src.engine.ecc import RSErrorCorrection
    from src.api.demo import _carrier_byte_spans_dssc, _role_for_bytes

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message must not be empty")

    if req.recipient_username:
        recipient = _auth.get_user_by_username(req.recipient_username)
        if recipient is None:
            raise HTTPException(status_code=404, detail=f"recipient {req.recipient_username!r} not found")
    else:
        default_peer = "alice" if current_user.username == "bob" else "bob"
        recipient = _auth.get_user_by_username(default_peer) or current_user

    session_key: Optional[bytes] = None
    session_key_hex: Optional[str] = None
    if req.engine_mode == "dssc":
        session_key = os.urandom(32)
        session_key_hex = session_key.hex()

    t0 = time.perf_counter()
    engine = _get_engine()
    with _engine_work_lock:
        enc_res = engine.encode(
            message=text,
            mode=req.engine_mode,
            session_key=session_key,
            use_ecc=True,
            ecc_parity_bytes=req.ecc_parity_bytes,
            diversity_mode=req.diversity_mode,
        )

    framed = frame_payload(text)
    codeword = RSErrorCorrection(parity_bytes=req.ecc_parity_bytes).encode(framed)

    slot_roles: list[str] = []
    slot_bytes_hex: list[list[str]] = []
    slot_byte_indices: list[list[int]] = []
    if req.engine_mode == "exact_vcp":
        for i in range(enc_res.carrier_count):
            slot_roles.append(_role_for_bytes([i], len(framed)))
            slot_bytes_hex.append([f"{codeword[i]:02x}"] if i < len(codeword) else [])
            slot_byte_indices.append([i] if i < len(codeword) else [])
    else:
        assert session_key is not None
        # Span computation reads shared engine state: hold the engine work
        # lock so it cannot interleave with a concurrent encode/decode.
        with _engine_work_lock:
            spans = _carrier_byte_spans_dssc(
                engine, enc_res.carrier_count, session_key, req.diversity_mode, len(codeword)
            )
        for _, _, b_indices in spans:
            slot_roles.append(_role_for_bytes(b_indices, len(framed)))
            slot_bytes_hex.append([f"{codeword[b]:02x}" for b in b_indices if b < len(codeword)])
            slot_byte_indices.append([b for b in b_indices if b < len(codeword)])

    # Schedule channel & delay assignment via StealthScheduler
    mode_used = req.stealth_mode
    channels: list[int] = [i % 3 for i in range(enc_res.carrier_count)]
    delays: list[float] = [0.25 for _ in range(enc_res.carrier_count)]
    try:
        from src.stealth.stealth_scheduler import StealthScheduler

        scheduler = StealthScheduler(num_channels=3, device="cpu", profile="casual")
        sched = scheduler.schedule(media_ids=enc_res.media_ids, mode=req.stealth_mode, base_delay=0.5)
        mode_used = sched.get("mode_used", req.stealth_mode)
        sched_pairs = [
            (ch, d)
            for mid, ch, d in zip(sched["items"], sched["channels"], sched["delays"])
            if mid is not None
        ]
        if len(sched_pairs) == enc_res.carrier_count:
            channels = [int(c) for c, _ in sched_pairs]
            delays = [round(float(d), 3) for _, d in sched_pairs]
    except Exception:
        pass

    session_id = f"wire_{uuid.uuid4().hex[:10]}"
    t_now = time.time()
    r_inbox = _inbox_dir(recipient.id)

    manifest = {
        "session_id": session_id,
        "sender_username": current_user.username,
        "sender_id": current_user.id,
        "recipient_username": recipient.username,
        "recipient_id": recipient.id,
        "timestamp": t_now,
        "mode_used": mode_used,
        "engine_mode": req.engine_mode,
        "diversity_mode": req.diversity_mode,
        "ecc_parity_bytes": req.ecc_parity_bytes,
        "sequence_base": 0,
        "session_key_hex": session_key_hex,
        "bits_per_carrier": enc_res.bits_per_carrier,
        "total_items": enc_res.carrier_count,
        "framed_bytes_len": len(framed),
        "slot_roles": slot_roles,
        "slot_bytes_hex": slot_bytes_hex,
        "slot_byte_indices": slot_byte_indices,
    }
    with open(r_inbox / f"_manifest_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for idx, mid in enumerate(enc_res.media_ids):
        media_item = engine.index.get_by_id(mid)
        modality = media_item.modality if media_item else "carrier"
        pkt = {
            "session_id": session_id,
            "sequence_number": idx,
            "media_id": mid,
            "modality": modality,
            "channel_id": channels[idx],
            "delay_seconds": delays[idx],
            "timestamp": t_now + idx * 0.001,
            "sender_username": current_user.username,
            "recipient_username": recipient.username,
            "mode_used": mode_used,
        }
        pkt_path = r_inbox / f"{session_id}__seq{idx:04d}.json"
        with open(pkt_path, "w", encoding="utf-8") as f:
            json.dump(pkt, f, indent=2)

    snapshot = _inspect_wire_session(r_inbox, session_id, manifest)
    snapshot["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return snapshot


@app.get("/api/wire/sessions")
def wire_sessions_list(
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """List real on-disk wire sessions sent by or addressed to the current user."""
    shared_root = _PROJECT_ROOT / "storage" / "shared_channel"
    if not shared_root.exists():
        return {"sessions": []}

    sessions: list[dict] = []
    for user_dir in sorted(shared_root.iterdir()):
        if not user_dir.is_dir():
            continue
        for mf_path in user_dir.glob("_manifest_*.json"):
            try:
                with open(mf_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue
            is_recipient = (
                str(user_dir.name) == str(current_user.id)
                or manifest.get("recipient_id") == current_user.id
                or manifest.get("recipient_username") == current_user.username
            )
            is_sender = (
                manifest.get("sender_id") == current_user.id
                or manifest.get("sender_username") == current_user.username
            )
            if not (is_recipient or is_sender):
                continue
            sid = manifest.get("session_id") or mf_path.stem.removeprefix("_manifest_")
            sessions.append(_inspect_wire_session(user_dir, sid, manifest))

    sessions.sort(key=lambda s: float(s.get("timestamp") or 0.0), reverse=True)
    return {"sessions": sessions[:25]}


@app.get("/api/wire/sessions/{session_id}")
def wire_session_detail(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Inspect one real on-disk wire session."""
    user_dir, manifest = _find_wire_session(session_id, current_user)
    return _inspect_wire_session(user_dir, session_id, manifest)


class WireDropBatchRequest(BaseModel):
    sequences: list[int] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Per-session on-disk consistency: one lock per session id serializes
# drop/restore/tamper/decode/delete mutations, and JSON writes go through an
# atomic temp-file + os.replace so a concurrent reader never sees a torn file.
# ---------------------------------------------------------------------------
_wire_session_locks: dict[str, threading.Lock] = {}
_wire_session_locks_guard = threading.Lock()


def _release_session_lock_entry(session_id: str) -> None:
    """Drop *session_id* from the lock map once no thread holds its lock.

    Under `_wire_session_locks_guard`; only removes the entry when the lock is
    currently unlocked, so a thread that just acquired it is never orphaned.
    """
    with _wire_session_locks_guard:
        lock = _wire_session_locks.get(session_id)
        if lock is not None and not lock.locked():
            _wire_session_locks.pop(session_id, None)


def _get_session_lock(session_id: str) -> threading.Lock:
    """Return the mutex serializing on-disk mutations for one wire session."""
    with _wire_session_locks_guard:
        lock = _wire_session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _wire_session_locks[session_id] = lock
        if len(_wire_session_locks) > 512:
            # Bound the map: evict the oldest UNLOCKED entries (insertion
            # order) until within the cap; locked entries are in use and the
            # current session's lock is never evicted.
            for sid, entry_lock in list(_wire_session_locks.items()):
                if len(_wire_session_locks) <= 512:
                    break
                if sid == session_id or entry_lock.locked():
                    continue
                _wire_session_locks.pop(sid, None)
        return lock


def _atomic_write_json(path: Path, data) -> None:
    """Atomically write JSON to *path* (temp file in the same dir + os.replace)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _dropped_archive_dir(user_dir: Path, session_id: str) -> Path:
    """Separate archive outside the active channel dir so unlinked packets can be restored for re-testing."""
    return _PROJECT_ROOT / "storage" / "shared_channel" / ".dropped_archive" / user_dir.name / session_id


def _unlink_wire_sequences(
    user_dir: Path,
    session_id: str,
    manifest: dict,
    sequences: list[int],
) -> list[str]:
    """Archive a backup copy outside the channel dir, then physically delete packet JSON files."""
    import shutil

    with _get_session_lock(session_id):
        total_items = int(manifest.get("total_items") or 0)
        pkt_files = sorted(user_dir.glob(f"{session_id}__*.json"))
        loaded: list[tuple[Path, dict]] = []
        for pfile in pkt_files:
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    loaded.append((pfile, json.load(f)))
            except Exception:
                continue
        seq_base = _infer_seq_base(session_id, manifest, [d for _, d in loaded], total_items or len(loaded))
        target_set = {int(s) for s in sequences}
        deleted_paths: list[str] = []
        archive_dir = _dropped_archive_dir(user_dir, session_id)

        for pfile, pdata in loaded:
            slot = int(pdata.get("sequence_number", 0)) - seq_base
            if slot in target_set and pfile.exists():
                try:
                    rel = str(pfile.relative_to(_PROJECT_ROOT))
                except Exception:
                    rel = str(pfile)
                archive_dir.mkdir(parents=True, exist_ok=True)
                backup_path = archive_dir / pfile.name
                if not backup_path.exists():
                    shutil.copy2(pfile, backup_path)
                # missing_ok: a concurrent drop of the same packet must never
                # raise FileNotFoundError.
                pfile.unlink(missing_ok=True)
                deleted_paths.append(rel)

        # Invalidate cached decode so the next decode must read surviving disk files
        sidecar = user_dir / f"_decoded_{session_id}.json"
        sidecar.unlink(missing_ok=True)
        return deleted_paths


class WireRestoreRequest(BaseModel):
    sequences: Optional[list[int]] = None


@app.post("/api/wire/sessions/{session_id}/restore")
def wire_restore_packets(
    session_id: str,
    req: Optional[WireRestoreRequest] = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Restore physically deleted packet JSON files from archive and/or revert tampered files."""
    import shutil

    user_dir, manifest = _find_wire_session(session_id, current_user)
    with _get_session_lock(session_id):
        total_items = int(manifest.get("total_items") or 0)
        # An explicit (possibly empty) sequences list restores exactly those
        # slots: {"sequences": []} restores NOTHING. Only a missing body /
        # null field means "restore all".
        target_set = (
            {int(s) for s in req.sequences}
            if (req is not None and req.sequences is not None)
            else None
        )
        archive_dir = _dropped_archive_dir(user_dir, session_id)

        restored_paths: list[str] = []

        # 1. Restore deleted packet files from archive back into the active channel directory
        if archive_dir.exists():
            archived_loaded: list[tuple[Path, dict]] = []
            for bfile in sorted(archive_dir.glob(f"{session_id}__*.json")):
                try:
                    with open(bfile, "r", encoding="utf-8") as f:
                        archived_loaded.append((bfile, json.load(f)))
                except Exception:
                    continue
            seq_base = _infer_seq_base(
                session_id, manifest, [d for _, d in archived_loaded], total_items or len(archived_loaded)
            )
            for bfile, bdata in archived_loaded:
                slot = int(bdata.get("sequence_number", 0)) - seq_base
                if target_set is not None and slot not in target_set:
                    continue
                dest = user_dir / bfile.name
                # If the backup was tampered before deletion, revert it to its original carrier ID too
                if bdata.get("tampered") and bdata.get("original_media_id"):
                    bdata["media_id"] = bdata.pop("original_media_id")
                    bdata["tampered"] = False
                    _atomic_write_json(dest, bdata)
                else:
                    shutil.copy2(bfile, dest)
                bfile.unlink(missing_ok=True)
                try:
                    restored_paths.append(str(dest.relative_to(_PROJECT_ROOT)))
                except Exception:
                    restored_paths.append(str(dest))

        # 2. Revert any tampered files currently on disk (for matching sequences or all)
        live_loaded: list[tuple[Path, dict]] = []
        for pfile in sorted(user_dir.glob(f"{session_id}__*.json")):
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    live_loaded.append((pfile, json.load(f)))
            except Exception:
                continue
        live_seq_base = _infer_seq_base(
            session_id, manifest, [d for _, d in live_loaded], total_items or len(live_loaded)
        )
        for pfile, pdata in live_loaded:
            if not pdata.get("tampered") or not pdata.get("original_media_id"):
                continue
            slot = int(pdata.get("sequence_number", 0)) - live_seq_base
            if target_set is not None and slot not in target_set:
                continue
            pdata["media_id"] = pdata.pop("original_media_id")
            pdata["tampered"] = False
            _atomic_write_json(pfile, pdata)
            try:
                rel = str(pfile.relative_to(_PROJECT_ROOT))
            except Exception:
                rel = str(pfile)
            if rel not in restored_paths:
                restored_paths.append(rel)

        # Clear cached decode so receiver re-reads fresh disk state
        sidecar = user_dir / f"_decoded_{session_id}.json"
        sidecar.unlink(missing_ok=True)

        snapshot = _inspect_wire_session(user_dir, session_id, manifest)
        snapshot["restored_files"] = restored_paths
        return snapshot


@app.delete("/api/wire/sessions/{session_id}/packets/{seq}")
def wire_drop_single_packet(
    session_id: str,
    seq: int,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Physically delete one packet JSON file from the shared channel directory."""
    user_dir, manifest = _find_wire_session(session_id, current_user)
    deleted = _unlink_wire_sequences(user_dir, session_id, manifest, [seq])
    snapshot = _inspect_wire_session(user_dir, session_id, manifest)
    snapshot["deleted_files"] = deleted
    return snapshot


@app.post("/api/wire/sessions/{session_id}/drop")
def wire_drop_batch_packets(
    session_id: str,
    req: WireDropBatchRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Physically delete multiple packet JSON files from the shared channel directory."""
    user_dir, manifest = _find_wire_session(session_id, current_user)
    deleted = _unlink_wire_sequences(user_dir, session_id, manifest, req.sequences)
    snapshot = _inspect_wire_session(user_dir, session_id, manifest)
    snapshot["deleted_files"] = deleted
    return snapshot


@app.post("/api/wire/sessions/{session_id}/tamper/{seq}")
def wire_tamper_single_packet(
    session_id: str,
    seq: int,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Physically overwrite a packet JSON file on disk with a wrong-cluster carrier ID."""
    from src.api.demo import _pick_wrong_carrier

    user_dir, manifest = _find_wire_session(session_id, current_user)
    with _get_session_lock(session_id):
        total_items = int(manifest.get("total_items") or 0)
        pkt_files = sorted(user_dir.glob(f"{session_id}__*.json"))
        loaded: list[tuple[Path, dict]] = []
        for pfile in pkt_files:
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    loaded.append((pfile, json.load(f)))
            except Exception:
                continue
        seq_base = _infer_seq_base(session_id, manifest, [d for _, d in loaded], total_items or len(loaded))

        target_file: Optional[Path] = None
        target_data: Optional[dict] = None
        for pfile, pdata in loaded:
            if int(pdata.get("sequence_number", 0)) - seq_base == seq:
                target_file = pfile
                target_data = pdata
                break

        if target_file is None or target_data is None:
            raise HTTPException(status_code=404, detail=f"packet slot #{seq} not present on disk")

        engine = _get_engine()
        orig_mid = target_data.get("original_media_id") or target_data.get("media_id", "")
        with _engine_work_lock:
            wrong_mid = _pick_wrong_carrier(engine, orig_mid, manifest.get("engine_mode") or "exact_vcp", seq)

        if wrong_mid == orig_mid:
            # No substitute carrier exists for this slot: the packet is
            # unchanged, so it must NOT be marked tampered=True.
            raise HTTPException(
                status_code=409,
                detail=f"no alternative carrier available to tamper slot #{seq}; packet left unchanged",
            )

        target_data["original_media_id"] = orig_mid
        target_data["media_id"] = wrong_mid
        target_data["tampered"] = True
        _atomic_write_json(target_file, target_data)

        sidecar = user_dir / f"_decoded_{session_id}.json"
        sidecar.unlink(missing_ok=True)

        return _inspect_wire_session(user_dir, session_id, manifest)


@app.post("/api/wire/sessions/{session_id}/decode")
def wire_decode_from_disk(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Reassemble ONLY the surviving packet files on disk and decode via RS-ECC erasure recovery."""
    user_dir, manifest = _find_wire_session(session_id, current_user)
    manifest_path = user_dir / f"_manifest_{session_id}.json"
    with _get_session_lock(session_id):
        snapshot = _inspect_wire_session(user_dir, session_id, manifest)
        # Disk fingerprint taken WHILE holding the session lock: if the
        # packets/manifest change on disk mid-decode, the sidecar writeback
        # below is skipped so a stale decode result is never cached.
        disk_fp = tuple(
            (p["seq"], p["status"], p["media_id"]) for p in snapshot["packets"]
        )
    total_items = snapshot["total_items"]
    if snapshot["surviving_count"] == 0:
        raise HTTPException(status_code=409, detail="all packet files for this session were deleted from disk")

    slots: list[Optional[str]] = [
        None if pkt["status"] == "deleted" else pkt["media_id"] for pkt in snapshot["packets"]
    ]
    erasure_positions = list(snapshot["dropped_sequences"])
    engine_mode = snapshot["engine_mode"]
    diversity_mode = snapshot["diversity_mode"]
    ecc_parity_bytes = int(snapshot["ecc_parity_bytes"])

    session_key: Optional[bytes] = None
    if engine_mode == "dssc":
        sk_hex = manifest.get("session_key_hex")
        if not sk_hex:
            raise HTTPException(status_code=400, detail="DSSC session manifest missing session_key_hex")
        session_key = bytes.fromhex(sk_hex)

    t0 = time.perf_counter()
    engine = _get_engine()
    with _engine_work_lock:
        decode_res = engine.decode(
            media_ids=slots,
            mode=engine_mode,
            session_key=session_key,
            diversity_mode=diversity_mode,
            use_ecc=True,
            ecc_parity_bytes=ecc_parity_bytes,
            erasure_positions=erasure_positions or None,
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    ecc_success = bool(decode_res.success)
    reconstructed = (decode_res.reconstructed_message or "") if ecc_success else ""

    # Build raw unprotected preview directly from the surviving on-disk codeword bytes
    raw_bytes: list[int] = []
    if decode_res.exact_vcp_result is not None:
        raw_bytes = list(decode_res.exact_vcp_result.payload_symbols)
    elif decode_res.dssc_result is not None:
        raw_bytes = list(decode_res.dssc_result.raw_payload_bytes)

    framed_len = int(manifest.get("framed_bytes_len") or max(0, len(raw_bytes) - ecc_parity_bytes))
    erased_slot_set = set(erasure_positions)
    # Erased CODEWORD BYTE indices come from the dropped slots' byte_indices
    # (the per-slot codeword spans recorded in the manifest snapshot). Only
    # when no dropped slot carries byte_indices at all do we fall back to
    # erased_slot_set (exact_vcp's 1-byte-per-carrier layout, where the slot
    # number IS the byte index).
    erased_byte_set: set[int] = set()
    byte_indices_known = False
    for slot_pkt in snapshot["packets"]:
        if slot_pkt["status"] != "deleted":
            continue
        pkt_byte_indices = slot_pkt.get("byte_indices")
        if pkt_byte_indices:
            byte_indices_known = True
            erased_byte_set.update(int(b) for b in pkt_byte_indices)
    if not byte_indices_known:
        erased_byte_set = set(erased_slot_set)
    unprotected_chars: list[str] = []
    for b_idx in range(5, min(framed_len, len(raw_bytes))):
        # Mark erased bytes with the erased marker in BOTH modes
        # (exact_vcp and dssc) - dssc spreads each carrier over several
        # codeword bytes via byte_indices, so the old slot-number-only
        # check was wrong for it.
        if b_idx in erased_byte_set:
            unprotected_chars.append("\u25a1")
        else:
            val = raw_bytes[b_idx]
            unprotected_chars.append(chr(val) if 32 <= val <= 126 else "?")
    unprotected_preview = "".join(unprotected_chars)

    # Persist _decoded_<session_id>.json when recovery succeeds so Inbox reflects the real decode
    sidecar = user_dir / f"_decoded_{session_id}.json"
    if ecc_success:
        items_for_cache = []
        for slot_pkt in snapshot["packets"]:
            if slot_pkt["status"] == "deleted" or not slot_pkt["media_id"]:
                items_for_cache.append(
                    {
                        "media_id": "[DROPPED]",
                        "modality": "text",
                        "content": "[DROPPED PACKET — RECOVERED BY RS-ECC]",
                        "verified": False,
                    }
                )
            else:
                m_item = engine.index.get_by_id(slot_pkt["media_id"])
                items_for_cache.append(
                    {
                        "media_id": slot_pkt["media_id"],
                        "modality": m_item.modality if m_item else slot_pkt["modality"],
                        "content": (m_item.content[:200] if m_item else ""),
                        "verified": m_item is not None,
                    }
                )
        cached_body = {
            "session_id": session_id,
            "sender_username": snapshot["sender_username"],
            "recipient_username": snapshot["recipient_username"],
            "mode_used": snapshot["mode_used"],
            "reconstructed_meaning": reconstructed,
            "verification_rate": decode_res.verification_rate,
            "all_verified": len(erasure_positions) == 0,
            "ecc_success": True,
            "ecc_errors_fixed": sorted(int(p) for p in decode_res.ecc_fixed_errors),
            "items": items_for_cache,
            "cached": False,
            "elapsed_ms": elapsed_ms,
        }
        with _get_session_lock(session_id):
            # Re-validate the disk state before caching: only write the
            # decoded sidecar if the manifest still exists and the on-disk
            # packets are byte-for-byte the same ones we decoded (same
            # seq/status/media_id fingerprint). Otherwise skip the stale
            # writeback entirely.
            if manifest_path.exists():
                current_fp = tuple(
                    (p["seq"], p["status"], p["media_id"])
                    for p in _inspect_wire_session(user_dir, session_id, manifest)["packets"]
                )
                if current_fp == disk_fp:
                    _atomic_write_json(sidecar, cached_body)

    return {
        "session_id": session_id,
        "channel_dir": snapshot["channel_dir"],
        "engine_mode": engine_mode,
        "ecc_parity_bytes": ecc_parity_bytes,
        "total_packets": total_items,
        "surviving_packets": snapshot["surviving_count"],
        "dropped_packets": snapshot["dropped_count"],
        "tampered_packets": snapshot["tampered_count"],
        "dropped_sequences": erasure_positions,
        "tampered_sequences": snapshot["tampered_sequences"],
        "ecc_success": ecc_success,
        "crc_verified": ecc_success,
        "reconstructed_message": reconstructed,
        "unprotected_preview": unprotected_preview,
        "repaired_byte_indices": sorted(int(p) for p in decode_res.ecc_fixed_errors),
        "verification_rate": round(decode_res.verification_rate, 4),
        "elapsed_ms": elapsed_ms,
    }


@app.delete("/api/wire/sessions/{session_id}")
def wire_delete_session(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Delete an entire wire session (manifest, packet files, decoded cache, and archive)."""
    import shutil

    user_dir, _ = _find_wire_session(session_id, current_user)
    with _get_session_lock(session_id):
        deleted = 0
        for candidate in (
            [user_dir / f"_manifest_{session_id}.json", user_dir / f"_decoded_{session_id}.json"]
            + list(user_dir.glob(f"{session_id}__*.json"))
        ):
            if candidate.exists():
                # missing_ok: a concurrent drop/delete of the same file must
                # never raise FileNotFoundError.
                candidate.unlink(missing_ok=True)
                deleted += 1
        archive_dir = _dropped_archive_dir(user_dir, session_id)
        if archive_dir.exists():
            shutil.rmtree(archive_dir, ignore_errors=True)
    # The session is gone from disk: release its lock-map entry (only if no
    # other thread currently holds the lock).
    _release_session_lock_entry(session_id)
    return {"success": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Demo showcase router: live steganalysis, carrier verification, pipeline traces
# Mounted after route definitions so require_api_token is already defined.
# ---------------------------------------------------------------------------
from src.api.demo import router as demo_router  # noqa: E402

app.include_router(
    demo_router,
    prefix="/api/demo",
    dependencies=[Depends(require_api_token)],
)

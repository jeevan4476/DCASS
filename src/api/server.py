# src/api/server.py
"""
DCASS FastAPI Backend.

Exposes the DCASS engine (encode, decode, search, status, benchmark)
as a REST API for the Next.js frontend.

Usage:
    uvicorn src.api.server:app --reload --port 8000
"""

from __future__ import annotations

import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.corpus.index.unified_index import resolve_indices_base_path


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DCASS API",
    description="Dynamic Context-Aware Semantic Steganography",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy engine singletons
# ---------------------------------------------------------------------------
_encoder = None
_decoder = None
_initializing = False
_ready = False

# Transmission state
_transmission_active = False
_transmission_progress = {"current": 0, "total": 0, "status": "idle"}
_transmission_lock = threading.Lock()


def _get_encoder():
    global _encoder
    if _encoder is None:
        from src.engine.encoder import SemanticEncoder
        _encoder = SemanticEncoder(expand_synonyms=True)
        _encoder.load()
    return _encoder


def _get_decoder():
    global _decoder
    if _decoder is None:
        from src.engine.decoder import SemanticDecoder
        _decoder = SemanticDecoder()
        _decoder.load()
    return _decoder


def warmup():
    """Pre-load encoder and decoder on startup."""
    global _initializing, _ready
    _initializing = True
    print("\n" + "=" * 70)
    print("🔥 Warming up DCASS engine...")
    print("=" * 70)
    try:
        # Load encoder (this triggers CLIP model loading)
        print("\n📦 Loading encoder and CLIP model...")
        encoder = _get_encoder()
        print(f"✅ Encoder ready: {encoder}")
        
        # Load decoder
        print("\n📦 Loading decoder...")
        decoder = _get_decoder()
        print(f"✅ Decoder ready: {decoder}")
        
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
    mode: Literal["best", "round_robin", "balanced"] = "best"
    modalities: list[str] = Field(default=["image", "text", "audio"])


class EncodeResponse(BaseModel):
    media_ids: list[str]
    chunks: list[str]
    encoded: list[dict]
    modality_breakdown: dict[str, int]
    elapsed_ms: float


class DecodeRequest(BaseModel):
    media_ids: list[str]


class DecodeResponse(BaseModel):
    reconstructed_meaning: str
    items: list[dict]
    verification_rate: float
    all_verified: bool
    elapsed_ms: float


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
def encode(req: EncodeRequest):
    t0 = time.perf_counter()
    try:
        encoder = _get_encoder()
        result = encoder.encode(req.message, modalities=req.modalities, diversity_mode=req.mode)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    encoded_items = []
    for enc in result.encoded:
        encoded_items.append({
            "media_id": enc.media.id,
            "modality": enc.media.modality,
            "score": round(enc.media.normalized_score, 4),
            "content": enc.media.content[:120],
        })

    return EncodeResponse(
        media_ids=result.media_ids,
        chunks=[c.original for c in result.chunks],
        encoded=encoded_items,
        modality_breakdown=result.modality_breakdown,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest):
    t0 = time.perf_counter()
    decoder = _get_decoder()
    result = decoder.decode(req.media_ids)

    items = []
    for d in result.decoded:
        items.append({
            "media_id": d.media_id,
            "modality": d.modality,
            "content": d.content[:200],
            "verified": d.verified,
        })

    return DecodeResponse(
        reconstructed_meaning=result.reconstructed_meaning,
        items=items,
        verification_rate=result.verification_rate,
        all_verified=result.all_verified,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    encoder = _get_encoder()
    results = encoder.index.search(req.query, k=req.k, modalities=req.modalities)

    items = []
    for r in results:
        items.append({
            "id": r.id,
            "modality": r.modality,
            "score": round(r.normalized_score, 4),
            "content": r.content[:200],
        })

    return SearchResponse(
        results=items,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/api/status", response_model=StatusResponse)
def status():
    import torch
    indices_path = resolve_indices_base_path()
    models_path = Path(__file__).parent.parent.parent / "storage" / "models"

    index_info = {}
    total = 0
    for mod in ["image", "text", "audio"]:
        idx_file = indices_path / f"{mod}.index"
        meta_file = indices_path / f"{mod}_metadata.json"
        if idx_file.exists() and meta_file.exists():
            try:
                import faiss
                idx = faiss.read_index(str(idx_file))
                count = idx.ntotal
                total += count
                index_info[mod] = {"status": "ok", "count": count}
            except Exception as e:
                index_info[mod] = {"status": "error", "error": str(e)}
        else:
            index_info[mod] = {"status": "missing"}

    stealth = {
        "gan_checkpoint": (models_path / "gan" / "final.pt").exists(),
        "rl_checkpoint": (models_path / "rl" / "ppo_agent_final.pt").exists(),
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
    # Consider ready if both encoder and decoder are loaded (even without warmup)
    is_ready = _ready or (_encoder is not None and _decoder is not None)
    return {
        "ready": is_ready,
        "initializing": _initializing,
        "encoder_loaded": _encoder is not None,
        "decoder_loaded": _decoder is not None,
    }


@app.get("/api/benchmark/latest")
def benchmark_latest():
    results_dir = Path(__file__).parent.parent.parent / "storage" / "data" / "benchmarks" / "results"
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
        return {"packets": [], "count": 0, "error": "shared_channel directory not found"}
    
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
def clear_wire_packets():
    """Clear all packets from shared_channel directory."""
    shared_dir = Path(__file__).parent.parent.parent / "storage" / "shared_channel"
    
    if not shared_dir.exists():
        return {"success": False, "error": "shared_channel directory not found"}
    
    deleted_count = 0
    try:
        for f in shared_dir.glob("*.json"):
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
    global _transmission_active, _transmission_progress
    
    items = schedule["items"]
    delays = schedule["delays"]
    channels = schedule["channels"]
    mode_used = schedule["mode_used"]
    
    with _transmission_lock:
        _transmission_active = True
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
            if media_id is None:
                # Noise gap - just wait
                actual_delay = delay / speed_multiplier
                if actual_delay > 0:
                    time.sleep(actual_delay)
                continue
            
            # Write packet
            packet = {
                "media_id": media_id,
                "channel_id": channel,
                "sequence_number": idx,
                "delay_seconds": round(delay, 3),
                "timestamp": time.time(),
                "mode_used": mode_used,
            }
            
            filename = f"{media_id}_{channel}_{idx:04d}.json"
            with open(shared_dir / filename, "w") as f:
                json.dump(packet, f, indent=2)
            
            packet_count += 1
            
            # Update progress
            with _transmission_lock:
                _transmission_progress["current"] = packet_count
            
            # Apply delay before next packet (scaled by speed_multiplier)
            actual_delay = delay / speed_multiplier
            if actual_delay > 0 and idx < len(items) - 1:
                time.sleep(actual_delay)
        
        # Mark as complete
        with _transmission_lock:
            _transmission_progress["status"] = "complete"
            _transmission_active = False
            
    except Exception as e:
        with _transmission_lock:
            _transmission_progress["status"] = f"error: {str(e)}"
            _transmission_active = False
        raise


@app.post("/api/transmit")
def transmit_sequence(req: TransmitRequest, background_tasks: BackgroundTasks):
    """
    Start transmitting a media sequence through the shared channel.
    
    This starts a background task that writes packets with real delays,
    simulating realistic transmission timing.
    """
    global _transmission_active, _transmission_progress
    
    # Check if already transmitting
    if _transmission_active:
        raise HTTPException(
            status_code=409,
            detail="Transmission already in progress. Wait for it to complete or check /api/transmit/status"
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transmit/status")
def get_transmission_status():
    """Get the current transmission status."""
    return {
        "active": _transmission_active,
        **_transmission_progress,
    }


@app.post("/api/transmit/stop")
def stop_transmission():
    """Stop the current transmission (best effort)."""
    global _transmission_active, _transmission_progress
    
    with _transmission_lock:
        if _transmission_active:
            _transmission_active = False
            _transmission_progress["status"] = "stopped"
            return {"success": True, "message": "Transmission stop requested"}
        else:
            return {"success": False, "message": "No active transmission"}

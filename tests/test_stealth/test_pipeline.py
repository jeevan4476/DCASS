# tests/test_stealth/test_pipeline.py
"""
Phase F end-to-end pipeline tests.

The static-mode round-trip test is the baseline — it needs only the engine
(FAISS indices + voronoi codebook), no GAN/RL checkpoints. The gan/rl/auto
tests are gated behind `pytest -m needs_models` because they require Phase A
and Phase B checkpoints under storage/models/.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.stealth.pipeline import run_pipeline

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _engine_available() -> bool:
    """
    Skip the round-trip tests when the engine's required artefacts aren't
    present. Running without a corpus + codebook would raise mid-pipeline;
    the skip keeps the suite green on a bare clone.
    """
    try:
        from src.corpus.index.unified_index import resolve_indices_base_path

        indices = resolve_indices_base_path()
        if not (indices / "voronoi_codebook.npz").exists():
            return False
        for mod in ("image", "text", "audio"):
            if (indices / f"{mod}.index").exists():
                return True
        return False
    except Exception:
        return False


ENGINE_REQUIRED = pytest.mark.skipif(
    not _engine_available(), reason="corpus indices / voronoi codebook not present"
)


@ENGINE_REQUIRED
def test_round_trip_static(tmp_path):
    """Baseline — no GAN/RL needed. Round-trip through static NoiseController."""
    result = run_pipeline(
        message="hello world",
        sender="alice",
        recipient="bob",
        mode="static",
        speed_multiplier=1000.0,  # effectively zero sleeps
        shared_dir=tmp_path,
    )
    assert result.decoded_ok, (
        f"round-trip failed: original={result.original_message!r} "
        f"decoded={result.decoded_message!r} error={result.error}"
    )
    assert result.mode_used == "static"
    assert len(result.schedule["delays"]) == len(result.schedule["items"])


@ENGINE_REQUIRED
@pytest.mark.needs_models
def test_round_trip_gan(tmp_path):
    """Needs storage/models/gan_generator.pt (Phase A)."""
    gan_ckpt = _PROJECT_ROOT / "storage" / "models" / "gan_generator.pt"
    if not gan_ckpt.exists():
        pytest.skip(f"missing GAN checkpoint at {gan_ckpt}")

    result = run_pipeline(
        message="hello world",
        sender="alice",
        recipient="bob",
        mode="gan",
        speed_multiplier=1000.0,
        shared_dir=tmp_path,
    )
    assert result.decoded_ok, result.error or (
        f"decoded={result.decoded_message!r} != original={result.original_message!r}"
    )
    assert result.mode_used == "gan"


@ENGINE_REQUIRED
@pytest.mark.needs_models
def test_round_trip_rl(tmp_path):
    """Needs storage/models/rl_agent.pt (Phase B)."""
    rl_ckpt = _PROJECT_ROOT / "storage" / "models" / "rl_agent.pt"
    if not rl_ckpt.exists():
        pytest.skip(f"missing RL checkpoint at {rl_ckpt}")

    result = run_pipeline(
        message="hi",
        sender="alice",
        recipient="bob",
        mode="rl",
        speed_multiplier=1000.0,
        shared_dir=tmp_path,
    )
    assert result.decoded_ok, result.error or (
        f"decoded={result.decoded_message!r} != original={result.original_message!r}"
    )
    assert result.mode_used == "rl"


@ENGINE_REQUIRED
def test_pipeline_recipient_isolation(tmp_path):
    """
    Two pipelines to DIFFERENT recipients, run back-to-back in the same
    tmp_path, must not leak packets into each other's subdirectories.
    """
    r1 = run_pipeline(
        message="for bob",
        sender="alice",
        recipient="bob",
        mode="static",
        speed_multiplier=1000.0,
        shared_dir=tmp_path,
        cleanup=False,  # keep around so we can inspect
    )
    r2 = run_pipeline(
        message="for charlie",
        sender="alice",
        recipient="charlie",
        mode="static",
        speed_multiplier=1000.0,
        shared_dir=tmp_path,
        cleanup=False,
    )
    assert r1.decoded_ok and r2.decoded_ok

    # Each recipient's subdir should contain exactly its own session's
    # packets — no cross-leakage from the other recipient.
    for sub in tmp_path.iterdir():
        if not sub.is_dir():
            continue
        session_ids_in_dir = {
            p.name.split("__", 1)[0]
            for p in sub.glob("*.json")
            if "__" in p.name
        }
        # Only this recipient's session_id should appear in this subdir.
        assert session_ids_in_dir in (
            {r1.session_id},
            {r2.session_id},
            set(),  # noise-only / empty dirs are fine
        ), f"cross-recipient packet leakage into {sub}: {session_ids_in_dir}"

# tests/test_checkpoint_io.py
"""Fail-closed behaviour of the shared checkpoint loader."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn


class _LegacyWidget:
    """A pickled class the strict unpickler refuses, for the refusal paths."""


def _state_dict() -> dict:
    return {"layer.weight": nn.Linear(3, 2).state_dict()["weight"]}


def test_strict_load_accepts_plain_state_dicts(tmp_path):
    from src.utils.checkpoint_io import load_trained_checkpoint

    original = _state_dict()
    path = tmp_path / "plain.pt"
    torch.save(original, path)

    ck = load_trained_checkpoint(path)
    assert torch.equal(ck["layer.weight"], original["layer.weight"])


def test_unverified_pickle_content_is_refused(tmp_path):
    from src.utils.checkpoint_io import load_trained_checkpoint

    path = tmp_path / "widget.pt"
    torch.save({"widget": _LegacyWidget()}, path)

    with pytest.raises(RuntimeError, match="no sha256 entry"):
        load_trained_checkpoint(path, manifest_path=tmp_path / "missing.json")


def test_hash_pinned_legacy_checkpoint_loads_via_fallback(tmp_path, capsys):
    from src.utils.checkpoint_io import load_trained_checkpoint, sha256_of

    path = tmp_path / "legacy.pt"
    torch.save({"widget": _LegacyWidget()}, path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"checkpoints": {str(path.resolve()): {"sha256": sha256_of(path)}}})
    )

    ck = load_trained_checkpoint(path, manifest_path=manifest)
    assert isinstance(ck["widget"], _LegacyWidget)
    assert "full-pickle" in capsys.readouterr().out


def test_byte_mismatch_against_manifest_is_refused(tmp_path):
    from src.utils.checkpoint_io import load_trained_checkpoint

    path = tmp_path / "tampered.pt"
    torch.save({"widget": _LegacyWidget()}, path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"checkpoints": {str(path.resolve()): {"sha256": "0" * 64}}})
    )

    with pytest.raises(RuntimeError, match="does not match"):
        load_trained_checkpoint(path, manifest_path=manifest)

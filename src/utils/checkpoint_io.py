"""Fail-closed torch-checkpoint loading for the whole repository.

Every ``torch.load`` of a project-trained checkpoint goes through
``load_trained_checkpoint``:

1. The strict loader (``weights_only=True``) runs first, with the
   project's own pickled config classes allowlisted. No executable pickle
   content passes.
2. If the strict unpickler refuses (an allowlist miss from an older
   checkpoint format), the file bytes are verified against
   ``storage/models/checkpoints_manifest.json`` before the full-pickle
   loader is permitted.
3. A checkpoint that neither loads strictly nor verifies against the
   manifest is refused outright. Loading a modified checkpoint silently
   is the worst possible failure for a research record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath, PosixPath, WindowsPath

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / "storage" / "models" / "checkpoints_manifest.json"


def _project_global_allowlist() -> list:
    """The project's own pickled config classes that appear inside
    checkpoints. Only dataclass config records are allowlisted; arbitrary
    classes and functions never pass."""
    allowed: list = [PosixPath, WindowsPath, PurePosixPath, PureWindowsPath]
    try:  # numpy scalars embedded in legacy checkpoints are data, not code

        try:
            from numpy._core.multiarray import scalar as _np_scalar
        except ImportError:  # numpy < 2
            from numpy.core.multiarray import scalar as _np_scalar
        allowed.append(_np_scalar)
    except Exception:  # noqa: BLE001 - allowlist must degrade, not crash
        pass
    try:
        from src.stealth.gan.trainer import TrainingConfig, TrainingMetrics

        allowed.append(TrainingConfig)
        allowed.append(TrainingMetrics)
    except Exception:  # noqa: BLE001 - allowlist must degrade, not crash
        pass
    try:
        from src.stealth.rl.agent import PPOConfig

        allowed.append(PPOConfig)
    except Exception:  # noqa: BLE001
        pass
    return allowed


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_hash(manifest_path: Path, path: Path) -> str | None:
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:  # noqa: BLE001 - unreadable manifest = no record
        return None
    entries = manifest.get("checkpoints", {})
    resolved = Path(path).resolve()
    candidates: list[str] = []
    try:
        candidates.append(str(resolved.relative_to(_PROJECT_ROOT)))
    except ValueError:  # file outside the project tree: absolute key only
        pass
    candidates.append(str(resolved))
    for key in candidates:
        entry = entries.get(key)
        if isinstance(entry, dict):
            recorded = entry.get("sha256")
            if recorded:
                return recorded
    return None


def load_trained_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
):
    """Load a project checkpoint, refusing unverified full pickle content.

    A strict unpickle failure is allowed to fall back to the full loader
    only when the file's sha256 matches the manifest. Anything else is a
    refusal, never a silent trust decision.
    """
    checkpoint_path = Path(path)
    try:
        with torch.serialization.safe_globals(_project_global_allowlist()):
            return torch.load(
                checkpoint_path, map_location=map_location, weights_only=True
            )
    except Exception as strict_error:  # noqa: BLE001 - fall through to verify
        recorded = _recorded_hash(manifest_path, checkpoint_path)
        if recorded is None:
            raise RuntimeError(
                f"Refusing full-pickle load of {checkpoint_path}: the strict "
                f"loader rejected it ({type(strict_error).__name__}) and it "
                "has no sha256 entry in checkpoints_manifest.json. Verify the "
                "checkpoint's origin, then record its hash in the manifest."
            ) from strict_error
        if sha256_of(checkpoint_path) != recorded:
            raise RuntimeError(
                f"Refusing to load {checkpoint_path}: its sha256 does not "
                f"match the recorded manifest entry ({recorded[:16]}...). "
                "The checkpoint bytes changed after verification."
            ) from strict_error
        print(
            f"checkpoint_io: {checkpoint_path.name} verified against the "
            "manifest but needs the full-pickle loader (legacy format)"
        )
        return torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

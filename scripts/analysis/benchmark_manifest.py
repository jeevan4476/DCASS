#!/usr/bin/env python3
"""
Run manifest helper for the benchmark scripts.

A manifest records enough about a run to reproduce it later: the exact
command, the git state, the timestamp, the seeds, the relevant package
versions, the inputs that were read, and the hashes of the outputs that
were written. It is written next to the artifact it describes.

This is provenance only. It never changes what a benchmark measures.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "numpy", "scipy", "sklearn", "faiss", "matplotlib"):
        try:
            module = __import__(name)
        except Exception:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def build_manifest(
    component: str,
    outputs: Iterable[Path],
    inputs: Iterable[Path] = (),
    seeds: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict for a benchmark run."""
    input_records = []
    for path in inputs:
        p = Path(path)
        if p.exists():
            input_records.append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
        else:
            input_records.append({"path": str(p), "missing": True})

    output_records = []
    for path in outputs:
        p = Path(path)
        if p.exists():
            output_records.append(
                {
                    "path": str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
        else:
            output_records.append({"path": str(p), "missing": True})

    status = _git("status", "--porcelain")
    manifest = {
        "component": component,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "argv": list(sys.argv),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "dirty_paths": status.splitlines() if status else [],
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "packages": package_versions(),
        "seeds": seeds or {},
        "inputs": input_records,
        "outputs": output_records,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def recorded_commit(artifact_path: Path) -> str | None:
    """The commit whose code produced the measurement in *artifact_path*.

    A table regenerated from a recorded artifact must carry the commit that
    made the measurement, not the commit that happens to be checked out.
    This is what keeps a regenerated table identical to the committed one.
    """
    path = Path(artifact_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    commit = provenance.get("git_commit")
    return str(commit) if commit else None


def scale_table_to_textwidth(path: Path) -> bool:
    """Rewrite a generated table as a full-width, width-scaled float.

    Six of the generated tables are wider than a single-column text block.
    Emitting the float as a full-width `table*` and scaling the tabular to
    the text width with `\\resizebox` makes them fit in both a one-column
    and a two-column paper.

    The transformation is idempotent: a table that already carries the
    scaled form is left alone.

    `\\resizebox` needs graphicx, which the integration guide already
    requires.
    """
    import re

    path = Path(path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if "\\resizebox" in text:
        return False

    updated = re.sub(r"\\begin\{table\}\[", r"\\begin{table*}[", text)
    updated = updated.replace(r"\end{table}", r"\end{table*}")
    updated = updated.replace(
        r"\begin{tabular}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
    )
    updated = updated.replace(r"\end{tabular}", "\\end{tabular}}")
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def latex_escape(text: str) -> str:
    """Escape the characters that would break a LaTeX caption.

    Commit hashes contain underscores, which LaTeX reads as a subscript
    operator and rejects outside math mode. Percent signs and other specials
    are escaped too.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def caption_provenance(
    n: str,
    scope: str,
    commit: str | None = None,
) -> str:
    """Build the standard caption suffix: sample size, scope, and commit.

    Every generated table carries this so a reader can tell what the table
    measured, over how many trials, and which commit produced it.

    The commit is the one whose code produced the *measurement*, not the
    commit that happened to regenerate the table. Pass it explicitly from
    the artifact's own provenance block; when it is absent the current HEAD
    is used, which is only accurate if the table was regenerated in the
    same commit as its measurement.
    """
    if commit is None:
        commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    parts = [
        f"n = {latex_escape(n)}",
        f"scope = {latex_escape(scope)}",
        f"commit = {latex_escape(commit)}",
    ]
    return " (" + "; ".join(parts) + ")"

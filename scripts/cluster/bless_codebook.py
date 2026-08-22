#!/usr/bin/env python3
# scripts/cluster/bless_codebook.py
"""
Bind a Voronoi codebook to the current index set via content fingerprints (WP-3).

Writes `voronoi_codebook.meta.json` beside the .npz recording per-modality
ntotal/d plus a cheap content fingerprint per index:

    sha256(reconstruct(0) || reconstruct(ntotal-1) || ntotal)

`VCPPayloadMapper.load()` refuses to run when the sidecar exists but the live
indices no longer match - so rebuilding an index without refitting produces a
loud refusal instead of silently wrong bytes.

Two modes:
  --check   Verify only (exit non-zero on mismatch). Default.
  --bless   Certify the CURRENT pairing. One-time assertion for codebooks
            fitted before this mechanism existed; gated behind the Phase 0
            count check (sum(ntotal) == len(cluster_assignments)) passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def compute_fingerprints(base_path: Path) -> tuple[dict, dict]:
    """Return ({modality: {fingerprint info}}, {errors}) from the live indices."""
    import faiss

    out = {}
    errors = {}
    for modality in ("image", "text", "audio"):
        idx_path = base_path / f"{modality}.index"
        meta_path = base_path / f"{modality}_metadata.json"
        if not idx_path.exists():
            errors[modality] = "index file missing"
            continue
        try:
            index = faiss.read_index(str(idx_path))
            ntotal = int(index.ntotal)
            d = int(index.d)
            v0 = np.asarray(index.reconstruct(0), dtype=np.float32).tobytes()
            vn = (
                np.asarray(index.reconstruct(ntotal - 1), dtype=np.float32).tobytes()
                if ntotal > 1
                else b""
            )
            blob = v0 + vn + str(ntotal).encode()
            meta_len = None
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_len = len(json.load(f))
            out[modality] = {
                "ntotal": ntotal,
                "d": d,
                "metadata_len": meta_len,
                "fingerprint": hashlib.sha256(blob).hexdigest()[:16],
            }
        except Exception as e:
            errors[modality] = str(e)
    return out, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bless", action="store_true", help="Certify the current pairing (writes the sidecar)"
    )
    parser.add_argument("--check", action="store_true", help="Verify only (default)")
    args = parser.parse_args()

    from src.corpus.index.unified_index import resolve_indices_base_path

    base_path = resolve_indices_base_path()
    sidecar_path = base_path / "voronoi_codebook.meta.json"
    npz_path = base_path / "voronoi_codebook.npz"

    if not npz_path.exists():
        print(f"ERROR: no codebook at {npz_path}")
        return 1

    print(f"Index set: {base_path}")
    fingerprints, errors = compute_fingerprints(base_path)
    for m, e in errors.items():
        print(f"  ERROR [{m}]: {e}")

    # Phase 0 gate: count check must pass before blessing means anything.
    z = np.load(npz_path)
    assignments_len = int(len(z["cluster_assignments"]))
    sum_ntotal = sum(v["ntotal"] for v in fingerprints.values())
    count_ok = assignments_len == sum_ntotal
    print(
        f"\nCount check: sum(ntotal)={sum_ntotal:,} vs "
        f"len(cluster_assignments)={assignments_len:,} -> "
        f"{'PASS' if count_ok else 'FAIL'}"
    )

    if args.bless:
        if not count_ok:
            print(
                "\nREFUSING to bless: the count check failed. The codebook was "
                "fitted against a different index set; re-fit or rebuild first."
            )
            return 1
        sidecar = {
            "blessed_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "num_clusters": int(z["num_clusters"]) if "num_clusters" in z else 256,
            "dim": int(z["dim"]) if "dim" in z else None,
            "delta_margin": float(z["delta_margin"]) if "delta_margin" in z else None,
            "index_fingerprints": {
                m: {"fingerprint": v["fingerprint"], "ntotal": v["ntotal"]}
                for m, v in fingerprints.items()
            },
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        print(f"\nBLESSED pairing certified -> {sidecar_path}")
        for m, v in sorted(fingerprints.items()):
            print(f"  {m}: fp={v['fingerprint']} ntotal={v['ntotal']:,}")
        return 0

    # --check mode
    if not sidecar_path.exists():
        print(f"\nNo sidecar at {sidecar_path}. Run with --bless to certify the current pairing.")
        return 1
    sidecar = json.loads(sidecar_path.read_text())
    expected = sidecar.get("index_fingerprints", {})
    ok = True
    for modality, info in sorted(fingerprints.items()):
        exp = expected.get(modality, {}).get("fingerprint")
        status = "MATCH" if exp == info["fingerprint"] else "MISMATCH"
        if exp != info["fingerprint"]:
            ok = False
        print(f"  {modality}: live={info['fingerprint']} sidecar={exp} -> {status}")
    missing = [m for m in ("image", "text", "audio") if m not in fingerprints]
    for m in missing:
        print(f"  {m}: MISSING - cannot verify")
        ok = False
    print(f"\nVERDICT: {'BINDING OK' if ok else 'BINDING BROKEN - decode would be WRONG'}")
    return 0 if ok else 1


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

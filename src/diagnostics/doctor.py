# src/diagnostics/doctor.py
"""
DCASS runtime diagnostics ("dcass doctor").

One command that validates the entire runtime and prints a verdict table:
dependencies importable, index counts, metadata/ntotal agreement,
codebook presence and fingerprint binding, cluster population histogram,
empty-cluster list, checkpoint presence, and disk footprint.

This is the instrument that produces the Phase-0 ground-truth numbers and
the definition of "the system is ready". Exits non-zero on any hard failure
when used from the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Population buckets for the cluster histogram (Phase 0 item 4).
POPULATION_BUCKETS = [(0, 0), (1, 4), (5, 20), (21, None)]


@dataclass
class CheckResult:
    name: str
    ok: bool
    hard: bool  # hard failures make `doctor` exit non-zero
    detail: str = ""

    def cell(self) -> str:
        mark = "PASS" if self.ok else ("FAIL" if self.hard else "WARN")
        return f"[{mark}] {self.name}" + (f" - {self.detail}" if self.detail else "")


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)
    phase0: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok or not c.hard for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "ok": c.ok, "hard": c.hard, "detail": c.detail}
                for c in self.checks
            ],
            "phase0": self.phase0,
        }

    def to_public_dict(self, project_root: Optional[Path] = None) -> dict:
        """
        API-safe report: scrub absolute host paths from check details.

        CLI `render()` keeps absolute paths for local debugging; remote
        callers get relative-to-project (or basename) strings only.
        """
        root = (project_root or Path(__file__).resolve().parent.parent.parent).resolve()
        root_s = str(root)

        def _scrub(text: str) -> str:
            if not text:
                return text
            if root_s in text:
                return text.replace(root_s, ".")
            # Also collapse any other absolute path segments to their basename
            # when the detail is a lone path.
            p = Path(text)
            if p.is_absolute():
                try:
                    return str(p.relative_to(root))
                except ValueError:
                    return p.name
            return text

        data = self.to_dict()
        for check in data["checks"]:
            check["detail"] = _scrub(check.get("detail", ""))
        return data

    def render(self) -> str:
        lines = ["DCASS DOCTOR", "=" * 60]
        for c in self.checks:
            lines.append(c.cell())
        if self.phase0:
            lines.append("")
            lines.append("Phase 0 ground truth:")
            lines.append(json.dumps(self.phase0, indent=2))
        lines.append("")
        lines.append("VERDICT: READY" if self.ok else "VERDICT: NOT READY (see FAIL above)")
        return "\n".join(lines)


def _check_dependencies(report: DoctorReport):
    import importlib

    deps = [
        ("numpy", False),
        ("faiss", True),
        ("torch", True),
        ("reedsolo", True),
        ("sentence_transformers", False),
        ("fastapi", False),
    ]

    for mod, hard in deps:
        try:
            importlib.import_module(mod)
            report.checks.append(CheckResult(f"dep:{mod}", True, hard))
        except Exception as e:
            report.checks.append(CheckResult(f"dep:{mod}", False, hard, str(e)[:80]))


def _content_fingerprint(index, modality: str) -> Optional[str]:
    """sha256(reconstruct(0) || reconstruct(ntotal-1) || ntotal) - cheap rebuild detector."""
    import hashlib

    idx = index.indices.get(modality)
    if idx is None or not hasattr(idx, "reconstruct") or idx.ntotal == 0:
        return None
    try:
        v0 = np.asarray(idx.reconstruct(0), dtype=np.float32).tobytes()
        vn = (
            np.asarray(idx.reconstruct(idx.ntotal - 1), dtype=np.float32).tobytes()
            if idx.ntotal > 1
            else b""
        )
        blob = v0 + vn + str(idx.ntotal).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return None


def run_doctor(base_path: Optional[Path] = None) -> DoctorReport:
    """Run all diagnostics and return a structured report."""
    report = DoctorReport()

    _check_dependencies(report)

    # ------------------------------------------------------------------
    # Indices + metadata agreement
    # ------------------------------------------------------------------
    from src.corpus.index.unified_index import UnifiedSemanticIndex

    try:
        index = UnifiedSemanticIndex(base_path=base_path) if base_path else UnifiedSemanticIndex()
        status = index.load()
        loaded_ok = all(status.values())
        report.checks.append(
            CheckResult(
                "indices:load",
                loaded_ok,
                hard=True,
                detail=", ".join(f"{m}:{'ok' if ok else 'MISSING'}" for m, ok in status.items()),
            )
        )
    except Exception as e:
        report.checks.append(CheckResult("indices:load", False, True, str(e)[:120]))
        return report

    phase0: dict = {"modalities": {}}
    total_ntotal = 0
    agreement_ok = True
    agreement_detail = []
    fingerprints = {}
    for modality in ("image", "text", "audio"):
        idx = index.indices.get(modality)
        meta_list = index.metadata.get(modality, [])
        if idx is None:
            continue
        ntotal = int(idx.ntotal)
        total_ntotal += ntotal
        agree = len(meta_list) == ntotal
        agreement_ok &= agree
        agreement_detail.append(f"{modality}: {len(meta_list) == ntotal}")
        dim = int(idx.d) if hasattr(idx, "d") else None
        phase0["modalities"][modality] = {
            "ntotal": ntotal,
            "dim": dim,
            "metadata_len": len(meta_list),
            "agreement": agree,
        }
        fp = _content_fingerprint(index, modality)
        if fp:
            fingerprints[modality] = fp
    phase0["modalities"]["fingerprints"] = fingerprints

    report.checks.append(
        CheckResult(
            "indices:metadata_agreement",
            agreement_ok,
            hard=True,
            detail=" ".join(agreement_detail),
        )
    )
    report.checks.append(CheckResult("indices:total_items", True, False, f"{total_ntotal:,}"))

    # ------------------------------------------------------------------
    # Codebook + cluster population histogram (Phase 0 items 3-5)
    # ------------------------------------------------------------------
    from src.corpus.cluster.voronoi_codebook import VoronoiCodebook
    from src.corpus.index.unified_index import resolve_indices_base_path

    indices_dir = base_path if base_path is not None else resolve_indices_base_path()
    codebook_path = indices_dir / "voronoi_codebook.npz"
    cb_present = codebook_path.exists()
    report.checks.append(
        CheckResult("codebook:present", cb_present, hard=True, detail=str(codebook_path))
    )

    assignments_len = None
    if cb_present:
        try:
            codebook = VoronoiCodebook()
            codebook.load(codebook_path)
            assignments = np.asarray(codebook.cluster_assignments)
            assignments_len = int(len(assignments))

            counts = np.bincount(assignments, minlength=codebook.num_clusters)
            histogram = {}
            empty = []
            for lo, hi in POPULATION_BUCKETS:
                label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
                mask = (counts >= lo) & (
                    (counts <= hi) if hi is not None else np.ones_like(counts, dtype=bool)
                )
                histogram[label] = int(mask.sum())
            empty = [int(c) for c in np.where(counts == 0)[0]]

            phase0["codebook"] = {
                "num_clusters": int(codebook.num_clusters),
                "dim": int(codebook.dim),
                "delta_margin": float(codebook.delta_margin),
                "cluster_assignments_len": assignments_len,
                "sum_ntotal": total_ntotal,
                "population_histogram": histogram,
                "empty_clusters": len(empty),
                "min_density": int(counts.min()) if len(counts) else 0,
            }

            # STOP-THE-LINE check: sum(ntotal) == len(cluster_assignments)
            count_match = assignments_len == total_ntotal
            phase0["codebook"]["count_check_pass"] = count_match
            report.checks.append(
                CheckResult(
                    "codebook:index_binding_count",
                    count_match,
                    hard=True,
                    detail=(
                        f"sum(ntotal)={total_ntotal:,} vs len(cluster_assignments)={assignments_len:,}"
                        + ("" if count_match else " - CODEBOOK FIT AGAINST A DIFFERENT INDEX SET")
                    ),
                )
            )
            report.checks.append(
                CheckResult(
                    "codebook:empty_clusters",
                    len(empty) == 0,
                    hard=False,
                    detail=f"{len(empty)} empty of {codebook.num_clusters}"
                    + (f": {empty[:12]}" if empty else ""),
                )
            )
            report.checks.append(
                CheckResult(
                    "codebook:min_density",
                    bool(len(counts)) and int(counts.min()) >= 5,
                    hard=False,
                    detail=f"min={int(counts.min()) if len(counts) else 0} carriers "
                    "(<5 limits avoid_duplicates on repeated bytes)",
                )
            )

            # Fingerprint sidecar check (WP-3)
            meta_sidecar = indices_dir / "voronoi_codebook.meta.json"
            if meta_sidecar.exists():
                try:
                    sidecar = json.loads(meta_sidecar.read_text())
                    expected_raw = sidecar.get("index_fingerprints", {})
                    expected = {
                        m: (v.get("fingerprint") if isinstance(v, dict) else v)
                        for m, v in expected_raw.items()
                    }
                    match = expected == fingerprints
                    report.checks.append(
                        CheckResult(
                            "codebook:fingerprint_match",
                            match,
                            hard=True,
                            detail="pairing certified"
                            if match
                            else f"MISMATCH: sidecar={expected} live={fingerprints}",
                        )
                    )
                except Exception as e:
                    report.checks.append(
                        CheckResult(
                            "codebook:fingerprint_match", False, True, f"unreadable sidecar: {e}"
                        )
                    )
            else:
                report.checks.append(
                    CheckResult(
                        "codebook:fingerprint_match",
                        False,
                        hard=False,
                        detail="no voronoi_codebook.meta.json - run scripts/cluster/bless_codebook.py --bless",
                    )
                )
        except Exception as e:
            report.checks.append(CheckResult("codebook:load", False, True, str(e)[:120]))

    # ------------------------------------------------------------------
    # Stealth checkpoints + disk footprint
    # ------------------------------------------------------------------
    project_root = Path(__file__).parent.parent.parent
    models = project_root / "storage" / "models"
    gan_ckpt = (models / "gan_generator.pt").exists()
    rl_ckpt = (models / "rl_agent.pt").exists()
    report.checks.append(
        CheckResult(
            "checkpoints",
            gan_ckpt or rl_ckpt,
            hard=False,
            detail=f"gan={'yes' if gan_ckpt else 'no'} rl={'yes' if rl_ckpt else 'no'} (static mode always available)",
        )
    )

    storage = project_root / "storage"
    size_mb = (
        sum(f.stat().st_size for f in storage.rglob("*") if f.is_file()) / (1024 * 1024)
        if storage.exists()
        else 0.0
    )
    report.checks.append(
        CheckResult(
            "disk:storage_footprint",
            size_mb < 1024,
            hard=False,
            detail=f"{size_mb:,.0f} MB (target < 1024)",
        )
    )

    report.phase0 = phase0
    return report

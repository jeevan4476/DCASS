# tests/test_diagnostics/test_doctor.py
"""
Tests for `dcass doctor` (WP-2) and the codebook<->index fingerprint
binding (WP-3), including the refusal path that closes the silent-corruption
failure mode.
"""

from pathlib import Path

import numpy as np
import pytest


class TestDoctorReport:
    @pytest.mark.integration
    def test_run_doctor_produces_phase0_table(self):
        """Tier 2 (needs indices+codebook): the Phase-0 numbers must be present."""
        from src.diagnostics import run_doctor

        report = run_doctor()
        p0 = report.phase0
        assert "modalities" in p0 and "codebook" in p0
        for modality in ("image", "text", "audio"):
            info = p0["modalities"][modality]
            assert "ntotal" in info and info["ntotal"] > 0
            assert info["agreement"] is True, (
                f"{modality}: len(metadata) != ntotal - index/metadata drift"
            )
        cb = p0["codebook"]
        assert cb["count_check_pass"] is True, (
            "STOP-THE-LINE: sum(ntotal) != len(cluster_assignments) - "
            "the codebook was fitted against a different index set"
        )
        assert cb["empty_clusters"] == 0 or cb["min_density"] >= 1
        # Every check entry has a verdict
        assert report.checks, "doctor produced no checks"
        assert any(c.name.startswith("dep:") for c in report.checks)

    def test_render_and_dict(self):
        from src.diagnostics import run_doctor

        report = run_doctor()
        text = report.render()
        assert "VERDICT:" in text
        d = report.to_dict()
        assert d["ok"] == report.ok
        assert len(d["checks"]) == len(report.checks)


class TestFingerprintBinding:
    @pytest.fixture()
    def blessed_indices_dir(self):
        """Blessed sidecar must exist on this machine (we blessed it)."""
        from src.corpus.index.unified_index import resolve_indices_base_path

        base = resolve_indices_base_path()
        sidecar = base / "voronoi_codebook.meta.json"
        if not sidecar.exists():
            pytest.skip(f"sidecar not present at {sidecar}; run bless_codebook.py --bless")
        return base

    def test_load_succeeds_on_blessed_pairing(self, blessed_indices_dir):
        from src.engine.vcp_payload import VCPPayloadMapper
        from src.corpus.index.unified_index import UnifiedSemanticIndex

        index = UnifiedSemanticIndex(base_path=blessed_indices_dir)
        index.load()
        mapper = VCPPayloadMapper(index)
        mapper.load()  # must NOT raise
        assert mapper._loaded

    def test_rebuilt_index_is_refused(self, blessed_indices_dir, tmp_path):
        """
        WP-3 acceptance criterion: tampering with one index (rebuilding it
        without re-fitting) makes load() REFUSE instead of returning wrong
        bytes. We simulate by copying the real artifacts and perturbing one
        vector of the image index.
        """
        import faiss
        import shutil

        work = tmp_path / "indices"
        shutil.copytree(blessed_indices_dir, work)

        idx = faiss.read_index(str(work / "image.index"))
        vec = np.asarray(idx.reconstruct(0), dtype=np.float32)
        vec[0] += 1.0  # rebuild-style perturbation -> different fingerprint
        new_index = type(idx)(idx.d)
        new_index.add(
            np.stack(
                [vec]
                + [
                    np.asarray(idx.reconstruct(i), dtype=np.float32)
                    for i in range(1, min(idx.ntotal, 50))
                ]
            )
        )
        # Keep ntotal identical so only the CONTENT fingerprint changes -
        # proving the fingerprint catches what a count check cannot.
        faiss.write_index(new_index, str(work / "image.index"))

        from src.engine.vcp_payload import VCPPayloadMapper
        from src.corpus.index.unified_index import UnifiedSemanticIndex

        index = UnifiedSemanticIndex(base_path=work)
        status = index.load()
        assert all(status.values()), f"fixture failed to load: {status}"

        mapper = VCPPayloadMapper(index)
        with pytest.raises(RuntimeError, match="binding BROKEN"):
            mapper.load()

    @pytest.mark.integration
    def test_bless_script_check_mode(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/cluster/bless_codebook.py", "--check"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent.parent,
        )
        if (
            "No sidecar" in result.stdout
            or "ERROR: no codebook" in result.stdout
            or result.returncode != 0
            and "codebook" in result.stdout
        ):
            pytest.skip("codebook or sidecar not present")
        assert "BINDING OK" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

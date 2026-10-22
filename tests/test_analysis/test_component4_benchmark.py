"""Tests for the component 4 codec benchmark scripts and their recorded artifacts.

Properties pinned here are the ones the component 4 audit had to reconstruct
from the scripts rather than the documentation: permutation determinism and
key decorrelation in the HMAC-DRBG engine, the RS(63,55) algebraic thresholds
of 4 byte errors or 8 known erasures, the Wilson interval construction behind
the recorded recovery bounds, the byte-identity of the committed LaTeX table
with what the script regenerates from the committed JSON, and the recorded
figures that the benchmark documents quote.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "analysis"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod4a = _load_module("benchmark_4a_capacity", "benchmark_4a_capacity.py")
mod4b = _load_module("benchmark_4b_robustness", "benchmark_4b_robustness.py")
mod4c = _load_module(
    "benchmark_4c_ecc_failure_analysis", "benchmark_4c_ecc_failure_analysis.py"
)


class TestPermutationEngine:
    KEY = b"dcass_test_session_key_32_bytes_ok"
    SALT = "dssc:test:salt"

    def test_permutation_is_deterministic_and_a_bijection(self):
        p1 = mod4b.derive_session_permutation(2000, self.KEY, context_salt=self.SALT)
        p2 = mod4b.derive_session_permutation(2000, self.KEY, context_salt=self.SALT)
        assert np.array_equal(p1, p2)
        assert sorted(int(i) for i in p1) == list(range(2000))

    def test_single_bit_key_flip_decorrelates(self):
        p1 = mod4b.derive_session_permutation(5000, self.KEY, context_salt=self.SALT)
        flipped = bytearray(self.KEY)
        flipped[15] ^= 0x01
        p2 = mod4b.derive_session_permutation(
            5000, bytes(flipped), context_salt=self.SALT
        )
        assert not np.array_equal(p1, p2)
        rho = float(np.corrcoef(p1.astype(float), p2.astype(float))[0, 1])
        assert abs(rho) < 0.1

    def test_run_permutation_benchmark_reports_determinism(self):
        results = mod4b.run_permutation_benchmark(candidate_sizes=[1000], num_trials=3)
        assert len(results) == 1
        row = results[0]
        assert row.determinism_passed is True
        assert abs(row.key_sensitivity_rho) < 0.1
        assert row.time_per_permutation_ms > 0.0


class TestTableRegeneration:
    def test_regenerated_4a_table_matches_committed_artifact(self, tmp_path):
        results = mod4a.load_results_from_json(
            BENCHMARK_DIR / "component_4a_capacity_results.json"
        )
        out = tmp_path / "table4a_capacity.tex"
        mod4a._generate_table(results, out)
        text = out.read_text()
        committed = TABLES_DIR / "table4a_capacity.tex"
        assert text == committed.read_text()
        assert "exact\\_vcp" in text
        assert "exact_vcp &" not in text

    def test_regenerated_4b_table_matches_committed_artifact(self, tmp_path):
        recorded = mod4b.load_results_from_json(
            BENCHMARK_DIR / "component_4b_robustness_results.json"
        )
        out = tmp_path / "table4b_codec_robustness.tex"
        mod4b.generate_latex_table(
            recorded.permutation_benchmarks,
            recorded.ecc_recovery_benchmarks,
            recorded.key_separation_benchmarks,
            out,
        )
        text = out.read_text()
        committed = TABLES_DIR / "table4b_codec_robustness.tex"
        assert text == committed.read_text()
        assert "Carrier Substitution" in text
        assert "Channel Loss" not in text

    def test_regenerated_4c_table_matches_committed_artifact(self, tmp_path):
        data = json.loads(
            (BENCHMARK_DIR / "component_4c_ecc_failure_results.json").read_text()
        )
        sub_results = [mod4c.E2EResult(**row) for row in data["substitution_results"]]
        drop_results = [mod4c.E2EResult(**row) for row in data["dropout_results"]]
        points = [mod4c.AlgebraicPoint(**row) for row in data["algebraic_points"]]
        out = tmp_path / "table4c_ecc_failure_modes.tex"
        mod4c.generate_latex_table(
            sub_results, drop_results, points, data["algebraic_meta"], out
        )
        text = out.read_text()
        committed = TABLES_DIR / "table4c_ecc_failure_modes.tex"
        assert text == committed.read_text()
        assert "Exact & Accepted" in text


class TestAlgebraicThresholds:
    def test_error_and_erasure_thresholds_and_meta(self):
        points, meta = mod4c.run_algebraic_benchmark(np.random.default_rng(0), reps=3)
        assert meta == {
            "codeword_bytes": 63,
            "framed_bytes": 55,
            "parity_bytes": 8,
            "t_errors": 4,
            "t_erasures": 8,
        }
        by_injected = {p.injected: p for p in points}
        assert sorted(by_injected) == list(range(13))
        for s in range(0, 5):
            assert by_injected[s].error_success_pct == 100.0
        for s in range(5, 13):
            assert by_injected[s].error_success_pct == 0.0
        for s in range(0, 9):
            assert by_injected[s].erasure_success_pct == 100.0
        for s in range(9, 13):
            assert by_injected[s].erasure_success_pct == 0.0


class TestStatistics:
    def test_wilson_interval_matches_recorded_bounds(self):
        assert mod4c.wilson_interval(80, 80) == pytest.approx((95.4, 100.0), abs=0.1)
        assert mod4c.wilson_interval(0, 80) == pytest.approx((0.0, 4.6), abs=0.1)
        assert mod4c.wilson_interval(49, 80) == pytest.approx((50.3, 71.2), abs=0.1)

    def test_binomial_cdf_endpoints_and_monotonicity(self):
        assert mod4c.binomial_cdf(0, 10, 0.3) == pytest.approx(0.7**10)
        assert mod4c.binomial_cdf(10, 10, 0.3) == pytest.approx(1.0)
        cdfs = [mod4c.binomial_cdf(k, 10, 0.3) for k in range(11)]
        assert cdfs == sorted(cdfs)


class TestRecordedArtifacts:
    """The figures quoted in the benchmark documentation must live in the JSON."""

    @staticmethod
    def _records(filename: str):
        return json.loads((BENCHMARK_DIR / filename).read_text())

    def test_capacity_record_pins_documented_figures(self):
        data = self._records("component_4a_capacity_results.json")
        assert data["schema"] == "component4a-capacity-v2"
        assert data["provenance"]["retrospective"] is False
        assert data["provenance"]["git_commit"]
        assert data["config"]["parity_bytes"] == 8
        assert data["config"]["session_key_bytes"] == 32
        by_mode = {}
        for row in data["results"]:
            by_mode.setdefault(row["mode"], {})[row["payload_bytes"]] = row
        sizes = (14, 50, 150, 300, 500)

        vcp = by_mode["exact_vcp (multi-modal)"]
        assert [vcp[b]["carrier_count"] for b in sizes] == [27, 63, 163, 321, 529]
        assert [vcp[b]["bits_per_carrier"] for b in sizes] == [8.0] * 5
        assert set(vcp[50]["modality_counts"]) == {"image", "text", "audio"}

        multi = by_mode["DSSC (multi-modal)"]
        bpc = [multi[b]["bits_per_carrier"] for b in sizes]
        assert bpc == pytest.approx([13.50, 13.62, 13.58, 13.66, 13.65], abs=0.005)
        assert float(np.mean(bpc)) == pytest.approx(13.602, abs=0.001)
        assert [multi[b]["carrier_reduction_pct"] for b in sizes] == pytest.approx(
            [40.74, 41.27, 41.10, 41.43, 41.40], abs=0.01
        )

        text = by_mode["DSSC (text-only)"]
        assert [text[b]["bits_per_carrier"] for b in sizes] == pytest.approx(
            [15.43, 15.75, 15.90, 15.95, 15.97], abs=0.005
        )
        assert [text[b]["carrier_reduction_pct"] for b in sizes] == pytest.approx(
            [48.15, 49.21, 49.69, 49.84, 49.91], abs=0.01
        )
        assert all(set(text[b]["modality_counts"]) == {"text"} for b in sizes)

    def test_robustness_record_pins_documented_figures(self):
        data = self._records("component_4b_robustness_results.json")
        assert data["schema"] == "component4b-robustness-v2"
        assert data["provenance"]["retrospective"] is False
        assert data["provenance"]["git_commit"]
        assert data["config"]["parity_bytes"] == 8

        perm = {row["candidate_size"]: row for row in data["permutation_benchmarks"]}
        assert perm[1000]["time_per_permutation_ms"] > 0
        assert perm[35000]["time_per_permutation_ms"] > perm[1000]["time_per_permutation_ms"]
        assert all(row["determinism_passed"] for row in perm.values())
        assert all(row["time_per_permutation_p50_ms"] > 0 for row in perm.values())

        ecc = {
            row["carrier_error_rate"]: row for row in data["ecc_recovery_benchmarks"]
        }
        assert ecc[0.0]["recovery_rate_pct"] == 100.0
        assert ecc[0.05]["recovery_rate_pct"] == 70.0
        assert ecc[0.05]["mean_fixed_errors"] == pytest.approx(2.7)
        assert ecc[0.10]["recovery_rate_pct"] == 0.0
        assert all(row["parity_bytes"] == 8 for row in ecc.values())
        assert all(row["num_false_acceptances"] == 0 for row in ecc.values())

        keys = {row["test_type"]: row for row in data["key_separation_benchmarks"]}
        assert keys["random_wrong_keys"]["num_false_acceptances"] == 0
        assert keys["random_wrong_keys"]["num_exact_message_matches"] == 0
        assert keys["random_wrong_keys"]["num_trials"] == 100
        assert keys["random_wrong_keys"]["false_acceptance_upper_95_pct"] == pytest.approx(
            3.70, abs=0.01
        )
        assert keys["single_bit_flipped_keys"]["num_false_acceptances"] == 0
        assert keys["single_bit_flipped_keys"]["num_exact_message_matches"] == 0
        assert keys["single_bit_flipped_keys"]["num_trials"] == 256
        assert keys["single_bit_flipped_keys"]["false_acceptance_upper_95_pct"] == pytest.approx(
            1.48, abs=0.01
        )

    def test_failure_anatomy_record_pins_documented_figures(self):
        data = self._records("component_4c_ecc_failure_results.json")
        assert data["schema"] == "component4c-ecc-failure-v2"
        assert data["provenance"]["retrospective"] is False
        assert data["provenance"]["git_commit"]
        assert data["config"]["trials_per_point"] == 80
        assert data["config"]["parity_bytes"] == 8
        assert data["config"]["framing_policy"].startswith("Framed payloads")
        assert data["decoder_hardening"]["applied"] is True
        assert data["decoder_hardening"]["pre_hardening_dropout_false_acceptances"] == 41
        assert data["decoder_hardening"]["pre_hardening_dropout_trials_total"] == 480
        assert (
            data["decoder_hardening"]["pre_hardening_dropout_trials_with_false_accepts"]
            == 320
        )
        assert data["algebraic_meta"] == {
            "codeword_bytes": 63,
            "framed_bytes": 55,
            "parity_bytes": 8,
            "t_errors": 4,
            "t_erasures": 8,
        }

        sub = {row["carrier_loss_rate"]: row for row in data["substitution_results"]}
        assert len(sub) == 8
        point = sub[0.05]
        assert point["recovery_rate_pct"] == pytest.approx(61.25)
        assert point["recovery_ci_low_pct"] == pytest.approx(50.29, abs=0.01)
        assert point["recovery_ci_high_pct"] == pytest.approx(71.18, abs=0.01)
        # Recovery matched the damage budget exactly in all 640 substitution trials.
        for row in sub.values():
            assert row["trials_damaged_le_t_failed"] == 0
            assert row["trials_damaged_le_t_false_acceptances"] == 0
            assert row["trials_damaged_gt_t_recovered"] == 0
            assert row["trials_damaged_gt_t_accepted"] == 0
            assert row["num_false_acceptances"] == 0
            assert row["num_decoder_successes"] == row["num_recovered"]
            if row["carrier_loss_rate"] > 0:
                assert row["num_symbol_changes"] > 0
                assert 0.0 < row["symbol_change_pct"] <= 100.0

        drop = {row["carrier_loss_rate"]: row for row in data["dropout_results"]}
        assert drop[0.02]["recovery_rate_pct"] == 0.0
        assert drop[0.02]["mean_damaged_bytes"] == pytest.approx(32.675)
        assert drop[0.02]["mean_received_carriers"] < drop[0.02]["mean_carriers"]
        assert drop[0.02]["mean_bitstream_length_delta_bytes"] > 0
        assert drop[0.02]["trials_damaged_le_t_failed"] > 0
        # Post-hardening: any CRC-passing decode is an exact recovery, so the
        # legacy raw fallback can no longer turn a shifted stream into success.
        assert all(row["num_false_acceptances"] == 0 for row in drop.values())
        assert all(
            row["num_decoder_successes"] == row["num_recovered"] for row in drop.values()
        )

        total_trials = sum(r["num_trials"] for r in sub.values()) + sum(
            r["num_trials"] for r in drop.values()
        )
        assert total_trials == 1120

        # Erasure-bound replay quoted in the failure-anatomy section.
        bound = {
            rate: 100.0
            * sum(1 for d in row["damaged_samples"] if d <= 8)
            / len(row["damaged_samples"])
            for rate, row in sub.items()
        }
        assert bound[0.06] == 100.0
        assert bound[0.08] == pytest.approx(98.75, abs=0.01)
        assert bound[0.10] == pytest.approx(51.25, abs=0.01)

        points = {p["injected"]: p for p in data["algebraic_points"]}
        assert points[4]["error_success_pct"] == 100.0
        assert points[5]["error_success_pct"] == 0.0
        assert points[8]["erasure_success_pct"] == 100.0
        assert points[9]["erasure_success_pct"] == 0.0

    def test_archived_pre_hardening_record_documents_false_accepts(self):
        data = json.loads(
            (
                BENCHMARK_DIR
                / "archive"
                / "component_4c_ecc_failure_results_pre_hardening.json"
            ).read_text()
        )
        assert sum(row["num_false_acceptances"] for row in data["dropout_results"]) == 41
        assert sum(row["num_trials"] for row in data["dropout_results"]) == 480
        assert sum(
            row["num_trials"]
            for row in data["dropout_results"]
            if row["carrier_loss_rate"] >= 0.02
        ) == 320

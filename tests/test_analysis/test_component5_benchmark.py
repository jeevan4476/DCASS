"""Tests for the component 5 codec concurrency benchmark and its artifacts.

The component 5 audit found that the recorded benchmark calls the engine
directly, not FastAPI, and that the generated table was missing the p90 and
p99 columns quoted by the report. These tests pin the direct-engine scope, the
observed-swap definition, the artifact schema, and the byte identity of the
report table block with the committed table and with a fresh regeneration from
the committed JSON.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "scripts" / "analysis"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
REPORT_PATH = PROJECT_ROOT / "docs" / "benchmarking" / "BENCHMARKING_REPORT.md"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod5 = _load_module("benchmark_5_system_concurrency", "benchmark_5_system_concurrency.py")


def _payload_result() -> "mod5.PayloadPerformanceResult":
    return mod5.PayloadPerformanceResult(
        payload_bytes=32,
        carrier_count=27,
        encode_latency_ms=94.6,
        encode_latency_std=2.0,
        decode_latency_ms=216.0,
        decode_latency_std=9.0,
        total_latency_ms=310.6,
        encode_throughput_kbps=0.33,
        decode_throughput_kbps=0.15,
        carrier_speed_per_sec=86.9,
        stage_breakdown={"faiss_search_ms": 96.21},
    )


def _concurrency_result() -> "mod5.ConcurrencyResult":
    return mod5.ConcurrencyResult(
        num_threads=50,
        total_requests=100,
        duration_sec=38.9,
        requests_per_sec=2.57,
        p50_latency_ms=17668.1,
        p90_latency_ms=19915.9,
        p95_latency_ms=20573.5,
        p99_latency_ms=22707.7,
        success_rate_pct=100.0,
        cross_talk_rate_pct=0.0,
    )


def test_generated_table_has_two_parts_and_percentile_columns(tmp_path):
    output = tmp_path / "table5.tex"
    mod5.generate_latex_table([_payload_result()], [_concurrency_result()], output)
    text = output.read_text()

    assert text.count(r"\begin{tabular}") == 2
    assert text.count(r"\end{tabular}") == 2
    # The generated table is a full-width float with a scaled tabular.
    assert text.count(r"\end{table*}") == 1
    assert text.count(r"\resizebox{\textwidth}{!}{%") == 2
    assert r"\begin{tabular}{lcccccc}" in text
    assert r"\begin{tabular}{lcccccccc}" in text
    assert r"$p_{90}$" in text
    assert r"$p_{99}$" in text
    assert "sessions/s" in text
    assert "19,915.9" in text or "19915.9" in text
    assert "22707.7" in text
    assert "Observed Swap" in text


def test_committed_table_regenerates_from_committed_json(tmp_path):
    output = tmp_path / "table5.tex"
    mod5.regenerate_table_from_json(
        json_path=BENCHMARK_DIR / "component_5_results.json",
        table_path=output,
    )
    committed = (TABLES_DIR / "table5_system_performance.tex").read_text().strip()
    assert output.read_text().strip() == committed


def test_artifacts_regenerate_from_committed_json(tmp_path):
    table, breakdown, concurrency = mod5.regenerate_artifacts_from_json(
        json_path=BENCHMARK_DIR / "component_5_results.json",
        table_path=tmp_path / "table5.tex",
        plot_breakdown_path=tmp_path / "fig5_latency_breakdown.png",
        plot_concurrency_path=tmp_path / "fig5_concurrency_scaling.png",
    )
    committed = (TABLES_DIR / "table5_system_performance.tex").read_text().strip()
    assert table.read_text().strip() == committed
    assert breakdown.stat().st_size > 1000
    assert concurrency.stat().st_size > 1000


def test_report_table5_block_matches_generated_table():
    table = (TABLES_DIR / "table5_system_performance.tex").read_text().strip()
    report = REPORT_PATH.read_text()
    marker = "% Component 5: Computational Performance & System Scalability"
    assert report.count(marker) == 1
    start = report.index(marker)
    # The generated table is a full-width float, so it closes with
    # \end{table*}. Accept either form.
    ends = [
        report.index(suffix, start)
        for suffix in (r"\end{table}", r"\end{table*}")
        if suffix in report[start:]
    ]
    end = min(ends) + len(r"\end{table*}")
    assert report[start:end].strip() == table


def test_artifact_has_scope_provenance_and_percentiles():
    data = json.loads((BENCHMARK_DIR / "component_5_results.json").read_text())

    assert data["schema"] == "component5-results-v2"
    # The artifact is a fresh run, so its provenance is recorded rather than
    # reconstructed. A retrospective block would be flagged as such.
    assert data["provenance"]["retrospective"] is False
    assert data["provenance"]["git_commit"]
    assert data["provenance"]["timestamp"]
    assert data["provenance"]["device"]
    assert "direct semanticengine" in data["metric_definitions"]["scope"].lower()
    assert "queue wait" in data["metric_definitions"]["latency"]
    assert "engine.encode" in data["metric_definitions"]["throughput"]
    assert "SECRET_PAYLOAD" in data["metric_definitions"]["cross_talk"]
    assert data["metric_definitions"] == mod5.metric_definitions()
    assert data["schema"] == mod5.SCHEMA
    assert len(data["concurrency_stress"]) == 3
    for row in data["concurrency_stress"]:
        assert {"p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"} <= row.keys()


class _FakeEngine:
    def __init__(self, decoded_message: str):
        self.decoded_message = decoded_message

    def encode(self, *args, **kwargs):
        return SimpleNamespace(media_ids=["media-1", "media-2"])

    def decode(self, *args, **kwargs):
        return SimpleNamespace(reconstructed_message=self.decoded_message)


def test_session_success_and_swap_classification():
    own = "SECRET_PAYLOAD_FOR_CLIENT_0007_VERIFY_ISOLATION!"
    other = "SECRET_PAYLOAD_FOR_CLIENT_0008_VERIFY_ISOLATION!"

    success = mod5.run_single_client_session(_FakeEngine(own), 7)
    assert success["success"] is True
    assert success["cross_talk"] is False

    swapped = mod5.run_single_client_session(_FakeEngine(other), 7)
    assert swapped["success"] is False
    assert swapped["cross_talk"] is True

    corrupted = mod5.run_single_client_session(_FakeEngine("garbled output"), 7)
    assert corrupted["success"] is False
    assert corrupted["cross_talk"] is False

# tests/test_stealth/test_script_defaults.py
"""FIX-C regression surface: the CLI data defaults and the baseline skip.

The three operator-facing scripts (train_gan, evaluate_stealth,
eval_warden) used to default their --data / --human-data arguments at
`data/human_traffic.json`, a path that does not exist in the repository —
every recorded human-reference comparison silently vanished unless the
operator knew to pass the path by hand, and nothing said so.

These tests pin:
  * each script's module-level DEFAULT_HUMAN_DATA points at the recorded
    corpus (`storage/data/traffic/human_traffic.json`), which has the
    schema the loaders consume: a list of records with delays, channels
    and time_of_day;
  * evaluate_stealth announces a skipped baseline instead of silently
    dropping it, and still produces its own report.

Script modules are loaded by file path (importlib
spec_from_file_location), the same convention
tests/test_analysis/test_component1_benchmark.py uses for script-shaped
modules. They deliberately live OUTSIDE tests/test_integration.py:
anything under a `test_integration` nodeid is marked integration by
tests/conftest.py and skipped by the sanctioned fast suite
(`pytest --no-integration`).
"""

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SCRIPT_MODULES = (
    "scripts/stealth/train_gan.py",
    "scripts/testing/evaluate_stealth.py",
    "scripts/stealth/eval_warden.py",
)


def _load_script_module(relative: str):
    """Load a script module by file path (never as part of a package)."""
    spec = importlib.util.spec_from_file_location(
        f"_fixc_script_{relative.rsplit('/', 1)[-1][:-3]}", _REPO_ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_human_data_paths_exist():
    """Every CLI default for --data / --human-data points at the recorded
    corpus path (storage/data/traffic/human_traffic.json). If present on disk,
    verifies that the corpus has the schema the loaders consume: a list of
    records with delays, channels, and time_of_day."""
    import pytest

    script_modules = [_load_script_module(rel) for rel in _SCRIPT_MODULES]
    for module in script_modules:
        assert str(module.DEFAULT_HUMAN_DATA).endswith("storage/data/traffic/human_traffic.json"), (
            f"{module.__name__}'s DEFAULT_HUMAN_DATA must point to storage/data/traffic/human_traffic.json: "
            f"{module.DEFAULT_HUMAN_DATA}"
        )

    data_path = Path(script_modules[0].DEFAULT_HUMAN_DATA)
    if not data_path.exists():
        pytest.skip(f"human traffic corpus not present on disk: {data_path}")

    payload = json.loads(data_path.read_text())
    assert isinstance(payload, list) and payload, "corpus must be a non-empty list"
    assert set(payload[0].keys()) == {"delays", "channels", "time_of_day"}


def test_evaluate_stealth_announces_missing_baseline(capsys, tmp_path, monkeypatch):
    """A missing baseline file used to be swallowed (load_human_traffic
    returned None and the `if human:` block silently vanished). The tool
    must say what it skipped and continue."""
    missing = tmp_path / "no_such_baseline.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stealth.py",
            "--mode", "static",
            "--num-sequences", "1",
            "--device", "cpu",
            "--human-data", str(missing),
        ],
    )
    module = _load_script_module("scripts/testing/evaluate_stealth.py")
    module.main()

    captured = capsys.readouterr()
    assert f"HUMAN BASELINE SKIPPED (file missing: {missing})" in captured.out
    # The tool continued with its own report after skipping the baseline.
    assert "Mode        : static" in captured.out


def test_evaluate_stealth_announces_empty_baseline(capsys, tmp_path, monkeypatch):
    """An existing-but-empty baseline file used to skip silently too: the
    loader returned [] and `if human:` never ran, with no word of it. The
    same loud treatment as the missing file applies."""
    empty = tmp_path / "empty_baseline.json"
    empty.write_text("[]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stealth.py",
            "--mode", "static",
            "--num-sequences", "1",
            "--device", "cpu",
            "--human-data", str(empty),
        ],
    )
    module = _load_script_module("scripts/testing/evaluate_stealth.py")
    module.main()

    captured = capsys.readouterr()
    assert f"HUMAN BASELINE SKIPPED (file empty: {empty})" in captured.out
    assert "Mode        : static" in captured.out


def test_dataset_announces_empty_corpus(capsys, tmp_path):
    """HumanTrafficDataset on an existing-but-empty corpus silently built a
    zero-record dataset (training then had nothing to iterate). The
    missing-file path already warns and falls back to synthetic data; an
    empty file is the same operational situation and gets the same loud
    treatment."""
    from src.stealth.gan.trainer import HumanTrafficDataset

    empty = tmp_path / "empty_corpus.json"
    empty.write_text("[]")
    dataset = HumanTrafficDataset(empty, max_sequence_length=100)

    captured = capsys.readouterr()
    assert "empty" in captured.out, "the empty-corpus fallback must announce itself"
    assert dataset.data, "empty corpus must fall back to synthetic data like a missing one"

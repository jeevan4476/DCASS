"""Tests for the terminal panel showcase readers."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "demo" / "panel_showcase.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("panel_showcase", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component1_rows_read_schema_v2_nested_fields():
    module = _load_module()
    summary = {
        "detectors": [
            {
                "name": "sr_net",
                "control": {"accuracy": 0.9367, "auc_mean": 0.9844},
                "dcass": {"auc_mean": 0.4973},
            }
        ]
    }
    assert module._component1_rows(summary) == [
        ["sr_net", "93.67%", "0.9844", "0.4973"]
    ]


def test_component1_rows_fall_back_to_legacy_flat_fields():
    module = _load_module()
    summary = {
        "detectors": [
            {
                "name": "zhu_net",
                "traditional_accuracy": 0.9505,
                "traditional_auc": 0.986,
                "dcass_auc": 0.4993,
            }
        ]
    }
    assert module._component1_rows(summary) == [
        ["zhu_net", "95.05%", "0.9860", "0.4993"]
    ]


def test_component1_rows_prefer_nested_fields_when_both_exist():
    module = _load_module()
    summary = {
        "detectors": [
            {
                "name": "ye_net",
                "traditional_accuracy": 0.10,
                "traditional_auc": 0.20,
                "dcass_auc": 0.30,
                "control": {"accuracy": 0.9397, "auc_mean": 0.9853},
                "dcass": {"auc_mean": 0.4993},
            }
        ]
    }
    assert module._component1_rows(summary) == [
        ["ye_net", "93.97%", "0.9853", "0.4993"]
    ]

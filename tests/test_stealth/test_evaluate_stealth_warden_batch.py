# tests/test_stealth/test_evaluate_stealth_warden_batch.py
"""FIX-D regression surface: evaluate_warden must score with the mask.

`evaluate_stealth.evaluate_warden` used to pad ragged batches with zeros
and call the detector with mask=None: neither the transformer (whose
attention then saw the padded steps as real context) nor the pooling
layer (whose mean/max then included them) knew those steps were
fabricated. Every other consumer of the detector — the benchmark and the
training loop — passes a validity mask, so a padded batch scored
differently from the same sequences scored alone, and the CLI's headline
bot probability depended on how ragged the happened-to-be batch was.

These tests are in a standalone file instead of test_integration.py
because tests/conftest.py marks any `test_integration` nodeid as
integration and the sanctioned fast suite would skip them.
"""

import importlib.util
import sys
from pathlib import Path

import torch

from src.analysis.adversarial.warden import DeepPacketInspectionWarden

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_script_module(relative: str):
    spec = importlib.util.spec_from_file_location(
        "_fixd_script_" + relative.rsplit("/", 1)[-1][:-3], _REPO_ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_warden_scores_ragged_batch_like_single_rows():
    """A ragged two-row batch (lengths 8 and 3) must produce the same
    per-row scores as scoring each row alone: the mask has to make the
    padding invisible."""
    module = _load_script_module("scripts/testing/evaluate_stealth.py")
    torch.manual_seed(42)
    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.eval()

    delays_rows = [
        [3.1, 4.2, 2.7, 5.9, 3.3, 8.1, 2.2, 6.4],
        [7.9, 6.8, 5.5],
    ]
    channels_rows = [
        [0, 1, 2, 0, 1, 2, 0, 1],
        [2, 1, 0],
    ]

    batch_scores = module.evaluate_warden(warden, delays_rows, channels_rows)
    single_scores = [
        module.evaluate_warden(warden, [row_d], [row_c])[0]
        for row_d, row_c in zip(delays_rows, channels_rows)
    ]

    assert len(batch_scores) == 2
    for k, (batched, single) in enumerate(zip(batch_scores, single_scores)):
        assert batched == single or abs(batched - single) < 1e-4, (
            f"row {k}: padded batch scored {batched!r}, scored alone {single!r} "
            "-- the mask is not making the padding invisible"
        )

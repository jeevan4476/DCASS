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


def test_evaluate_warden_truncates_rows_to_the_convention_length():
    """A row longer than DETECTOR_SEQ_LEN used to stretch the whole batch
    past the detector's fixed scoring convention (max_len took the max of
    DETECTOR_SEQ_LEN and the row lengths), so a >64-row scored under a
    convention the detector never trained on. evaluate_warden's own
    populations — the CLI's default human baseline (`human_traffic.json`,
    max session 50) and scheduler rows (20) — never exceed 64. The pooled
    detector training corpus does (`real_human_traffic.json` max session
    80), but training itself truncates to SEQ_LEN=64
    (`train_traffic_detector.py` `take = min(n, SEQ_LEN)`), so truncation
    here applies exactly the trained convention. The guard pins that any
    long row is scored identically to its own first DETECTOR_SEQ_LEN
    steps."""
    module = _load_script_module("scripts/testing/evaluate_stealth.py")
    torch.manual_seed(42)
    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.eval()

    from src.analysis.adversarial.warden import DETECTOR_SEQ_LEN

    rng = torch.Generator().manual_seed(7)
    long_row = (torch.rand(DETECTOR_SEQ_LEN + 6, generator=rng) * 20 + 0.5).tolist()
    long_channels = [int(c) for c in (torch.randint(0, 3, (DETECTOR_SEQ_LEN + 6,), generator=rng)).tolist()]

    score_long = module.evaluate_warden(warden, [long_row], [long_channels])[0]
    score_truncated = module.evaluate_warden(
        warden, [long_row[:DETECTOR_SEQ_LEN]], [long_channels[:DETECTOR_SEQ_LEN]]
    )[0]

    assert score_long == score_truncated, (
        f"row longer than the convention scored {score_long!r} but its first "
        f"{DETECTOR_SEQ_LEN} steps scored {score_truncated!r} -- the row was "
        "not truncated to the scoring convention"
    )


def test_evaluate_warden_handles_empty_batch():
    """An empty batch must return an empty list without crashing."""
    module = _load_script_module("scripts/testing/evaluate_stealth.py")
    warden = DeepPacketInspectionWarden(num_channels=3)
    warden.eval()

    res = module.evaluate_warden(warden, [], [])
    assert res == []

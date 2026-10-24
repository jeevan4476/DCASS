# tests/test_stealth/test_tiny_schedule.py
"""Tiny-schedule regression surface: media_ids of length 1 and 2.

Every existing scheduler test exercises 5–6 item sequences; the B3 review
flagged that the literal "tiny" end of the contract — a single item, then a
pair — had no test. These pin that the scheduler handles them exactly as
the contract states: n delays for n ids, n channels, a valid mode_used, no
exception. Static mode needs no checkpoints; auto mode exercises the
rl→gan→static fallback cascade on a checkpoint-less environment.

Plain file (NOT test_integration.py): tests/conftest.py marks any
`test_integration` nodeid as integration and the sanctioned fast suite
would skip it.
"""

import pytest

from src.stealth.stealth_scheduler import StealthScheduler

_VALID_MODES = {"static", "gan", "rl", "auto"}


@pytest.fixture(scope="module")
def scheduler():
    return StealthScheduler(num_channels=3)


@pytest.mark.parametrize("mode", ["static", "auto"])
@pytest.mark.parametrize("n_ids", [1, 2])
def test_tiny_schedule_returns_matching_shapes(scheduler, mode, n_ids):
    """A schedule for n media ids must carry exactly n delays and n
    channels, in a valid mode, for the smallest n values there are."""
    media_ids = [f"m{i}" for i in range(n_ids)]
    result = scheduler.schedule(media_ids, mode=mode)

    assert len(result["delays"]) == n_ids
    assert len(result["channels"]) == n_ids
    assert result["mode_used"] in _VALID_MODES
    assert all(d >= 0 for d in result["delays"])
    assert all(0 <= c < 3 for c in result["channels"])

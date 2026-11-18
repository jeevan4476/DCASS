import numpy as np
import pytest

from src.stealth.activity_timing import (
    ContiguousGapBlockModel,
    TwoTimescaleActivityModel,
    route_gaps_with_cooldowns,
)


def sample_session(contributor, delays, channels=None):
    return {
        "contributor_id": contributor,
        "delays": delays,
        "channels": channels or [i % 3 for i in range(len(delays))],
        "time_of_day": 12,
    }


def test_long_gaps_are_pauses_not_active_carrier_delays():
    model = TwoTimescaleActivityModel(
        [sample_session("u1", [1.0, 2.0, 45.0, 3.0, 4.0])],
        max_active_gap_s=30.0,
        block_size=2,
    )

    assert model.pause_count == 1
    assert model.active_gap_count == 4
    assert model.sample_pause(np.random.default_rng(1)) == 45.0
    burst = model.sample_active_burst(8, np.random.default_rng(2))
    assert len(burst["delays"]) == len(burst["channels"]) == 8
    assert burst["contributor_id"] == "u1"
    assert all(0 < delay <= 30.0 for delay in burst["delays"])


def test_empirical_model_requires_contributor_and_aligned_channels():
    with pytest.raises(ValueError, match="contributor_id"):
        TwoTimescaleActivityModel([{"delays": [1.0], "channels": [0]}])

    with pytest.raises(ValueError, match="same length"):
        TwoTimescaleActivityModel([sample_session("u1", [1.0, 2.0], [0])])


def test_empirical_burst_sampling_is_reproducible():
    sessions = [
        sample_session("u1", [1, 2, 3, 4, 5, 6]),
        sample_session("u2", [7, 8, 9, 10, 11, 12]),
    ]
    model = TwoTimescaleActivityModel(sessions, block_size=3)

    first = model.sample_active_burst(11, np.random.default_rng(9))
    second = model.sample_active_burst(11, np.random.default_rng(9))

    assert first == second


def test_contiguous_block_model_preserves_long_pauses_and_order():
    model = ContiguousGapBlockModel(
        [sample_session("u1", [1.0, 2.0, 90.0, 4.0, 5.0, 6.0])],
        block_size=3,
    )

    sampled = model.sample_delays(3, np.random.default_rng(5))

    assert sampled["delays"] in (
        [1.0, 2.0, 90.0],
        [2.0, 90.0, 4.0],
        [90.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    )
    assert max(sampled["delays"]) == 90.0


def test_contiguous_block_model_does_not_require_channel_semantics():
    model = ContiguousGapBlockModel(
        [{"contributor_id": "u1", "delays": [1.0, 40.0]}], block_size=2
    )

    sampled = model.sample_delays(2, np.random.default_rng(1))

    assert sampled == {"delays": [1.0, 40.0], "contributor_id": "u1"}


def test_route_gaps_with_cooldowns_preserves_pauses_and_is_legal():
    routed, channels = route_gaps_with_cooldowns(
        [1.0, 2.0, 90.0],
        np.random.default_rng(4),
        {0: 6.0, 1: 12.0, 2: 4.0},
    )

    assert len(routed) == len(channels) == 3
    assert routed[2] >= 90.0
    landing = np.cumsum(routed)
    last_by_channel = {}
    for time, channel in zip(landing, channels):
        if channel in last_by_channel:
            assert time - last_by_channel[channel] >= {0: 6.0, 1: 12.0, 2: 4.0}[channel]
        last_by_channel[channel] = time

"""Pin the API transmission admission semantics.

Component 5 measures the direct engine, not `/api/transmit`. The API admits
several concurrent transmissions, bounded by
`MAX_CONCURRENT_TRANSMISSIONS`: `_claim_transmission` succeeds while the
bound is not reached and fails once it is, and the endpoint maps that to
HTTP 429 rather than the 409 that the previous single-active design used.

The sessions themselves are independent. Each writes packets under its own
`<session_id>__<media_id>.json` name and its own `_manifest_<session_id>.json`,
so two sessions to the same recipient do not collide.
"""

import pytest

from src.api import server


@pytest.fixture(autouse=True)
def _clean_sessions():
    server._transmission_sessions.clear()
    yield
    server._transmission_sessions.clear()


def test_claim_admits_up_to_the_concurrency_bound():
    limit = server.MAX_CONCURRENT_TRANSMISSIONS
    assert limit >= 2, "the bound should allow concurrent sessions"

    claimed = []
    for i in range(limit):
        session_id = f"test-session-concurrent-{i}"
        assert server._claim_transmission(session_id) is True
        claimed.append(session_id)

    # The next session is over the bound.
    assert server._claim_transmission("test-session-over-bound") is False

    # Releasing one makes room again.
    server._cleanup_session(claimed[0])
    assert server._claim_transmission("test-session-after-release") is True


def test_two_sessions_are_admitted_concurrently():
    """The previous design admitted exactly one. Two must now be allowed."""
    assert server._claim_transmission("test-session-par-a") is True
    assert server._claim_transmission("test-session-par-b") is True
    assert server._active_transmission_count() == 2


def test_claim_is_idempotent_for_the_same_session():
    """Re-claiming a session that is already active must not double-count
    it against the bound."""
    session_id = "test-session-repeat"
    assert server._claim_transmission(session_id) is True
    assert server._claim_transmission(session_id) is True
    assert server._active_transmission_count() == 1


def test_active_count_tracks_cleanup():
    first = "test-session-count-a"
    second = "test-session-count-b"
    server._claim_transmission(first)
    server._claim_transmission(second)
    assert server._active_transmission_count() == 2
    server._cleanup_session(first)
    assert server._active_transmission_count() == 1

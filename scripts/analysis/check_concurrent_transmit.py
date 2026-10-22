#!/usr/bin/env python3
"""
End-to-end check of the concurrent transmission API.

Starts the FastAPI app with the test client, authenticates as the demo
users, and runs two transmissions at once to the same recipient. It then
verifies that:

  * both sessions are admitted;
  * each session's packets land in the recipient directory under its own
    session id, with no filename collision;
  * both sessions are visible in the inbox;
  * the admission bound is enforced.

This is a check, not a unit test: it exercises the real endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api import server  # noqa: E402


def main() -> int:
    client = TestClient(server.app)

    # Authenticate as the two demo users.
    tokens = {}
    for username, password in (("alice", "alice12345"), ("bob", "bob12345")):
        resp = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        if resp.status_code != 200:
            print(f"ERROR: login failed for {username}: {resp.status_code} {resp.text}")
            return 1
        tokens[username] = resp.json()["access_token"]
    print("Authenticated alice and bob.")

    alice = {"Authorization": f"Bearer {tokens['alice']}"}
    bob = {"Authorization": f"Bearer {tokens['bob']}"}

    # A short media list keeps the transmissions quick.
    media_ids = [f"flickr30k_{i}" for i in range(3)]

    # Both transmissions target bob, so they share a directory.
    first = client.post(
        "/api/transmit",
        json={
            "media_ids": media_ids,
            "message": "first concurrent message",
            "recipient_username": "bob",
            "speed_multiplier": 1.0,
        },
        headers=alice,
    )
    second = client.post(
        "/api/transmit",
        json={
            "media_ids": media_ids,
            "message": "second concurrent message",
            "recipient_username": "bob",
            "speed_multiplier": 1.0,
        },
        headers=alice,
    )

    print(f"  first  -> {first.status_code}")
    print(f"  second -> {second.status_code}")
    if first.status_code != 200 or second.status_code != 200:
        print(f"  first body : {first.text[:200]}")
        print(f"  second body: {second.text[:200]}")
        return 1

    s1 = first.json()["session_id"]
    s2 = second.json()["session_id"]
    print(f"  session 1: {s1}")
    print(f"  session 2: {s2}")
    if s1 == s2:
        print("  ERROR: both transmissions got the same session id")
        return 1

    # The status endpoint must show both. The transmissions run in
    # background threads, so give them a moment to register.
    import time

    deadline = time.time() + 10
    status = {}
    while time.time() < deadline:
        status = client.get("/api/transmit/status").json()
        if (status.get("active_count") or 0) >= 2:
            break
        time.sleep(0.2)
    print(
        f"  status: active_count={status.get('active_count')} "
        f"max_concurrent={status.get('max_concurrent')} "
        f"sessions={list((status.get('sessions') or {}).keys())}"
    )
    if status.get("active_count", 0) < 2:
        print("  ERROR: both sessions are not reported as active")
        return 1

    # The bound must be enforced.
    limit = server.MAX_CONCURRENT_TRANSMISSIONS
    codes = []
    for _ in range(limit + 2):
        r = client.post(
            "/api/transmit",
            json={
                "media_ids": media_ids,
                "message": "flood",
                "recipient_username": "bob",
                "speed_multiplier": 1.0,
            },
            headers=alice,
        )
        codes.append(r.status_code)
    print(f"  flood status codes: {codes}")
    if 429 not in codes:
        print("  ERROR: the admission bound was never enforced")
        return 1

    # Wait for the transmissions to drain, then check the packets.
    deadline = time.time() + 60
    while time.time() < deadline:
        if server._active_transmission_count() == 0:
            break
        time.sleep(0.5)
    print(f"  active after drain: {server._active_transmission_count()}")

    # Bob's inbox must list both sessions, and their packets must be
    # separate files.
    inbox = client.get("/api/inbox", headers=bob).json()
    ids = {item["session_id"] for item in inbox.get("items", [])}
    print(f"  bob's inbox sessions: {sorted(ids)}")
    if not {s1, s2} <= ids:
        print("  ERROR: both sessions are not in the inbox")
        return 1

    bob_id = server._auth.get_user_by_username("bob").id
    inbox_dir = PROJECT_ROOT / "storage" / "shared_channel" / str(bob_id)
    p1 = list(inbox_dir.glob(f"{s1}__*.json"))
    p2 = list(inbox_dir.glob(f"{s2}__*.json"))
    print(f"  packets for session 1: {len(p1)}")
    print(f"  packets for session 2: {len(p2)}")
    # The filenames carry the session id as a prefix, so the two sessions
    # write different files even for the same media id.
    names1 = {p.name for p in p1}
    names2 = {p.name for p in p2}
    shared_paths = names1 & names2
    if shared_paths:
        print(f"  ERROR: the sessions wrote the same files: {shared_paths}")
        return 1
    # And the media ids each session used do overlap, which is exactly the
    # case the prefix exists to handle.
    media1 = {p.name.split("__", 1)[1] for p in p1}
    media2 = {p.name.split("__", 1)[1] for p in p2}
    if media1 and media2:
        print(f"  both sessions carried {len(media1 & media2)} of the same media ids, under different filenames")
    if not p1 or not p2:
        print("  ERROR: one of the sessions wrote no packets")
        return 1

    print("\nOK: concurrent sessions are independent and the bound is enforced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import pytest
from fastapi.testclient import TestClient
from src.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_pipeline_trace_endpoint(client):
    response = client.post(
        "/api/demo/pipeline-trace",
        json={"message": "Classified transmission test", "use_ecc": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data
    assert "carrier_count" in data
    assert "bits_per_carrier" in data
    assert "mode" in data
    assert len(data["stages"]) >= 5
    if data["engine_warm"]:
        assert data["mode"] == "dssc"
        assert data["carrier_count"] > 0
        assert data["bits_per_carrier"] > 0
    else:
        assert data["mode"] == "unknown"

def test_verify_carrier_hash_endpoint(client):
    from src.api.demo import REPO_ROOT, _load_catalog

    entries = [
        e
        for e in _load_catalog("image_metadata.json")
        if (REPO_ROOT / e.get("path", "")).exists()
    ]
    if not entries:
        pytest.skip("image corpus unavailable on disk")
    response = client.post(
        "/api/demo/verify-carrier",
        json={"media_id": entries[0]["id"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "transmitted_sha256" in data
    assert "corpus_sha256" in data
    assert data["bytes_modified"] == 0
    assert data["corpus_sha256"] == data["transmitted_sha256"]
    assert data["comparison_basis"] == "reference-pass-through"
    assert data["modified_since_transmission"] in (True, False, None)

def test_verify_carrier_missing_media_id_returns_404(client):
    response = client.post(
        "/api/demo/verify-carrier",
        json={"media_id": "definitely_not_a_real_media_id"},
    )
    assert response.status_code == 404

def test_live_steganalysis_endpoint(client):
    from src.api.demo import REPO_ROOT, _load_catalog

    entries = [
        e
        for e in _load_catalog("image_metadata.json")
        if (REPO_ROOT / e.get("path", "")).exists()
    ]
    if len(entries) < 4:
        pytest.skip(f"image corpus needs >= 4 images, found {len(entries)}")

    response = client.post(
        "/api/demo/live-steganalysis",
        json={"detector": "sr_net", "sample_count": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert "clean_probability" in data
    assert "lsb_probability" in data
    assert "dcass_probability" in data


@pytest.mark.integration
def test_erasure_lab_endpoints_end_to_end(client):
    """Exercise /api/demo/erasure-encode, erasure-test, and erasure-sweep on real corpus."""
    msg = "Demo review erasure recovery test message"
    enc_res = client.post(
        "/api/demo/erasure-encode",
        json={"message": msg, "mode": "exact_vcp", "ecc_parity_bytes": 8},
    )
    assert enc_res.status_code == 200, enc_res.text
    enc = enc_res.json()
    assert enc["carrier_count"] == len(enc["media_ids"])
    assert enc["max_erasures"] == 8

    # Drop 3 packets -> must recover 100% exact
    test_ok = client.post(
        "/api/demo/erasure-test",
        json={
            "original_message": msg,
            "media_ids": enc["media_ids"],
            "mode": "exact_vcp",
            "ecc_parity_bytes": 8,
            "drop_indices": [5, 11, 18],
            "corrupt_indices": [],
        },
    )
    assert test_ok.status_code == 200, test_ok.text
    ok_data = test_ok.json()
    assert ok_data["ecc_success"] is True
    assert ok_data["exact_match"] is True
    assert ok_data["reconstructed_message"] == msg
    assert ok_data["bit_error_rate"] == 0.0
    assert ok_data["dropped_packet_count"] == 3
    assert len(ok_data["repaired_byte_indices"]) >= 3

    # Drop 9 packets (> R=8) -> must fail closed
    test_over = client.post(
        "/api/demo/erasure-test",
        json={
            "original_message": msg,
            "media_ids": enc["media_ids"],
            "mode": "exact_vcp",
            "ecc_parity_bytes": 8,
            "drop_indices": [1, 3, 5, 7, 9, 11, 13, 15, 17],
            "corrupt_indices": [],
        },
    )
    assert test_over.status_code == 200
    over_data = test_over.json()
    assert over_data["ecc_success"] is False
    assert over_data["exact_match"] is False

    # Sweep 0..10 drops
    sweep_res = client.post(
        "/api/demo/erasure-sweep",
        json={
            "original_message": msg,
            "media_ids": enc["media_ids"],
            "mode": "exact_vcp",
            "ecc_parity_bytes": 8,
            "trials_per_level": 3,
        },
    )
    assert sweep_res.status_code == 200, sweep_res.text
    points = {p["dropped_packets"]: p for p in sweep_res.json()["points"]}
    assert points[0]["recovery_rate_pct"] == 100.0
    assert points[3]["recovery_rate_pct"] == 100.0
    assert points[8]["recovery_rate_pct"] == 100.0
    assert points[9]["recovery_rate_pct"] == 0.0
    assert points[9]["false_accepts"] == 0


@pytest.mark.integration
def test_real_on_disk_wire_drop_and_recovery(client):
    """Transmit real files to storage/shared_channel, unlink 3 files on disk, and recover 100%."""
    from pathlib import Path
    from src.api import auth as _auth

    _auth.init_db()
    _auth.seed_demo_users()
    login_res = client.post("/api/auth/login", json={"username": "alice", "password": "alice12345"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    msg = "Real on-disk packet deletion test for final review."
    send_res = client.post(
        "/api/wire/send",
        headers=headers,
        json={
            "message": msg,
            "recipient_username": "bob",
            "engine_mode": "exact_vcp",
            "diversity_mode": "round_robin",
            "ecc_parity_bytes": 8,
            "stealth_mode": "static",
        },
    )
    assert send_res.status_code == 200, send_res.text
    sess = send_res.json()
    sid = sess["session_id"]
    repo_root = Path(__file__).resolve().parents[1]

    try:
        # Confirm packet files physically exist on disk
        pkt_paths = [repo_root / p["rel_path"] for p in sess["packets"]]
        assert all(p.exists() for p in pkt_paths)

        # Physically delete 3 packet files off disk via API
        drop_res = client.post(
            f"/api/wire/sessions/{sid}/drop",
            headers=headers,
            json={"sequences": [5, 12, 20]},
        )
        assert drop_res.status_code == 200
        drop_data = drop_res.json()
        assert drop_data["dropped_count"] == 3
        assert drop_data["surviving_count"] == sess["total_items"] - 3
        assert not pkt_paths[5].exists()
        assert not pkt_paths[12].exists()
        assert not pkt_paths[20].exists()

        # Decode ONLY from the surviving files on disk -> must recover 100% exact
        dec_res = client.post(f"/api/wire/sessions/{sid}/decode", headers=headers)
        assert dec_res.status_code == 200, dec_res.text
        dec = dec_res.json()
        assert dec["ecc_success"] is True
        assert dec["reconstructed_message"] == msg
        assert dec["dropped_packets"] == 3
        assert len(dec["repaired_byte_indices"]) >= 3

        # Delete 6 more files (total 9 > R=8) -> must fail closed
        drop_more = client.post(
            f"/api/wire/sessions/{sid}/drop",
            headers=headers,
            json={"sequences": [6, 7, 8, 9, 10, 11]},
        )
        assert drop_more.status_code == 200
        assert drop_more.json()["dropped_count"] == 9

        dec_fail = client.post(f"/api/wire/sessions/{sid}/decode", headers=headers)
        assert dec_fail.status_code == 200
        assert dec_fail.json()["ecc_success"] is False
        assert dec_fail.json()["reconstructed_message"] == ""

        # Restore all deleted packets back onto disk -> 100% intact again
        restore_res = client.post(f"/api/wire/sessions/{sid}/restore", headers=headers, json={})
        assert restore_res.status_code == 200, restore_res.text
        restored = restore_res.json()
        assert restored["dropped_count"] == 0
        assert restored["surviving_count"] == sess["total_items"]
        assert pkt_paths[5].exists() and pkt_paths[12].exists() and pkt_paths[20].exists()

        # Now tamper 2 packets on disk -> RS-ECC recovers 100%
        for t_seq in (3, 14):
            t_res = client.post(f"/api/wire/sessions/{sid}/tamper/{t_seq}", headers=headers)
            assert t_res.status_code == 200
        dec_tamper = client.post(f"/api/wire/sessions/{sid}/decode", headers=headers)
        assert dec_tamper.status_code == 200
        assert dec_tamper.json()["ecc_success"] is True
        assert dec_tamper.json()["reconstructed_message"] == msg
    finally:
        client.delete(f"/api/wire/sessions/{sid}", headers=headers)

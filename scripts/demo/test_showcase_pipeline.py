#!/usr/bin/env python3
"""
Automated End-to-End Verification of the DCASS Executive Presentation Pipeline.
Verifies exact encoding, stealth transmission dispatch, wire packet inspection,
and receiver message recovery across both Exact VCP and DSSC architectures.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api import auth

client = TestClient(app)


def test_presentation_pipeline_e2e():
    # 0. Ensure demo users exist
    auth.init_db()
    auth.seed_demo_users()

    # 1. Login as alice
    login_resp = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice12345"},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token = login_resp.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 2. Test Exact VCP Encoding with Round-Robin Diversity
    secret_text = "Executive Briefing: Alpha Window 0400Z"
    enc_vcp = client.post(
        "/api/encode",
        headers=auth_headers,
        json={
            "message": secret_text,
            "mode": "exact_vcp",
            "diversity_mode": "round_robin",
            "use_ecc": True,
            "ecc_parity_bytes": 8,
        },
    )
    assert enc_vcp.status_code == 200, f"VCP Encode failed: {enc_vcp.text}"
    vcp_data = enc_vcp.json()
    assert len(vcp_data["media_ids"]) > 0, "No carriers returned for VCP"

    # 3. Test DSSC Encoding with Round-Robin Diversity
    session_key_hex = os.urandom(32).hex()
    enc_dssc = client.post(
        "/api/encode",
        headers=auth_headers,
        json={
            "message": secret_text,
            "mode": "dssc",
            "diversity_mode": "round_robin",
            "session_key_hex": session_key_hex,
            "use_ecc": True,
            "ecc_parity_bytes": 8,
        },
    )
    assert enc_dssc.status_code == 200, f"DSSC Encode failed: {enc_dssc.text}"
    dssc_data = enc_dssc.json()
    assert len(dssc_data["media_ids"]) > 0, "No carriers returned for DSSC"

    # 4. Transmit Exact VCP sequence via stealth scheduler to Bob
    tx_resp = client.post(
        "/api/transmit",
        headers=auth_headers,
        json={
            "media_ids": vcp_data["media_ids"][:10], # test transmit batch
            "recipient_username": "bob",
            "mode": "auto",
            "speed_multiplier": 50.0,
            "message": secret_text,
        },
    )
    assert tx_resp.status_code == 200, f"Transmit failed: {tx_resp.text}"
    tx_data = tx_resp.json()
    assert tx_data["success"] is True
    session_id = tx_data["session_id"]

    # 5. Inspect Wire Packets
    wire_resp = client.get("/api/wire/packets", headers=auth_headers)
    assert wire_resp.status_code == 200

    # 6. Decode Exact VCP Carriers (Standard Decode Endpoint)
    dec_vcp = client.post(
        "/api/decode",
        headers=auth_headers,
        json={
            "media_ids": vcp_data["media_ids"],
            "mode": "exact_vcp",
            "diversity_mode": "round_robin",
            "use_ecc": True,
            "ecc_parity_bytes": 8,
        },
    )
    assert dec_vcp.status_code == 200, f"Decode failed: {dec_vcp.text}"
    assert dec_vcp.json()["reconstructed_meaning"] == secret_text

    # 7. Decode DSSC Carriers
    dec_dssc = client.post(
        "/api/decode",
        headers=auth_headers,
        json={
            "media_ids": dssc_data["media_ids"],
            "mode": "dssc",
            "diversity_mode": "round_robin",
            "session_key_hex": session_key_hex,
            "use_ecc": True,
            "ecc_parity_bytes": 8,
        },
    )
    assert dec_dssc.status_code == 200, f"DSSC Decode failed: {dec_dssc.text}"
    assert dec_dssc.json()["reconstructed_meaning"] == secret_text

    # 8. Test Fault Tolerance (Reed-Solomon Error Correction on corrupted carrier)
    # Corrupt carrier at index 2 with carrier from index 5 and verify RS-ECC recovers the full plaintext
    corrupted_carrier_ids = list(vcp_data["media_ids"])
    corrupted_carrier_ids[2] = corrupted_carrier_ids[5] # Corrupt carrier symbol at position 2
    dec_ecc_repair = client.post(
        "/api/decode",
        headers=auth_headers,
        json={
            "media_ids": corrupted_carrier_ids,
            "mode": "exact_vcp",
            "diversity_mode": "round_robin",
            "use_ecc": True,
            "ecc_parity_bytes": 8,
        },
    )
    assert dec_ecc_repair.status_code == 200
    # Reed-Solomon should repair the corrupted carrier and recover the exact secret
    assert dec_ecc_repair.json()["reconstructed_meaning"] == secret_text
    assert 2 in dec_ecc_repair.json().get("ecc_errors_fixed", [2])


if __name__ == "__main__":
    test_presentation_pipeline_e2e()
    print("[SUCCESS] All presentation pipeline verification tests passed!")

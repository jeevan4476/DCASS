# tests/test_api_endpoints.py
"""
Comprehensive tests for FastAPI endpoints in src/api/server.py:
- GET /api/health
- GET /api/status
- GET /api/ready
- POST /api/encode
- POST /api/decode
- POST /api/search
- GET /api/wire/packets
- DELETE /api/wire/packets
- POST /api/transmit
- GET /api/transmit/status
- End-to-end encoding and decoding flow
"""

import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_ready():
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "initializing" in data
    assert "encoder_loaded" in data
    assert "decoder_loaded" in data


def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "indices" in data
    assert "total_items" in data
    assert "device" in data
    assert "stealth_models" in data
    assert isinstance(data["total_items"], int)
    assert data["total_items"] >= 0


@pytest.mark.integration
def test_api_encode_and_decode_e2e():
    # 1. Test POST /api/encode
    encode_payload = {
        "message": "Secret operation delta at midnight",
        "mode": "best",
        "modalities": ["image", "text", "audio"]
    }
    encode_res = client.post("/api/encode", json=encode_payload)
    assert encode_res.status_code == 200
    encode_data = encode_res.json()

    assert "media_ids" in encode_data
    assert "chunks" in encode_data
    assert "encoded" in encode_data
    assert "media_sequence" in encode_data
    assert "modality_breakdown" in encode_data
    assert "elapsed_ms" in encode_data

    media_ids = encode_data["media_ids"]
    assert len(media_ids) > 0
    assert len(encode_data["encoded"]) == len(media_ids)
    assert len(encode_data["media_sequence"]) == len(media_ids)
    assert len(encode_data["chunks"]) > 0

    for item in encode_data["encoded"]:
        assert "file_path" in item
        assert isinstance(item["file_path"], str)
        assert len(item["file_path"]) > 0

    for item in encode_data["media_sequence"]:
        assert "file_path" in item
        assert isinstance(item["file_path"], str)
        assert len(item["file_path"]) > 0

    # 2. Test POST /api/decode with returned media_ids
    decode_payload = {"media_ids": media_ids}
    decode_res = client.post("/api/decode", json=decode_payload)
    assert decode_res.status_code == 200
    decode_data = decode_res.json()

    assert "reconstructed_meaning" in decode_data
    assert "items" in decode_data
    assert "decoded" in decode_data
    assert "verification_rate" in decode_data
    assert "all_verified" in decode_data
    assert "elapsed_ms" in decode_data

    assert decode_data["all_verified"] is True
    assert decode_data["verification_rate"] == 1.0
    assert len(decode_data["items"]) == len(media_ids)
    assert len(decode_data["decoded"]) == len(media_ids)
    assert len(decode_data["reconstructed_meaning"]) > 0

    for item in decode_data["items"]:
        assert "file_path" in item
        assert isinstance(item["file_path"], str)
        assert len(item["file_path"]) > 0

    for item in decode_data["decoded"]:
        assert "file_path" in item
        assert isinstance(item["file_path"], str)
        assert len(item["file_path"]) > 0


@pytest.mark.integration
def test_api_search():
    search_payload = {
        "query": "golden retriever dog in grass",
        "k": 3,
        "modalities": ["image", "text"]
    }
    res = client.post("/api/search", json=search_payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    for item in data["results"]:
        assert "id" in item
        assert "modality" in item
        assert "score" in item
        assert "content" in item


def test_api_wire_packets_flow():
    # 1. Clear packets
    del_res = client.delete("/api/wire/packets")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 2. Get packets (should be empty)
    get_res = client.get("/api/wire/packets")
    assert get_res.status_code == 200
    assert get_res.json()["count"] == 0

    # 3. Transmit packets
    transmit_payload = {
        "media_ids": ["img_001", "txt_002"],
        "mode": "static",
        "base_delay": 0.01,
        "speed_multiplier": 100.0,
        "message": "test msg"
    }
    tx_res = client.post("/api/transmit", json=transmit_payload)
    assert tx_res.status_code == 200
    assert tx_res.json()["success"] is True

    # 4. Check status
    st_res = client.get("/api/transmit/status")
    assert st_res.status_code == 200
    assert "active" in st_res.json()


def test_api_cors_preflight_and_headers():
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"):
        # Preflight OPTIONS check
        opt_res = client.options(
            "/api/ready",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert opt_res.status_code == 200
        assert opt_res.headers.get("access-control-allow-origin") == origin

        # Actual GET request with Origin
        get_res = client.get("/api/ready", headers={"Origin": origin})
        assert get_res.status_code == 200
        assert get_res.headers.get("access-control-allow-origin") == origin

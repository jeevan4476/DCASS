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

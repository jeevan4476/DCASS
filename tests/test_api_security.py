# tests/test_api_security.py
"""
Regression tests for API/security hardening:
- Bearer auth when DCASS_API_TOKEN is set
- Transmit media_id path traversal rejection
- Doctor path redaction
- Invalid context_epoch_hint -> 400
- Context mode obfuscation vs keyed
- Full-seed permutation + HMAC framing
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.server import (
    app,
    sanitize_packet_filename,
    validate_transmit_media_ids,
)


@pytest.fixture()
def client():
    return TestClient(app)


class TestApiAuth:
    def test_sensitive_routes_require_bearer_when_token_set(self, client):
        with patch.dict(os.environ, {"DCASS_API_TOKEN": "test-token-xyz"}, clear=False):
            assert client.get("/api/doctor").status_code == 401
            assert client.post("/api/encode", json={"message": "x"}).status_code == 401
            assert client.post("/api/decode", json={"media_ids": []}).status_code == 401
            assert client.delete("/api/wire/packets").status_code == 401
            assert client.post(
                "/api/transmit",
                json={"media_ids": ["ok_id"], "speed_multiplier": 100},
            ).status_code == 401

            headers = {"Authorization": "Bearer test-token-xyz"}
            # Doctor may fail readiness but must not be 401 once authorized.
            doctor = client.get("/api/doctor", headers=headers)
            assert doctor.status_code != 401

    def test_health_remains_open_with_token_configured(self, client):
        with patch.dict(os.environ, {"DCASS_API_TOKEN": "test-token-xyz"}, clear=False):
            assert client.get("/api/health").status_code == 200


class TestTransmitPathSanitization:
    def test_validate_rejects_traversal(self):
        with pytest.raises(ValueError):
            validate_transmit_media_ids(["../../tmp/evil"])
        with pytest.raises(ValueError):
            validate_transmit_media_ids(["foo/bar"])
        with pytest.raises(ValueError):
            validate_transmit_media_ids([".."])
        validate_transmit_media_ids(["image_0001", "text-ok.wav"])

    def test_sanitize_packet_filename_stays_in_shared_dir(self, tmp_path):
        shared = tmp_path / "shared_channel"
        shared.mkdir()
        path = sanitize_packet_filename(shared, "media_abc", channel=0, idx=1)
        assert path.is_relative_to(shared.resolve())
        assert path.name == "media_abc_0_0001.json"

        with pytest.raises(ValueError):
            sanitize_packet_filename(shared, "../escape", channel=0, idx=0)

    def test_transmit_rejects_traversal_with_400(self, client):
        res = client.post(
            "/api/transmit",
            json={
                "media_ids": ["../../../../tmp/evil"],
                "speed_multiplier": 100.0,
                "base_delay": 0.01,
            },
        )
        assert res.status_code == 400
        assert "invalid media_id" in res.json()["detail"].lower() or "path" in res.json()[
            "detail"
        ].lower()


class TestDoctorPublicDict:
    def test_to_public_dict_scrubs_absolute_paths(self, tmp_path):
        from src.diagnostics.doctor import CheckResult, DoctorReport

        root = tmp_path / "proj"
        root.mkdir()
        abs_path = root / "storage" / "indices" / "voronoi_codebook.npz"
        report = DoctorReport(
            checks=[
                CheckResult(
                    "codebook:present",
                    True,
                    True,
                    detail=str(abs_path),
                )
            ]
        )
        public = report.to_public_dict(project_root=root)
        detail = public["checks"][0]["detail"]
        assert str(root) not in detail
        assert detail.startswith(".") or "voronoi_codebook" in detail


class TestContextEpochHint:
    def test_malformed_hint_raises_value_error(self):
        from src.engine.context import ContextKeyManager

        mgr = ContextKeyManager(bucket_seconds=3600)
        with pytest.raises(ValueError):
            mgr.resolve_epoch_hint("not-a-hint")
        with pytest.raises(ValueError):
            mgr.resolve_epoch_hint("abc|3600|time")
        with pytest.raises(ValueError):
            mgr.resolve_epoch_hint("100|0|time")

    def test_valid_time_hint_resolves(self):
        from src.engine.context import ContextKeyManager

        mgr = ContextKeyManager(bucket_seconds=3600)
        epoch = mgr.current_epoch()
        resolved = mgr.resolve_epoch_hint(epoch.epoch_id)
        assert resolved.bucket_start == epoch.bucket_start
        assert resolved.bucket_seconds == epoch.bucket_seconds

    def test_decode_api_returns_400_on_bad_hint(self, client):
        from src.engine.decoder import SemanticDecoder
        from tests.test_engine.test_exact_vcp_recovery import FakeIndex

        decoder = SemanticDecoder(index=FakeIndex())
        decoder._loaded = True
        with patch("src.api.server._get_decoder", return_value=decoder):
            res = client.post(
                "/api/decode",
                json={
                    "media_ids": ["x"],
                    "use_dynamic_context": True,
                    "context_epoch_hint": "100|notanint|time",
                },
            )
        assert res.status_code == 400
        assert "context_epoch_hint" in res.json()["detail"].lower() or "integer" in res.json()[
            "detail"
        ].lower()


class TestContextModeAndCrypto:
    def test_secret_changes_permutation(self):
        from src.engine.context import ContextKeyManager

        ts = 1_750_000_000.0
        plain = ContextKeyManager(bucket_seconds=3600)
        keyed = ContextKeyManager(bucket_seconds=3600, secret=b"s3cret")
        p1 = plain.derive_permutation(plain.epoch_at(ts))
        p2 = keyed.derive_permutation(keyed.epoch_at(ts))
        assert not np.array_equal(p1, p2)
        assert sorted(p2.tolist()) == list(range(256))

    def test_full_digest_seed_is_deterministic(self):
        from src.engine.context import ContextKeyManager

        ts = 1_750_000_000.0
        a = ContextKeyManager(bucket_seconds=3600, secret=b"k")
        b = ContextKeyManager(bucket_seconds=3600, secret=b"k")
        np.testing.assert_array_equal(
            a.derive_permutation(a.epoch_at(ts)),
            b.derive_permutation(b.epoch_at(ts)),
        )

    def test_hmac_frame_roundtrip(self):
        from src.engine.payload_framing import (
            FRAME_VERSION_HMAC,
            FrameError,
            frame_payload,
            unframe_payload,
        )

        secret = b"shared-out-of-band"
        frame = frame_payload("Meet at noon", secret=secret)
        assert frame[0] == FRAME_VERSION_HMAC
        text, framed = unframe_payload(frame, secret=secret)
        assert framed is True
        assert text == "Meet at noon"

        with pytest.raises(FrameError):
            unframe_payload(frame, secret=b"wrong")
        with pytest.raises(FrameError):
            unframe_payload(frame)  # HMAC frame requires secret

    def test_crc_frame_still_default_without_secret(self):
        from src.engine.payload_framing import FRAME_VERSION, frame_payload, unframe_payload

        frame = frame_payload("hello")
        assert frame[0] == FRAME_VERSION
        text, framed = unframe_payload(frame)
        assert framed and text == "hello"

    def test_build_context_manager_respects_env_secret(self):
        from src.api.server import _build_context_manager

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DCASS_CONTEXT_SECRET", None)
            mgr = _build_context_manager(3600)
            assert mgr.secret is None
            info_mode = "keyed" if mgr.secret else "obfuscation"
            assert info_mode == "obfuscation"

        with patch.dict(os.environ, {"DCASS_CONTEXT_SECRET": "env-secret"}, clear=False):
            mgr = _build_context_manager(3600)
            assert mgr.secret == b"env-secret"


class TestKeyedRoundtripWithHmac:
    """Encoder/decoder must round-trip when secret enables HMAC framing."""

    @pytest.fixture()
    def codec(self):
        from tests.test_engine.test_exact_vcp_recovery import FakeIndex
        from src.engine.encoder import SemanticEncoder
        from src.engine.decoder import SemanticDecoder

        index = FakeIndex()
        encoder = SemanticEncoder(index=index)
        encoder._loaded = True
        decoder = SemanticDecoder(index=index)
        decoder._loaded = True
        return encoder, decoder

    def test_hmac_keyed_roundtrip(self, codec):
        from src.engine.context import ContextKeyManager

        encoder, decoder = codec
        mgr = ContextKeyManager(bucket_seconds=3600, secret=b"roundtrip-secret")
        msg = "HMAC keyed channel"
        result = encoder.encode(
            msg, payload_mode="exact_vcp", use_ecc=True, context_manager=mgr
        )
        assert result.context_info.get("context_mode") == "keyed"
        decoded = decoder.decode(
            result.media_ids,
            payload_mode="exact_vcp",
            use_ecc=True,
            context_manager=mgr,
            context_epoch_hint=result.context_info["epoch_id"],
        )
        assert decoded.ecc_success
        assert decoded.reconstructed_meaning == msg

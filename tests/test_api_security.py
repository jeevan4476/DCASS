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

    def test_production_requires_bearer_without_api_token(self, client):
        with patch.dict(
            os.environ,
            {
                "DCASS_ENV": "production",
                "DCASS_JWT_SECRET": "test-jwt-secret",
            },
            clear=False,
        ):
            os.environ.pop("DCASS_API_TOKEN", None)
            assert client.get("/api/auth/me").status_code == 401
            assert client.get("/api/wire/packets").status_code == 401
            assert client.get("/api/users").status_code == 401

    def test_production_refuses_to_serve_open_when_api_token_unset(self, client):
        """The token-optional skip must fail closed, never silently open."""
        with patch.dict(
            os.environ,
            {"DCASS_ENV": "production", "DCASS_JWT_SECRET": "test-jwt-secret"},
            clear=False,
        ):
            os.environ.pop("DCASS_API_TOKEN", None)
            # Routes guarded by require_api_token must refuse, with a 503
            # that names the missing configuration, rather than serve.
            res = client.post("/api/encode", json={"message": "x"})
            assert res.status_code == 503
            assert "DCASS_API_TOKEN" in res.json()["detail"]

    def test_valid_token_for_unknown_user_is_rejected_not_fallback(self, client):
        """A structurally valid token for a missing user must never become alice."""
        import time as _time
        from jose import jwt as _jwt

        from src.api import auth as _auth

        claims = {
            "sub": "424242",  # no such user
            "exp": int(_time.time()) + 600,
        }
        token = _jwt.encode(claims, _auth._jwt_secret(), algorithm="HS256")
        headers = {"Authorization": f"Bearer {token}"}

        assert client.get("/api/auth/me", headers=headers).status_code == 401
        assert client.get("/api/users", headers=headers).status_code == 401


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
        assert path.name == "media_abc.json"

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

    def test_transmit_rejects_non_positive_speed_multiplier(self, client):
        res = client.post(
            "/api/transmit",
            json={
                "media_ids": ["ok_id"],
                "speed_multiplier": 0,
            },
        )
        assert res.status_code == 422


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
        from src.engine.semantic_engine import SemanticEngine

        decoder = SemanticDecoder(index=FakeIndex())
        decoder._loaded = True
        engine = SemanticEngine()
        engine._loaded = True
        engine._exact_decoder = decoder

        with patch("src.api.server._get_engine", return_value=engine):
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
                msg, use_ecc=True, context_manager=mgr
            )
            assert result.context_info.get("context_mode") == "keyed"
            decoded = decoder.decode(
                result.media_ids,
                use_ecc=True,
                context_manager=mgr,
                context_epoch_hint=result.context_info["epoch_id"],
            )
            assert decoded.ecc_success
            assert decoded.reconstructed_meaning == msg


class TestOwnershipScoping:
    """Destructive and lifecycle verbs act on the caller's own objects only."""

    def _register(self, client, username):
        from src.api import auth as _auth

        client.post(
            "/api/auth/register", json={"username": username, "password": "pw-owner-1"}
        )
        user = _auth.get_user_by_username(username)
        token = _auth.create_access_token(user.id, user.username)
        return user, {"Authorization": f"Bearer {token}"}

    def test_delete_wire_packets_is_scoped_to_the_caller(self, client):
        import uuid
        from pathlib import Path as _P

        user_a, headers_a = self._register(client, "scoped_a")
        user_b, headers_b = self._register(client, "scoped_b")

        # The endpoint derives the directory from its module path, so stage
        # sentinels under the real storage tree with unique file names.
        real_base = _P("storage") / "shared_channel"
        real_mine = real_base / str(user_a.id) / f"{uuid.uuid4().hex}.json"
        real_theirs = real_base / str(user_b.id) / f"{uuid.uuid4().hex}.json"
        real_mine.parent.mkdir(parents=True, exist_ok=True)
        real_theirs.parent.mkdir(parents=True, exist_ok=True)
        real_mine.write_text('{"marker": "mine"}')
        real_theirs.write_text('{"marker": "theirs"}')
        try:
            res = client.delete("/api/wire/packets", headers=headers_a)
            assert res.status_code == 200
            assert res.json()["success"] is True
            assert not real_mine.exists()
            assert real_theirs.exists()  # untouched: belongs to another user

            res = client.delete("/api/wire/packets", headers=headers_b)
            assert res.status_code == 200
            assert not real_theirs.exists()
        finally:
            real_mine.unlink(missing_ok=True)
            real_theirs.unlink(missing_ok=True)

    def test_transmit_stop_is_scoped_to_the_owner(self, client):
        import uuid

        from src.api.server import (
            _transmission_sessions,
            _transmission_sessions_lock,
        )

        owner, headers_owner = self._register(
            client, f"stopowner_{uuid.uuid4().hex[:6]}"
        )
        other, headers_other = self._register(
            client, f"stopother_{uuid.uuid4().hex[:6]}"
        )

        sid = f"test-stop-{uuid.uuid4().hex[:8]}"
        with _transmission_sessions_lock:
            _transmission_sessions[sid] = {
                "active": True,
                "stop_requested": False,
                "progress": {"current": 0, "total": 3, "status": "running"},
                "owner": owner.id,
            }
        try:
            # Another user, by id: another owner's (or unknown) session is a
            # 404, never a stop and never an existence confirmation.
            res = client.post(
                f"/api/transmit/stop?session_id={sid}", headers=headers_other
            )
            assert res.status_code == 404
            with _transmission_sessions_lock:
                assert _transmission_sessions[sid]["stop_requested"] is False

            # Another user, blanket stop: reaches nothing of the owner's.
            res = client.post("/api/transmit/stop", headers=headers_other)
            assert res.status_code == 200
            with _transmission_sessions_lock:
                assert _transmission_sessions[sid]["active"] is True
                assert _transmission_sessions[sid]["stop_requested"] is False

            # The owner stops their own session.
            res = client.post(
                f"/api/transmit/stop?session_id={sid}", headers=headers_owner
            )
            assert res.status_code == 200
            with _transmission_sessions_lock:
                assert _transmission_sessions[sid]["stop_requested"] is True
        finally:
            with _transmission_sessions_lock:
                _transmission_sessions.pop(sid, None)

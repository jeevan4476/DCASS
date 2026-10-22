# tests/test_api_auth.py
"""
Unit tests for the auth stack's token lifecycle. The existing API security
tests (test_api_security.py) cover only the missing-bearer 401 responses;
every run leaves the rest untested:

- user CRUD and authenticate() against a per-test SQLite database
- JWT issue/decode, expiry, and the production secret guard
- get_current_user's valid, invalid, expired, missing-token and
  valid-token-for-unknown-user branches, including the dev default user
- the /api/auth/register, /api/auth/login, /api/auth/me and /api/users
  endpoints, which no test on any path exercised before this file
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from src.api import auth
from src.api.server import app


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture()
def auth_db(monkeypatch, tmp_path):
    """A per-test users database and a clean development-mode environment.

    The auth module reads its module-level DB_PATH constant at call time,
    so redirecting the attribute redirects every _connect() in the test.
    """
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "auth.db")
    for name in ("DCASS_API_TOKEN", "DCASS_JWT_SECRET", "DCASS_ENV"):
        monkeypatch.delenv(name, raising=False)
    auth.init_db()
    return tmp_path


@pytest.fixture()
def client():
    # No `with`: as in test_api_security.py, the server's lifespan does not
    # run and the warm-up thread never starts. auth_db stands in for the
    # startup init_db call against the redirected database.
    return TestClient(app)


class TestUserCrud:
    def test_create_and_lookup(self, auth_db):
        user = auth.create_user("carol", "pw-carol-12")

        assert user.username == "carol"
        assert int(user.created_at) > 0
        assert auth.get_user_by_username("carol").id == user.id
        assert auth.get_user_by_id(user.id).username == "carol"
        assert auth.get_user_by_username("nobody") is None
        assert auth.get_user_by_id(9999) is None

    def test_rejects_short_password(self, auth_db):
        with pytest.raises(ValueError, match="at least 8 characters"):
            auth.create_user("carol", "pw-7")

    def test_rejects_duplicate_username(self, auth_db):
        auth.create_user("carol", "pw-carol-12")
        with pytest.raises(ValueError, match="already exists"):
            auth.create_user("carol", "pw-carol-34")

    def test_list_all_and_exclude(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")
        dave = auth.create_user("dave", "pw-dave-1234")

        assert [u.username for u in auth.list_all_users()] == ["carol", "dave"]
        assert [u.username for u in auth.list_users_excluding(dave.id)] == ["carol"]
        assert [u.username for u in auth.list_users_excluding(carol.id)] == ["dave"]
        assert len(auth.list_users_excluding(0)) == 2

    def test_password_hashing_roundtrip(self, auth_db):
        from src.api.auth import hash_password, verify_password

        digest = hash_password("carol-pw-123")
        assert digest != "carol-pw-123"
        assert verify_password("carol-pw-123", digest) is True
        assert verify_password("wrong-password", digest) is False

    def test_authenticate(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")

        assert auth.authenticate("carol", "pw-carol-12").id == carol.id
        assert auth.authenticate("carol", "wrong-password") is None
        assert auth.authenticate("nobody", "pw-carol-12") is None

    def test_seed_demo_users_is_idempotent(self, auth_db, capsys):
        auth.seed_demo_users()

        names = [u.username for u in auth.list_all_users()]
        assert names == ["alice", "bob", "charlie"]
        assert "Seeded demo users" in capsys.readouterr().out

        auth.seed_demo_users()
        assert [u.username for u in auth.list_all_users()] == names
        assert "Seeded demo users" not in capsys.readouterr().out


class TestJwt:
    def test_roundtrip_and_claims(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")

        token = auth.create_access_token(carol.id, carol.username)
        payload = auth.decode_access_token(token)

        assert payload["sub"] == str(carol.id)
        assert payload["username"] == "carol"
        assert payload["exp"] - payload["iat"] == auth.DEFAULT_TOKEN_TTL_SECONDS

        short = auth.decode_access_token(
            auth.create_access_token(carol.id, carol.username, ttl_seconds=60)
        )
        assert short["exp"] - short["iat"] == 60

    def test_secret_guard(self, monkeypatch):
        with patch.dict(os.environ, {"DCASS_ENV": "production"}, clear=False):
            with pytest.raises(RuntimeError, match="DCASS_JWT_SECRET must be set"):
                auth._jwt_secret()

        with patch.dict(
            os.environ,
            {"DCASS_ENV": "production", "DCASS_JWT_SECRET": "real-secret"},
            clear=False,
        ):
            assert auth._jwt_secret() == "real-secret"

    def test_dev_secret_marker(self, auth_db):
        assert "DO-NOT-USE-IN-PRODUCTION" in auth._jwt_secret()


class TestGetCurrentUser:
    def test_valid_token(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")
        token = auth.create_access_token(carol.id, carol.username)

        user = auth.get_current_user(_bearer(token))

        assert user.id == carol.id
        assert user.username == "carol"

    def test_garbage_token_rejected(self, auth_db):
        with pytest.raises(HTTPException) as exc:
            auth.get_current_user(_bearer("not-a-jwt"))
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")
        token = auth.create_access_token(carol.id, carol.username, ttl_seconds=-10)

        with pytest.raises(HTTPException) as exc:
            auth.get_current_user(_bearer(token))
        assert exc.value.status_code == 401

    def test_missing_user_after_valid_decode_falls_through(self, auth_db):
        # A decodable token naming a user that is not in the database does not
        # raise: control drops to the normal fall-through, so in dev mode the
        # request continues as the dev default user. Pinned semantics.
        carol = auth.create_user("carol", "pw-carol-12")
        token = auth.create_access_token(9999, "ghost")
        auth.seed_demo_users()

        user = auth.get_current_user(_bearer(token))

        assert user.username == "alice" and user.id != 9999
        assert user.id != carol.id  # 9999 never resolves

    def test_unknown_scheme_or_empty_credentials_ignored(self, auth_db):
        # A non-Bearer scheme or an empty credential is treated as no token
        # at all, so the dev default user is served for both.
        for creds in (
            HTTPAuthorizationCredentials(scheme="Basic", credentials="z"),
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
        ):
            assert auth.get_current_user(creds).username == "test_user"

    def test_dev_default_user_created(self, auth_db):
        user = auth.get_current_user(credentials=None)

        assert user.username == "test_user"
        # Second call finds the existing row instead of re-creating it.
        assert auth.get_current_user(credentials=None).username == "test_user"
        assert len(auth.list_all_users()) == 1

    def test_token_mode_still_requires_a_token(self, monkeypatch, auth_db):
        with patch.dict(os.environ, {"DCASS_API_TOKEN": "k"}, clear=False):
            with pytest.raises(HTTPException) as exc:
                auth.get_current_user(credentials=None)
        assert exc.value.status_code == 401
        assert auth.list_all_users() == []

    def test_optional_user(self, auth_db):
        carol = auth.create_user("carol", "pw-carol-12")
        token = auth.create_access_token(carol.id, carol.username)

        assert auth.get_optional_current_user(_bearer(token)).username == "carol"
        assert auth.get_optional_current_user(_bearer("not-a-jwt")) is None
        assert auth.get_optional_current_user(credentials=None) is None


class TestAuthEndpoints:
    def test_register_login_me_roundtrip(self, auth_db, client):
        register = client.post(
            "/api/auth/register",
            json={"username": "carol", "password": "pw-carol-12"},
        )
        assert register.status_code == 201
        body = register.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["username"] == "carol"
        carol_id = body["user"]["id"]

        wrong = client.post(
            "/api/auth/login",
            json={"username": "carol", "password": "wrong-password"},
        )
        assert wrong.status_code == 401

        login = client.post(
            "/api/auth/login",
            json={"username": "carol", "password": "pw-carol-12"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json() == {"id": carol_id, "username": "carol"}

    def test_me_without_token_uses_dev_default(self, auth_db, client):
        me = client.get("/api/auth/me")

        assert me.status_code == 200
        assert me.json()["username"] == "test_user"

    def test_register_rejects_duplicates_and_invalid_bodies(self, auth_db, client):
        ok = client.post(
            "/api/auth/register",
            json={"username": "carol", "password": "pw-carol-12"},
        )
        assert ok.status_code == 201

        duplicate = client.post(
            "/api/auth/register",
            json={"username": "carol", "password": "pw-carol-34"},
        )
        assert duplicate.status_code == 409

        short_password = client.post(
            "/api/auth/register",
            json={"username": "erin", "password": "pw-7"},
        )
        assert short_password.status_code == 422

        short_username = client.post(
            "/api/auth/register",
            json={"username": "ab", "password": "pw-erin-1234"},
        )
        assert short_username.status_code == 422

    def test_users_endpoint_lists_and_excludes(self, auth_db, client):
        for name, pw in (("carol", "pw-carol-12"), ("dave", "pw-dave-1234")):
            response = client.post(
                "/api/auth/register", json={"username": name, "password": pw}
            )
            assert response.status_code == 201

        token = client.post(
            "/api/auth/login", json={"username": "carol", "password": "pw-carol-12"}
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        everyone = client.get("/api/users")
        assert [u["username"] for u in everyone.json()["users"]] == ["carol", "dave"]

        excluded = client.get(
            "/api/users", params={"exclude_current": "true"}, headers=headers
        )
        assert [u["username"] for u in excluded.json()["users"]] == ["dave"]

# src/api/auth.py
"""
Minimal demo authentication for DCASS (Phase F of the GAN/RL integration
plan — see docs/GAN_RL_INTEGRATION_PLAN.md).

Deliberately NOT production-grade: bcrypt + SQLite + JWT, no OAuth, no
email verification, no password reset, no refresh-token rotation. This is
enough to demonstrate "Alice logs in and sends a message only Bob can
read" for a research-paper demo. See the plan's Phase F.6 (R6) for the
localStorage-JWT caveat and what a real deployment would need instead.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = _PROJECT_ROOT / "storage" / "auth.db"

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer_scheme = HTTPBearer(auto_error=False)

JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_TTL_SECONDS = 86400  # 24h


def _jwt_secret() -> str:
    """
    Resolve the JWT signing secret.

    Fails fast if DCASS_ENV=production and DCASS_JWT_SECRET is unset —
    a demo default secret must never sign tokens for a real deployment.
    Otherwise falls back to a fixed (but clearly-marked) dev secret so
    local runs don't need extra setup.
    """
    secret = os.environ.get("DCASS_JWT_SECRET")
    if secret:
        return secret
    if os.environ.get("DCASS_ENV", "development") == "production":
        raise RuntimeError(
            "DCASS_JWT_SECRET must be set when DCASS_ENV=production. "
            "Refusing to sign tokens with the dev default."
        )
    return "dcass-dev-secret-DO-NOT-USE-IN-PRODUCTION"


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the users table if it doesn't exist. Call once at startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


# Demo seed users — printed once at startup. Not a secret; this is a local
# research demo, not a deployment with real user data.
_SEED_USERS = [
    ("alice", "alice1234"),
    ("bob", "bob1234"),
    ("charlie", "charlie1234"),
]


def seed_demo_users() -> None:
    """Idempotently create the demo users (alice/bob/charlie) if missing."""
    created = []
    for username, password in _SEED_USERS:
        if get_user_by_username(username) is None:
            create_user(username, password)
            created.append(username)
    if created:
        print("=" * 60)
        print("[DCASS Auth] Seeded demo users (CHANGE PASSWORDS before any")
        print("             non-local use):")
        for username, password in _SEED_USERS:
            if username in created:
                print(f"               {username} / {password}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    created_at: str


def create_user(username: str, password: str) -> User:
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    password_hash = hash_password(password)
    created_at = str(int(time.time()))
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, created_at),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError(f"username {username!r} already exists") from e
        return User(id=cur.lastrowid, username=username, password_hash=password_hash, created_at=created_at)


def get_user_by_username(username: str) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return User(**dict(row)) if row else None


def get_user_by_id(user_id: int) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return User(**dict(row)) if row else None


def list_users_excluding(exclude_user_id: int) -> list[User]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE id != ? ORDER BY username",
            (exclude_user_id,),
        ).fetchall()
    return [User(**dict(r)) for r in rows]


def authenticate(username: str, password: str) -> Optional[User]:
    user = get_user_by_username(username)
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def create_access_token(user_id: int, username: str, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises JWTError on invalid/expired token."""
    return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


@dataclass
class AuthenticatedUser:
    id: int
    username: str


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")

    user_id = int(payload["sub"])
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return AuthenticatedUser(id=user.id, username=user.username)

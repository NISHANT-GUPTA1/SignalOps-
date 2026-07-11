"""Dead-simple, DB-free auth for a test/demo deployment.

Users live in a JSON file inside DATA_DIR (a mounted volume in Docker), so
registrations survive restarts without any database. Passwords are salted +
hashed (PBKDF2) — never stored in plaintext. Sessions are opaque tokens held
in memory and handed to the browser as an httpOnly cookie.

This is intentionally lightweight — good enough to gate a demo, not a bank.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading

from .config import settings

_USERS_PATH = os.path.join(settings.data_dir, "users.json")
_lock = threading.Lock()

# token -> username, kept in memory (cleared on restart; users just log in again)
_sessions: dict[str, str] = {}

# Optional seed login so the app is usable the moment it boots (even before
# anyone registers). Configure via env vars — NEVER hardcode credentials here.
# Set SEED_EMAIL + SEED_PASSWORD (e.g. in .env, which is gitignored) to enable;
# SEED_USERNAME is optional and defaults to the email. Leave unset to disable.
# Username or email both work as the login identifier.
def _seed_from_env() -> list[dict]:
    email = os.getenv("SEED_EMAIL", "").strip()
    password = os.getenv("SEED_PASSWORD", "")
    if not (email and password):
        return []
    username = os.getenv("SEED_USERNAME", "").strip() or email
    return [{"username": username, "email": email, "password": password}]


_SEED = _seed_from_env()

COOKIE_NAME = "bt_session"


# ---------------- password hashing ----------------
def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000)
    return dk.hex()


def _make_record(email: str, username: str, password: str) -> dict:
    salt = secrets.token_hex(16)
    return {"email": email, "username": username, "salt": salt, "hash": _hash_password(password, salt)}


# ---------------- storage ----------------
def _load() -> dict[str, dict]:
    try:
        with open(_USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(users: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(_USERS_PATH) or ".", exist_ok=True)
    tmp = _USERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, _USERS_PATH)


def _ensure_seed() -> None:
    with _lock:
        users = _load()
        changed = False
        for s in _SEED:
            key = s["username"].lower()
            if key not in users:
                users[key] = _make_record(s["email"], s["username"], s["password"])
                changed = True
        if changed:
            _save(users)


# Run once at import so the seed login always exists.
_ensure_seed()


# ---------------- public API ----------------
def register(email: str, username: str, password: str) -> None:
    """Create a new user. Raises ValueError on bad input or a taken name/email."""
    email = (email or "").strip().lower()
    username = (username or "").strip()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if len(username) < 2:
        raise ValueError("Username must be at least 2 characters.")
    if len(password or "") < 3:
        raise ValueError("Password must be at least 3 characters.")

    with _lock:
        users = _load()
        if username.lower() in users:
            raise ValueError("That username is already taken.")
        if any(u.get("email", "").lower() == email for u in users.values()):
            raise ValueError("That email is already registered.")
        users[username.lower()] = _make_record(email, username, password)
        _save(users)


def verify(identifier: str, password: str) -> str | None:
    """Check a username-or-email + password. Returns the canonical username or None."""
    identifier = (identifier or "").strip().lower()
    users = _load()
    rec = users.get(identifier)
    if rec is None:  # maybe they typed an email
        rec = next((u for u in users.values() if u.get("email", "").lower() == identifier), None)
    if rec is None:
        return None
    expected = rec.get("hash", "")
    got = _hash_password(password or "", rec.get("salt", ""))
    if hmac.compare_digest(expected, got):
        return rec.get("username")
    return None


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = username
    return token


def username_for_token(token: str | None) -> str | None:
    if not token:
        return None
    return _sessions.get(token)


def destroy_session(token: str | None) -> None:
    if token:
        _sessions.pop(token, None)

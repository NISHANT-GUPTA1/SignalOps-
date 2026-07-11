"""Read-only Slack integration (Slack Web API).

Lists channels and fetches recent messages so you can pick a real message and
triage it. It NEVER posts, edits, or deletes anything in Slack — read-only.

Requires a Bot token (xoxb-...) with scopes: channels:read, channels:history,
users:read (+ groups:read, groups:history for private channels). The bot must be
invited to a channel to read its history (`/invite @your-bot`).
"""
from __future__ import annotations

import requests

from ..config import settings

API = "https://slack.com/api"
_user_cache: dict[str, str] = {}


def is_configured() -> bool:
    return bool(settings.slack_bot_token)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.slack_bot_token}"}


def _call(method: str, params: dict) -> dict:
    resp = requests.get(f"{API}/{method}", headers=_headers(), params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API '{method}' error: {data.get('error', 'unknown')}")
    return data


def list_channels(limit: int = 200) -> list[dict]:
    """Public + private channels the bot can see (member channels first)."""
    data = _call(
        "conversations.list",
        {"types": "public_channel,private_channel", "exclude_archived": "true", "limit": limit},
    )
    chans = [
        {"id": c["id"], "name": c.get("name", c["id"]), "is_member": c.get("is_member", False)}
        for c in data.get("channels", [])
    ]
    chans.sort(key=lambda c: (not c["is_member"], c["name"]))
    return chans


def _username(user_id: str) -> str:
    if not user_id:
        return "unknown"
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        info = _call("users.info", {"user": user_id})["user"]
        name = info.get("real_name") or info.get("name") or user_id
    except Exception:
        name = user_id
    _user_cache[user_id] = name
    return name


def fetch_messages(channel_id: str, limit: int = 20) -> list[dict]:
    """Recent human messages in a channel (newest first), with author resolved."""
    data = _call("conversations.history", {"channel": channel_id, "limit": limit})
    out = []
    for m in data.get("messages", []):
        # skip joins/leaves/system events and bot posts
        if m.get("subtype") or m.get("bot_id"):
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "ts": m.get("ts", ""),
                "user": _username(m.get("user", "")),
                "text": text,
            }
        )
    return out

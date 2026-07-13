"""Block Kit renderers for the Slack agent.

Turns a `TriageResponse` into the rich, interactive card the bot posts back into
Slack, plus the follow-up states after an issue is filed. Kept separate from the
event handlers (app/slack_agent.py) so the presentation layer is easy to tweak.
"""
from __future__ import annotations

from .models import TriageResponse

# Priority / severity → emoji, so the card is scannable at a glance.
_PRIORITY_EMOJI = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🟢"}
_TYPE_EMOJI = {"bug": "🐞", "feature_request": "✨", "question": "❓", "task": "🧩"}
_CONF_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _trunc(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def triage_blocks(resp: TriageResponse, token: str, target: str, repo_url: str = "") -> list[dict]:
    """The main triage card. `token` keys the cached triage for the action buttons."""
    t = resp.triage
    pri = _PRIORITY_EMOJI.get(t.priority, "⚪")
    typ = _TYPE_EMOJI.get(t.type, "🐞")

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": _trunc(f"{typ} {t.title}", 150), "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Priority*\n{pri} {t.priority}"},
                {"type": "mrkdwn", "text": f"*Severity*\n{t.severity}"},
                {"type": "mrkdwn", "text": f"*Type*\n{t.type}"},
                {"type": "mrkdwn", "text": f"*Components*\n{', '.join(t.components) or '—'}"},
            ],
        },
    ]

    if t.priority_reason:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"*Why {t.priority}:* {_trunc(t.priority_reason, 400)}"}],
        })

    if t.summary:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary*\n{_trunc(t.summary, 2800)}"},
        })

    if t.reproduction_steps:
        steps = "\n".join(f"{i}. {_trunc(s, 200)}" for i, s in enumerate(t.reproduction_steps, 1))
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Steps to reproduce*\n{_trunc(steps, 2800)}"},
        })

    if t.expected_behavior or t.actual_behavior:
        exp = f"*Expected*\n{_trunc(t.expected_behavior, 1200)}" if t.expected_behavior else ""
        act = f"*Actual*\n{_trunc(t.actual_behavior, 1200)}" if t.actual_behavior else ""
        fields = [{"type": "mrkdwn", "text": x} for x in (exp, act) if x]
        blocks.append({"type": "section", "fields": fields})

    if t.duplicate_candidates:
        lines = []
        for d in t.duplicate_candidates[:5]:
            emoji = _CONF_EMOJI.get(d.confidence, "⚪")
            lines.append(f"{emoji} `{d.id}` {_trunc(d.title, 80)} — _{d.confidence}_ ({_trunc(d.reason, 120)})")
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*⚠️ Possible duplicates*\n" + "\n".join(lines)},
        })

    if t.suggested_next_action:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"➡️ *Next:* {_trunc(t.suggested_next_action, 400)}"}],
        })

    # Agentic-RAG trace: the searches Claude issued itself + how many docs grounded it.
    trace_bits = []
    if resp.agent_searches:
        qs = ", ".join(f"`{_trunc(q, 40)}`" for q in resp.agent_searches[:4])
        trace_bits.append(f"🔎 Agent searched: {qs}")
    if resp.retrieved:
        trace_bits.append(f"📚 grounded on {len(resp.retrieved)} existing docs")
    if trace_bits:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(trace_bits)}],
        })

    blocks.append({"type": "divider"})
    blocks.append(_actions_block(token, target, repo_url))
    return blocks


def _actions_block(token: str, target: str, repo_url: str) -> dict:
    """File / dismiss buttons. `value` carries the short cache token, not the payload."""
    file_label = "📮 File to GitHub" if target == "github" else "📮 File to Plane"
    elements = [
        {
            "type": "button",
            "style": "primary",
            "text": {"type": "plain_text", "text": file_label, "emoji": True},
            "action_id": f"file_{target}",
            "value": token,
        }
    ]
    # Offer the other target too when both are viable.
    other = "plane" if target == "github" else "github"
    if other == "plane":
        elements.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "File to Plane", "emoji": True},
            "action_id": "file_plane",
            "value": token,
        })
    elements.append({
        "type": "button",
        "text": {"type": "plain_text", "text": "Dismiss", "emoji": True},
        "action_id": "dismiss",
        "value": token,
        "style": "danger",
    })
    return {"type": "actions", "block_id": f"triage_actions::{repo_url}", "elements": elements}


def filed_blocks(title: str, target: str, issue_id: str, url: str | None) -> list[dict]:
    """Replace the card's action row after a successful file."""
    where = "GitHub" if target == "github" else "Plane"
    link = f"<{url}|#{issue_id}>" if url else f"#{issue_id}"
    return [{
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"✅ Filed to {where} as {link} — _{_trunc(title, 120)}_"}],
    }]


def error_blocks(message: str) -> list[dict]:
    return [{
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"⚠️ {_trunc(message, 500)}"}],
    }]


def summary_line(resp: TriageResponse) -> str:
    """Short plaintext fallback / notification text for the card."""
    t = resp.triage
    return f"{_PRIORITY_EMOJI.get(t.priority, '')} [{t.priority}] {t.title}"

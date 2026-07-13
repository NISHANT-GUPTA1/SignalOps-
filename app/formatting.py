"""Shared rendering helpers used by both the web API and the Slack agent.

`issue_body` turns a structured triage into the Markdown body we file into
GitHub / Plane. Kept in one place so the FastAPI surface (app/main.py) and the
Slack surface (app/slack_agent.py) produce identical issues.
"""
from __future__ import annotations

from .models import TriagedIssue


def issue_body(t: TriagedIssue) -> str:
    """GitHub/Plane-ready Markdown body for a triaged issue."""
    lines = [t.summary, ""]
    if t.reproduction_steps:
        lines.append("### Steps to reproduce")
        lines += [f"{i}. {s}" for i, s in enumerate(t.reproduction_steps, 1)]
        lines.append("")
    if t.expected_behavior:
        lines += ["### Expected", t.expected_behavior, ""]
    if t.actual_behavior:
        lines += ["### Actual", t.actual_behavior, ""]
    lines += [
        "### Triage",
        f"- **Type:** {t.type}",
        f"- **Priority:** {t.priority} — {t.priority_reason}",
        f"- **Severity:** {t.severity}",
        f"- **Components:** {', '.join(t.components) or 'n/a'}",
    ]
    if t.duplicate_candidates:
        lines += ["", "### Possible duplicates"]
        for d in t.duplicate_candidates:
            lines.append(f"- `{d.id}` {d.title} — {d.confidence} ({d.reason})")
    if t.suggested_next_action:
        lines += ["", "### Suggested next action", t.suggested_next_action]
    lines += ["", "_Filed by AI Bug Triage & Release Operator._"]
    return "\n".join(lines)

"""Plane engineering-board integration (real).

Creates real issues on a Plane project via the Plane API. Requires
PLANE_API_KEY + PLANE_WORKSPACE_SLUG + PLANE_PROJECT_ID. No mock — if Plane
isn't configured, callers should target GitHub instead (the API surfaces a
clear error).
"""
from __future__ import annotations

import requests

from ..config import settings

# Plane priority enum: urgent | high | medium | low | none
_PRIORITY_MAP = {"P0": "urgent", "P1": "high", "P2": "medium", "P3": "low", "critical": "urgent"}


def is_live() -> bool:
    return settings.plane_is_live


def create_issue(title: str, description: str, priority: str = "medium", labels=None) -> dict:
    if not is_live():
        raise PermissionError(
            "Plane is not configured. Set PLANE_API_KEY, PLANE_WORKSPACE_SLUG and "
            "PLANE_PROJECT_ID, or create the issue in GitHub instead."
        )
    slug = settings.plane_workspace_slug
    proj = settings.plane_project_id
    # Plane v1 API: work items live at /work-items/ (the older /issues/ path 404s).
    import html

    safe = html.escape(description)
    base = settings.plane_base_url  # self-hosted host or Plane Cloud (api.plane.so)
    resp = requests.post(
        f"{base}/api/v1/workspaces/{slug}/projects/{proj}/work-items/",
        headers={"X-API-Key": settings.plane_api_key, "Content-Type": "application/json"},
        json={
            "name": title[:255],
            "description_html": f"<p>{safe.replace(chr(10), '<br>')}</p>",
            "priority": _PRIORITY_MAP.get(priority, "medium"),
        },
        timeout=30,
    )
    resp.raise_for_status()
    issue_id = resp.json().get("id", "")
    # Web link: Plane Cloud serves the UI at app.plane.so; a self-hosted instance
    # serves both API and UI from the same host.
    web_base = "https://app.plane.so" if base == "https://api.plane.so" else base
    return {
        "id": str(issue_id),
        "url": f"{web_base}/{slug}/projects/{proj}/issues/{issue_id}/",
    }

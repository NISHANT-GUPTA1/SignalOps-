"""Real GitHub integration (free). Reads issues/PRs; creates issues with a PAT.

Public repos work with no token (60 req/hr). A free PAT gives 5,000 req/hr and
write access (create issues).
"""
from __future__ import annotations

import re
from typing import Optional

import requests

from ..config import settings

API = "https://api.github.com"


def parse_repo(url_or_slug: str) -> tuple[str, str]:
    """Accept 'owner/repo' or a full GitHub URL -> (owner, repo)."""
    s = url_or_slug.strip().rstrip("/")
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        return m.group(1), m.group(2)
    parts = s.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Could not parse repo from: {url_or_slug!r}")


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        h["Authorization"] = f"Bearer {settings.github_token}"
    return h


def list_issues(owner: str, repo: str, state: str = "all", limit: int = 50) -> list[dict]:
    """List issues (excluding PRs). Returns simplified dicts."""
    out: list[dict] = []
    page = 1
    while len(out) < limit:
        resp = requests.get(
            f"{API}/repos/{owner}/{repo}/issues",
            headers=_headers(),
            params={"state": state, "per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:  # skip PRs
                continue
            out.append(
                {
                    "number": it["number"],
                    "title": it["title"],
                    "body": it.get("body") or "",
                    "state": it["state"],
                    "labels": [l["name"] for l in it.get("labels", [])],
                    "url": it["html_url"],
                }
            )
            if len(out) >= limit:
                break
        page += 1
    return out


def get_default_branch(owner: str, repo: str) -> str:
    resp = requests.get(f"{API}/repos/{owner}/{repo}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json().get("default_branch", "main")


def list_markdown_files(owner: str, repo: str, branch: Optional[str] = None, limit: int = 25) -> list[str]:
    """Return paths of markdown docs in the repo (README + docs/**)."""
    branch = branch or get_default_branch(owner, repo)
    resp = requests.get(
        f"{API}/repos/{owner}/{repo}/git/trees/{branch}",
        headers=_headers(),
        params={"recursive": "1"},
        timeout=30,
    )
    resp.raise_for_status()
    tree = resp.json().get("tree", [])
    paths = [
        t["path"]
        for t in tree
        if t.get("type") == "blob" and t["path"].lower().endswith((".md", ".mdx"))
    ]
    # prioritise top-level README and docs/ folder
    paths.sort(key=lambda p: (0 if "readme" in p.lower() else 1, 0 if p.lower().startswith("docs/") else 1, p))
    return paths[:limit]


def get_file_text(owner: str, repo: str, path: str, branch: Optional[str] = None) -> str:
    import base64

    resp = requests.get(
        f"{API}/repos/{owner}/{repo}/contents/{path}",
        headers=_headers(),
        params={"ref": branch} if branch else None,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def file_html_url(owner: str, repo: str, path: str, branch: str) -> str:
    return f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"


def create_issue(
    owner: str, repo: str, title: str, body: str, labels: Optional[list[str]] = None
) -> dict:
    if not settings.github_token:
        raise PermissionError(
            "Creating a GitHub issue requires GITHUB_TOKEN (free PAT with Issues:write)."
        )
    resp = requests.post(
        f"{API}/repos/{owner}/{repo}/issues",
        headers=_headers(),
        json={"title": title, "body": body, "labels": labels or []},
        timeout=30,
    )
    resp.raise_for_status()
    it = resp.json()
    return {"id": str(it["number"]), "url": it["html_url"]}

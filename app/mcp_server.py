"""MCP server exposing the Bug-Triage engine as Model Context Protocol tools.

This is the same agentic-RAG brain that powers the Slack agent, published over
MCP so *any* MCP client (Slack's agent runtime, Claude Desktop, an IDE, another
agent) can call it. It satisfies the challenge's "MCP server integration" pillar
and cleanly decouples the engine from the Slack shell.

Run it (stdio transport):

    python -m app.mcp_server

Tools
-----
* triage_report(text, source?, reporter?, repo?)  -> structured, prioritized,
      de-duplicated triage (title, priority P0–P3 + reason, severity, components,
      repro steps, duplicate candidates) grounded in your real issue history.
* search_issues(query, k?)                         -> hybrid (dense+BM25/RRF)
      retrieval over ingested issues & docs.
* file_issue(...)                                  -> create the issue in GitHub
      or Plane from a prior triage.
* release_notes(repo)                              -> draft release notes from
      resolved (closed) issues.

Requires the `mcp` package:  pip install "mcp[cli]>=1.2"
"""
from __future__ import annotations

from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "The MCP SDK is not installed. Run:  pip install \"mcp[cli]>=1.2\"\n"
        f"(import error: {exc})"
    )

from . import agent
from .config import settings
from .formatting import issue_body
from .integrations import github_client, plane_client
from .vectorstore import get_store

mcp = FastMCP("ai-bug-triage")


@mcp.tool()
def triage_report(
    text: str,
    source: str = "other",
    reporter: Optional[str] = None,
) -> dict:
    """Triage a raw, messy bug report into a clean, prioritized, de-duplicated
    engineering issue grounded in the team's existing issue history.

    Args:
        text: The raw report / feedback text.
        source: Where it came from (slack | email | github | form | other).
        reporter: Who reported it, if known.

    Returns a dict with the structured triage plus the retrieval trace.
    """
    resp = agent.triage(text, source=source, reporter=reporter)
    t = resp.triage
    return {
        "triage": t.model_dump(),
        "markdown": issue_body(t),
        "retrieved": [c.model_dump() for c in resp.retrieved],
        "agent_searches": resp.agent_searches,
    }


@mcp.tool()
def search_issues(query: str, k: int = 5) -> list[dict]:
    """Hybrid semantic + keyword (dense HNSW + BM25, fused by RRF) search over
    ingested issues and product docs. Use to find duplicates or context."""
    hits = get_store().search(query, k=k)
    return [
        {
            "id": h.doc.id,
            "title": h.doc.title,
            "source": h.doc.source,
            "url": h.doc.url,
            "score": h.score,
            "snippet": h.doc.text[:280].replace("\n", " "),
        }
        for h in hits
    ]


@mcp.tool()
def file_issue(
    title: str,
    body_markdown: str,
    target: str = "github",
    repo: str = "",
    labels: Optional[list[str]] = None,
    priority: str = "P2",
) -> dict:
    """Create an issue in GitHub or Plane. Typically called after triage_report,
    passing its `triage.title`, `markdown`, `triage.labels`, `triage.priority`."""
    labels = labels or []
    if target == "github":
        repo = repo or settings.default_repo_url
        if not repo:
            raise ValueError("A repo (owner/name) is required to file to GitHub.")
        owner, name = github_client.parse_repo(repo)
        res = github_client.create_issue(owner, name, title, body_markdown, labels)
        return {"ok": True, "target": "github", "id": res["id"], "url": res["url"]}
    res = plane_client.create_issue(title, body_markdown, priority=priority, labels=labels)
    return {"ok": True, "target": "plane", "id": res["id"], "url": res.get("url")}


@mcp.tool()
def release_notes(repo: str = "") -> dict:
    """Draft release notes (Markdown) from resolved (closed) issues for a repo."""
    repo = repo or settings.default_repo_url
    if not repo:
        raise ValueError("A repo (owner/name) is required.")
    owner, name = github_client.parse_repo(repo)
    docs = get_store().closed_issues(repo=f"{owner}/{name}")
    seen, items = set(), []
    for d in docs:
        num = d.metadata.get("number", d.id)
        if num in seen:
            continue
        seen.add(num)
        items.append({"number": num, "title": d.title,
                      "labels": d.metadata.get("labels", []), "url": d.url})
        if len(items) >= 30:
            break
    if not items:
        return {"issue_count": 0, "markdown": "_No resolved issues found; ingest the repo first._"}
    return {"issue_count": len(items), "markdown": agent.generate_release_notes(items)}


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()

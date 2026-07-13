"""FastAPI app: AI Bug Triage & Release Operator backend + static UI.

Genuine end-to-end: real GitHub ingestion (issues + repo markdown docs, chunked),
real Qdrant vector DB, real embeddings, agentic-RAG triage, real issue creation.
No seed data — the knowledge base is whatever you ingest.
"""
from __future__ import annotations

import os
import threading
import uuid

from fastapi import Cookie, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, auth
from .chunking import chunk_text
from .config import settings
from .formatting import issue_body as _issue_body
from .integrations import github_client, plane_client, slack_client
from .llm import describe as describe_llm
from .llm import get_active_provider
from .models import (
    CreateIssueRequest,
    CreateIssueResponse,
    IngestRequest,
    IngestResponse,
    ReleaseNotesRequest,
    ReleaseNotesResponse,
    TriageRequest,
    TriageResponse,
    TriagedIssue,
)
from .vectorstore import Doc, get_store

app = FastAPI(title="AI Bug Triage & Release Operator")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
def _startup() -> None:
    s = get_store()  # opens Qdrant, probes embedding dim, loads existing corpus
    print(f"[startup] vector store ready: {s.stats()} | embeddings={settings.effective_embedding_provider}")


@app.on_event("shutdown")
def _shutdown() -> None:
    get_store().close()


# ---------------- API ----------------
@app.get("/api/status")
def status() -> dict:
    return {
        "llm": describe_llm(),  # active provider + per-tier availability (fallback chain)
        "github_write": settings.github_can_write,
        "slack": settings.has_slack,
        "plane_live": settings.plane_is_live,
        "embedding_provider": settings.effective_embedding_provider,
        "embeddings_real": settings.embeddings_are_real,
        "vector_db": f"qdrant ({settings.qdrant_mode})",
        "store": get_store().stats(),
    }


@app.get("/api/slack/channels")
def slack_channels() -> dict:
    if not slack_client.is_configured():
        raise HTTPException(400, "Slack is not configured. Set SLACK_BOT_TOKEN.")
    try:
        return {"channels": slack_client.list_channels()}
    except Exception as exc:
        raise HTTPException(400, f"Slack error: {exc}")


@app.get("/api/slack/messages")
def slack_messages(channel_id: str, limit: int = 20) -> dict:
    if not slack_client.is_configured():
        raise HTTPException(400, "Slack is not configured. Set SLACK_BOT_TOKEN.")
    try:
        return {"messages": slack_client.fetch_messages(channel_id, limit=limit)}
    except Exception as exc:
        raise HTTPException(400, f"Slack error: {exc}")


@app.post("/api/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    if get_active_provider() is None:
        raise HTTPException(
            400,
            "No LLM provider available. Set ANTHROPIC_API_KEY, or OPENAI_COMPAT_* vars, "
            "or run a local Ollama model.",
        )
    try:
        return agent.triage(req.raw_text, source=req.source, reporter=req.reporter)
    except Exception as exc:
        raise HTTPException(500, f"Triage failed: {exc}")


@app.post("/api/create-issue", response_model=CreateIssueResponse)
def create_issue(req: CreateIssueRequest) -> CreateIssueResponse:
    t = req.triage
    body = _issue_body(t)
    try:
        if req.target == "github":
            if not req.repo_url:
                raise HTTPException(400, "repo_url is required to create a GitHub issue.")
            owner, repo = github_client.parse_repo(req.repo_url)
            res = github_client.create_issue(owner, repo, t.title, body, t.labels)
            return CreateIssueResponse(
                ok=True, target="github", id=res["id"], url=res["url"],
                message=f"Created GitHub issue #{res['id']}",
            )
        res = plane_client.create_issue(t.title, body, priority=t.priority, labels=t.labels)
        return CreateIssueResponse(
            ok=True, target="plane", id=res["id"], url=res["url"], message="Created Plane issue",
        )
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Create issue failed: {exc}")


# ---------------- Background ingestion ----------------
# Ingest fetches + embeds many chunks, which can take minutes on a CPU-only
# host. A single synchronous HTTP request would blow past reverse-proxy /
# Cloudflare timeouts (HTTP 524). So we run ingestion in a background thread,
# return a job id immediately, and let the UI poll /api/ingest/status/{id}.
_ingest_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _ingest_lock:
        _ingest_jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with _ingest_lock:
        j = _ingest_jobs.get(job_id)
        return dict(j) if j else None


def _github_error_message(exc: Exception, repo_url: str) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 404:
        return (
            f"GitHub returned 404 for '{repo_url}'. This almost always means the repo is "
            "PRIVATE and your GITHUB_TOKEN can't access it (GitHub hides private repos as 404). "
            "Check: (1) the owner/repo path is exact; (2) if it's an organization repo, the "
            "fine-grained token's Resource owner is that ORG with Contents: Read + Issues: "
            "Read/write and the org approved the token; or use a classic token with 'repo' scope."
        )
    if status_code in (401, 403):
        return f"GitHub auth error ({status_code}) — your GITHUB_TOKEN is invalid or lacks scope."
    return f"GitHub ingest failed: {exc}"


def _run_ingest_job(job_id: str, repo_url: str, state: str, limit: int) -> None:
    store = get_store()
    try:
        owner, repo = github_client.parse_repo(repo_url)
        slug = f"{owner}/{repo}"
        docs: list[Doc] = []

        # 1) Issues (each issue body chunked if long; parent linkage in metadata).
        issues = github_client.list_issues(owner, repo, state=state, limit=limit)
        for it in issues:
            chunks = chunk_text(it["body"]) or [""]
            for ci, chunk in enumerate(chunks):
                suffix = "" if len(chunks) == 1 else f"::chunk{ci}"
                docs.append(
                    Doc(
                        id=f"{slug}#{it['number']}{suffix}",
                        title=it["title"],
                        text=chunk,
                        source="github_issue",
                        url=it["url"],
                        metadata={
                            "state": it["state"], "repo": slug, "number": it["number"],
                            "labels": it["labels"], "chunk": ci,
                        },
                    )
                )

        # 2) Repo markdown docs (README, docs/**) — real document loading + chunking.
        branch = github_client.get_default_branch(owner, repo)
        for path in github_client.list_markdown_files(owner, repo, branch=branch, limit=20):
            try:
                text = github_client.get_file_text(owner, repo, path, branch=branch)
            except Exception:
                continue
            for ci, chunk in enumerate(chunk_text(text)):
                docs.append(
                    Doc(
                        id=f"{slug}:{path}::chunk{ci}",
                        title=f"{path}",
                        text=chunk,
                        source="doc",
                        url=github_client.file_html_url(owner, repo, path, branch),
                        metadata={"repo": slug, "path": path, "chunk": ci},
                    )
                )
    except Exception as exc:
        _set_job(job_id, status="error", error=_github_error_message(exc, repo_url))
        return

    # Embed in small slices so progress is visible and one call never runs too long.
    total = len(docs)
    _set_job(job_id, status="embedding", total_chunks=total, embedded=0, issue_count=len(issues))
    try:
        added = 0
        BATCH = 20
        for i in range(0, total, BATCH):
            added += store.add(docs[i:i + BATCH])
            _set_job(job_id, embedded=min(i + BATCH, total))
    except Exception as exc:
        _set_job(job_id, status="error", error=f"Embedding failed during ingest: {exc}")
        return

    _set_job(
        job_id,
        status="done",
        ingested=added,
        total_in_store=store.stats()["total"],
        message=f"Ingested {added} new chunks from {slug} ({len(issues)} issues + repo docs).",
    )


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    """Start a background ingestion job; returns a job id to poll. Returns fast
    so reverse-proxy / Cloudflare timeouts (HTTP 524) never trigger."""
    job_id = uuid.uuid4().hex
    _set_job(job_id, status="starting", repo=req.repo_url)
    threading.Thread(
        target=_run_ingest_job,
        args=(job_id, req.repo_url, req.state, req.limit),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "started"}


@app.get("/api/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown ingest job id.")
    return job


@app.post("/api/release-notes", response_model=ReleaseNotesResponse)
def release_notes(req: ReleaseNotesRequest) -> ReleaseNotesResponse:
    if get_active_provider() is None:
        raise HTTPException(400, "No LLM provider available for release notes.")
    store = get_store()
    owner, repo = github_client.parse_repo(req.repo_url)
    docs = store.closed_issues(repo=f"{owner}/{repo}")
    # de-dup by issue number (chunking can produce multiple docs per issue)
    seen, items = set(), []
    for d in docs:
        num = d.metadata.get("number", d.id)
        if num in seen:
            continue
        seen.add(num)
        items.append({"number": num, "title": d.title, "labels": d.metadata.get("labels", []), "url": d.url})
        if len(items) >= req.limit:
            break
    if not items:
        return ReleaseNotesResponse(
            markdown="_No resolved (closed) issues found for this repo. Ingest it first "
            "with state=all or state=closed._",
            issue_count=0,
        )
    md = agent.generate_release_notes(items)
    return ReleaseNotesResponse(markdown=md, issue_count=len(items))


# ---------------- Auth (simple, DB-free demo gate) ----------------
class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    identifier: str  # username OR email
    password: str


@app.post("/api/auth/register")
def auth_register(req: RegisterRequest) -> dict:
    try:
        auth.register(req.email, req.username, req.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.post("/api/auth/login")
def auth_login(req: LoginRequest, response: Response) -> dict:
    username = auth.verify(req.identifier, req.password)
    if not username:
        raise HTTPException(401, "Invalid username/email or password.")
    token = auth.create_session(username)
    response.set_cookie(
        auth.COOKIE_NAME, token, httponly=True, samesite="lax",
        max_age=60 * 60 * 24 * 7, path="/",
    )
    return {"ok": True, "username": username}


@app.post("/api/auth/logout")
def auth_logout(response: Response, bt_session: str | None = Cookie(default=None)) -> dict:
    auth.destroy_session(bt_session)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(bt_session: str | None = Cookie(default=None)) -> dict:
    username = auth.username_for_token(bt_session)
    return {"authenticated": bool(username), "username": username}


# ---------------- Static UI ----------------
@app.get("/")
def landing() -> FileResponse:
    """Marketing landing page (animated particle hero)."""
    return FileResponse(os.path.join(_STATIC_DIR, "landing.html"))


@app.get("/login")
def login_page() -> FileResponse:
    """Sign in / sign up page."""
    return FileResponse(os.path.join(_STATIC_DIR, "login.html"))


@app.get("/app")
def index(bt_session: str | None = Cookie(default=None)):
    """The operator tool — gated behind a login."""
    if not auth.username_for_token(bt_session):
        return RedirectResponse(url="/login?next=/app", status_code=302)
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

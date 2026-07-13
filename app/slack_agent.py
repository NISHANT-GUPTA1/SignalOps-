"""The Slack-native AI Bug Triage agent (Slack Bolt, Socket Mode).

Slack is the front door: users triage a messy report without ever leaving Slack.

Entry points
------------
* `/triage <text>`            slash command — triage pasted text
* "⚡ Triage this" shortcut    message shortcut — triage any message in place
* `@BugTriage <text>`         app mention — triage in a channel thread
* AI Assistant panel          native Slack agent — chat, and just paste a report
* `/release-notes [repo]`     draft release notes from resolved issues

Every triage posts an interactive Block Kit card; the **File to GitHub / Plane**
buttons run the same issue-creation path as the web API. The heavy lifting is the
existing agentic-RAG engine in app/agent.py — this module is only the Slack shell.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from slack_bolt import App

from . import agent
from .config import settings
from .formatting import issue_body
from .integrations import github_client, plane_client
from .llm import describe as describe_llm
from .llm import get_active_provider
from .models import TriageResponse
from . import slack_blocks as B

# ── pending-triage cache ──────────────────────────────────────────────────────
# Button payloads can only carry ~2 KB, so we stash the full TriageResponse in
# process and pass a short token in the button `value`. Single-process Socket
# Mode makes an in-memory dict the right tool; entries expire after 1 hour.
_PENDING: dict[str, dict] = {}
_TTL = 3600


def _remember(resp: TriageResponse, repo_url: str) -> str:
    _sweep()
    token = f"t{int(time.time()*1000)%10_000_000}{len(_PENDING)}"
    _PENDING[token] = {"resp": resp, "repo_url": repo_url, "at": time.time()}
    return token


def _recall(token: str) -> Optional[dict]:
    _sweep()
    return _PENDING.get(token)


def _sweep() -> None:
    now = time.time()
    for k in [k for k, v in _PENDING.items() if now - v["at"] > _TTL]:
        _PENDING.pop(k, None)


# ── helpers ───────────────────────────────────────────────────────────────────
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _clean(text: str) -> str:
    return _MENTION_RE.sub("", text or "").strip()


def _resolve_repo(text: str) -> tuple[str, str]:
    """Pull an optional trailing `repo:owner/name` (or a github URL) out of the
    command text; fall back to the configured default repo. Returns (text, repo)."""
    repo = settings.default_repo_url
    m = re.search(r"\brepo:(\S+)", text)
    if m:
        repo = m.group(1)
        text = (text[: m.start()] + text[m.end():]).strip()
    else:
        m = re.search(r"(https?://github\.com/\S+|\b[\w.-]+/[\w.-]+)$", text.strip())
        # only treat a trailing owner/repo as a repo if it really looks like one
        if m and "/" in m.group(1) and " " not in m.group(1) and len(text.split()) > 1:
            candidate = m.group(1)
            if re.match(r"^[\w.-]+/[\w.-]+$", candidate) or "github.com" in candidate:
                repo = candidate
                text = text[: m.start()].strip()
    return text, repo


def _target_for(repo_url: str) -> str:
    if repo_url and settings.github_token:
        return "github"
    if settings.plane_is_live:
        return "plane"
    return "github"


def _llm_guard() -> Optional[str]:
    if get_active_provider() is None:
        return (
            "No LLM provider is available. Set `ANTHROPIC_API_KEY`, the `OPENAI_COMPAT_*` "
            "vars, or run a local Ollama model, then restart the agent."
        )
    return None


def _run_triage_card(text: str, source: str, reporter: Optional[str]) -> tuple[list[dict], str]:
    """Triage `text` and return (blocks, notification_text)."""
    text, repo_url = _resolve_repo(text)
    resp = agent.triage(text, source=source, reporter=reporter)
    target = _target_for(repo_url)
    token = _remember(resp, repo_url)
    return B.triage_blocks(resp, token, target, repo_url), B.summary_line(resp)


# ── app factory ───────────────────────────────────────────────────────────────
def build_app() -> App:
    app = App(token=settings.slack_bot_token)
    _register_commands(app)
    _register_events(app)
    _register_actions(app)
    _register_assistant(app)
    return app


# ── slash commands ────────────────────────────────────────────────────────────
def _register_commands(app: App) -> None:
    @app.command("/triage")
    def triage_command(ack, respond, command, logger):
        ack()
        text = _clean(command.get("text", ""))
        if not text:
            respond(":point_right: Usage: `/triage <paste the bug report>` "
                    "(optionally end with `repo:owner/name`).")
            return
        err = _llm_guard()
        if err:
            respond(err)
            return
        respond({"response_type": "ephemeral", "text": ":hourglass_flowing_sand: Triaging…"})
        try:
            blocks, summary = _run_triage_card(text, source="slack", reporter=command.get("user_name"))
            respond({"response_type": "in_channel", "text": summary, "blocks": blocks})
        except Exception as exc:  # noqa: BLE001
            logger.exception("triage command failed")
            respond({"response_type": "ephemeral", "text": f":warning: Triage failed: {exc}"})

    @app.command("/release-notes")
    def release_notes_command(ack, respond, command, logger):
        ack()
        text = _clean(command.get("text", "")).strip()
        repo = text or settings.default_repo_url
        if not repo:
            respond(":point_right: Usage: `/release-notes owner/name` "
                    "(or set `GITHUB_REPO_URL`).")
            return
        if _llm_guard():
            respond(_llm_guard())
            return
        respond({"response_type": "ephemeral", "text": ":memo: Drafting release notes…"})
        try:
            md = _release_notes_markdown(repo)
            respond({"response_type": "in_channel", "text": "Release notes",
                     "blocks": [{"type": "section",
                                 "text": {"type": "mrkdwn", "text": B._trunc(md, 2900)}}]})
        except Exception as exc:  # noqa: BLE001
            logger.exception("release notes failed")
            respond({"response_type": "ephemeral", "text": f":warning: {exc}"})


def _release_notes_markdown(repo_url: str) -> str:
    from .vectorstore import get_store

    owner, repo = github_client.parse_repo(repo_url)
    docs = get_store().closed_issues(repo=f"{owner}/{repo}")
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
        return ("_No resolved (closed) issues found for this repo. Ingest it first "
                "in the web console with state=all or state=closed._")
    return agent.generate_release_notes(items)


# ── events: @mention + message shortcut ───────────────────────────────────────
def _register_events(app: App) -> None:
    @app.event("app_mention")
    def on_mention(event, client, say, logger):
        text = _clean(event.get("text", ""))
        thread_ts = event.get("thread_ts") or event.get("ts")
        channel = event["channel"]
        if not text:
            say(text="Paste a bug report after mentioning me and I'll triage it. "
                     "Example: `@BugTriage login button 500s on mobile`.", thread_ts=thread_ts)
            return
        err = _llm_guard()
        if err:
            say(text=err, thread_ts=thread_ts)
            return
        try:
            client.reactions_add(channel=channel, timestamp=event["ts"], name="hourglass_flowing_sand")
        except Exception:  # noqa: BLE001
            pass
        try:
            blocks, summary = _run_triage_card(text, source="slack", reporter=event.get("user"))
            say(text=summary, blocks=blocks, thread_ts=thread_ts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("mention triage failed")
            say(text=f":warning: Triage failed: {exc}", thread_ts=thread_ts)

    @app.shortcut("triage_message")
    def on_message_shortcut(ack, shortcut, client, logger):
        ack()
        msg = shortcut.get("message", {})
        text = _clean(msg.get("text", ""))
        channel = shortcut["channel"]["id"]
        thread_ts = msg.get("thread_ts") or msg.get("ts")
        if not text:
            client.chat_postEphemeral(channel=channel, user=shortcut["user"]["id"],
                                      text="That message has no text to triage.")
            return
        err = _llm_guard()
        if err:
            client.chat_postEphemeral(channel=channel, user=shortcut["user"]["id"], text=err)
            return
        try:
            blocks, summary = _run_triage_card(text, source="slack", reporter=msg.get("user"))
            client.chat_postMessage(channel=channel, text=summary, blocks=blocks, thread_ts=thread_ts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("shortcut triage failed")
            client.chat_postEphemeral(channel=channel, user=shortcut["user"]["id"],
                                      text=f":warning: Triage failed: {exc}")


# ── interactive buttons: file / dismiss ───────────────────────────────────────
def _register_actions(app: App) -> None:
    @app.action("file_github")
    def file_github(ack, body, client, logger):
        ack()
        _handle_file(body, client, logger, target="github")

    @app.action("file_plane")
    def file_plane(ack, body, client, logger):
        ack()
        _handle_file(body, client, logger, target="plane")

    @app.action("dismiss")
    def dismiss(ack, body, client):
        ack()
        _PENDING.pop(_button_value(body), None)
        _replace_actions(body, client, B.error_blocks("Dismissed — not filed."))


def _button_value(body: dict) -> str:
    return body["actions"][0]["value"]


def _handle_file(body: dict, client, logger, target: str) -> None:
    token = _button_value(body)
    entry = _recall(token)
    channel = body["channel"]["id"]
    user = body["user"]["id"]
    if not entry:
        client.chat_postEphemeral(channel=channel, user=user,
                                  text="This triage expired — re-run `/triage` to file it.")
        return
    resp: TriageResponse = entry["resp"]
    repo_url = entry["repo_url"]
    t = resp.triage
    body_md = issue_body(t)
    try:
        if target == "github":
            if not repo_url:
                client.chat_postEphemeral(channel=channel, user=user,
                    text="No repo configured. Re-run with `repo:owner/name`, or set `GITHUB_REPO_URL`.")
                return
            owner, repo = github_client.parse_repo(repo_url)
            res = github_client.create_issue(owner, repo, t.title, body_md, t.labels)
        else:
            res = plane_client.create_issue(t.title, body_md, priority=t.priority, labels=t.labels)
        _PENDING.pop(token, None)
        _replace_actions(body, client, B.filed_blocks(t.title, target, res["id"], res.get("url")))
    except PermissionError as exc:
        client.chat_postEphemeral(channel=channel, user=user, text=f":warning: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("file issue failed")
        client.chat_postEphemeral(channel=channel, user=user, text=f":warning: Filing failed: {exc}")


def _replace_actions(body: dict, client, new_tail: list[dict]) -> None:
    """Swap the trailing actions block for a result/context block, in place."""
    msg = body.get("message", {})
    blocks = [b for b in msg.get("blocks", []) if b.get("type") != "actions"]
    blocks += new_tail
    client.chat_update(channel=body["channel"]["id"], ts=msg["ts"],
                       text=msg.get("text", "Triage"), blocks=blocks)


# ── native AI Assistant panel (Slack AI) ──────────────────────────────────────
def _register_assistant(app: App) -> None:
    """Register the Agents & AI Apps assistant. Optional: only wired up if this
    Bolt build ships the Assistant middleware and the app has the feature on."""
    try:
        from slack_bolt import Assistant
    except Exception:  # noqa: BLE001
        return

    assistant = Assistant()

    @assistant.thread_started
    def start_thread(say, set_suggested_prompts):
        say("Hi! I'm your AI Bug Triage agent. Paste any messy bug report and I'll "
            "return a clean, prioritized, de-duplicated issue — then file it to "
            "GitHub or Plane for you.")
        try:
            set_suggested_prompts(prompts=[
                {"title": "Triage a report", "message": "Login button 500s on mobile Safari after the last deploy"},
                {"title": "What can you do?", "message": "What can you do?"},
            ])
        except Exception:  # noqa: BLE001
            pass

    @assistant.user_message
    def respond_in_thread(payload, set_status, say, client, context, logger):
        text = _clean(payload.get("text", ""))
        low = text.lower()
        if low in {"what can you do?", "help", "what can you do"}:
            say("Paste a bug report (Slack/email/form text) and I'll triage it: title, "
                "priority (P0–P3) with a reason, severity, components, repro steps, and "
                "duplicate detection grounded in your real issue history. I'll post a card "
                "with a *File to GitHub/Plane* button. You can also use `/triage`, the "
                "*Triage this* message shortcut, or `@`-mention me in a channel.")
            return
        err = _llm_guard()
        if err:
            say(err)
            return
        try:
            set_status("triaging…")
        except Exception:  # noqa: BLE001
            pass
        try:
            blocks, summary = _run_triage_card(text, source="slack", reporter=None)
            say(text=summary, blocks=blocks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("assistant triage failed")
            say(f":warning: Triage failed: {exc}")

    app.use(assistant)


# ── convenience for the entrypoint ────────────────────────────────────────────
def startup_banner() -> str:
    llm = describe_llm()
    target = settings.default_target
    repo = settings.default_repo_url or "(none — pass repo:owner/name)"
    model = llm.get("model") or ""
    active = f"{llm.get('active_provider', 'none')}" + (f" / {model}" if model else "")
    return (
        "AI Bug Triage — Slack agent starting\n"
        f"  LLM       : {active}\n"
        f"  file to   : {target}  (repo: {repo})\n"
        f"  embeddings: {settings.effective_embedding_provider}\n"
        f"  vector db : qdrant ({settings.qdrant_mode})"
    )

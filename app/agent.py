"""The triage agent: an agentic-RAG tool loop running on whichever LLM provider
is active (Anthropic -> OpenAI-compatible -> local Ollama).

Tools exposed to the model:
  * search_existing_issues(query, k) -> hybrid (dense+BM25/RRF) retrieval. The
    model can call this several times (query expansion) to hunt duplicates/context.
  * submit_triage(...)               -> structured output; ends the loop. On the
    final iteration it is forced, so even weaker local models emit valid output.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import get_active_provider, invalidate
from .llm.base import ProviderError, ToolDef, tool_result_msg, user_msg
from .models import RetrievedContext, TriageResponse, TriagedIssue
from .vectorstore import get_store

MAX_ITERS = 6

SYSTEM_PROMPT = """You are an AI Bug Triage & Release Operator for a software engineering team.
You turn messy, raw feedback (from Slack, email, forms, GitHub) into a clean, prioritized, \
de-duplicated engineering issue.

Workflow:
1. Use the `search_existing_issues` tool to look for DUPLICATES and relevant context. \
Search more than once if useful — try the user's words, then rephrase with likely component \
or error terms (query expansion). Only cite a duplicate whose id actually appeared in a search result.
2. Then call `submit_triage` EXACTLY ONCE with your final structured triage.

Priority rubric:
- P0: outage / data loss / security / billing-incorrect-charges — affects many users, no workaround.
- P1: major feature broken, common path, workaround is painful.
- P2: real bug but limited scope or easy workaround.
- P3: minor / cosmetic / nice-to-have / feature request.

Be specific in reproduction_steps. If the report lacks detail, infer the most likely steps and \
say so in suggested_next_action. Keep the title short and searchable.
Always finish by calling submit_triage."""

SEARCH_TOOL = ToolDef(
    name="search_existing_issues",
    description="Hybrid semantic + keyword search over existing issues and product docs. "
    "Use to find duplicate issues and grounding context before triaging.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "k": {"type": "integer", "description": "Number of results (default 5)."},
        },
        "required": ["query"],
    },
)

SUBMIT_TOOL = ToolDef(
    name="submit_triage",
    description="Submit the final structured triage for this report. Call exactly once.",
    strict=True,
    parameters={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "type": {"type": "string", "enum": ["bug", "feature_request", "question", "task"]},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "priority_reason": {"type": "string"},
            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
            "components": {"type": "array", "items": {"type": "string"}},
            "labels": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "reproduction_steps": {"type": "array", "items": {"type": "string"}},
            "expected_behavior": {"type": "string"},
            "actual_behavior": {"type": "string"},
            "duplicate_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "title", "confidence", "reason"],
                },
            },
            "suggested_next_action": {"type": "string"},
        },
        "required": [
            "title", "type", "priority", "priority_reason", "severity", "components",
            "labels", "summary", "reproduction_steps", "expected_behavior",
            "actual_behavior", "duplicate_candidates", "suggested_next_action",
        ],
    },
)


def _run_search(query: str, k: int = 5) -> tuple[str, list[RetrievedContext]]:
    hits = get_store().search(query, k=k)
    contexts: list[RetrievedContext] = []
    lines = []
    for h in hits:
        snippet = h.doc.text[:240].replace("\n", " ")
        contexts.append(
            RetrievedContext(
                id=h.doc.id, title=h.doc.title, snippet=snippet,
                score=h.score, source=h.doc.source, url=h.doc.url,
            )
        )
        lines.append(f"- id={h.doc.id} | {h.doc.title} ({h.doc.source}) | score={h.score}\n    {snippet}")
    return ("\n".join(lines) if lines else "(no results)"), contexts


SYSTEM_PROMPT_SINGLE = """You are an AI Bug Triage & Release Operator for a software engineering team.
You are given a raw, messy report and a list of EXISTING issues & docs retrieved from the repo.
Produce a clean, prioritized, de-duplicated triage by calling submit_triage exactly once.

- Flag duplicates ONLY from the provided existing-issues list, citing them by their id.
- Be specific in reproduction_steps; if the report is vague, infer the most likely steps and note that
  in suggested_next_action.

Priority rubric:
- P0: outage / data loss / security / billing-incorrect-charges — many users, no workaround.
- P1: major feature broken on a common path, painful workaround.
- P2: real bug, limited scope or easy workaround.
- P3: minor / cosmetic / nice-to-have / feature request."""

_STOP = set("the a an and or of to for in on is are was were be been it this that with from your you we "
            "i me my our they them so but if then when at as by not no can cant cant get got keep keeps".split())

# Plain-JSON shape for local models (JSON mode is far more reliable than tool calls on 8B models).
_JSON_SHAPE = (
    '{"title": "short title", "type": "bug|feature_request|question|task", '
    '"priority": "P0|P1|P2|P3", "priority_reason": "why", '
    '"severity": "critical|high|medium|low", "components": ["area"], "labels": ["label"], '
    '"summary": "1-2 sentences", "reproduction_steps": ["step 1", "step 2"], '
    '"expected_behavior": "...", "actual_behavior": "...", '
    '"duplicate_candidates": [{"id": "<id from the existing list>", "title": "...", '
    '"confidence": "high|medium|low", "reason": "..."}], "suggested_next_action": "..."}'
)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _keywords(text: str, n: int = 8) -> str:
    seen, words = set(), []
    for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        if len(w) > 2 and w not in _STOP and w not in seen:
            seen.add(w)
            words.append(w)
        if len(words) >= n:
            break
    return " ".join(words)


def _triage_single(provider, raw_text: str, source: str, reporter: Optional[str]) -> TriageResponse:
    """Fast single-pass RAG triage for local/slow models: retrieve in code, then ONE
    forced submit_triage call (short output → no timeout)."""
    queries = [raw_text]
    kw = _keywords(raw_text)
    if kw:
        queries.append(kw)  # light query expansion (multi-stage retrieval)

    all_ctx: dict[str, RetrievedContext] = {}
    searches: list[str] = []
    for q in queries:
        searches.append(q[:80])
        _, ctx = _run_search(q, k=5)
        for c in ctx:
            all_ctx.setdefault(c.id, c)
    retrieved = sorted(all_ctx.values(), key=lambda c: -c.score)[:6]

    # Keep the prompt small so local CPU inference stays fast: top 4, short snippets.
    candidates = "\n".join(
        f"- id={c.id} | {c.title} ({c.source}) | {c.snippet[:110]}" for c in retrieved[:4]
    ) or "(no existing issues found)"
    user_text = (
        f"Source: {source}\nReporter: {reporter or 'unknown'}\n\n"
        f"--- RAW REPORT ---\n{raw_text}\n\n"
        f"--- EXISTING ISSUES & DOCS (cite any duplicates by their id) ---\n{candidates}"
    )
    # JSON mode (not tool-calling) — far more reliable on local 8B models.
    system = (
        SYSTEM_PROMPT_SINGLE
        + "\n\nReturn ONLY a single JSON object (no prose, no markdown) with exactly this shape:\n"
        + _JSON_SHAPE
    )
    turn = provider.chat(
        system=system, messages=[user_msg(user_text)],
        max_tokens=1500, response_format={"type": "json_object"},
    )
    data = _extract_json(turn.text)
    if not isinstance(data, dict):
        raise RuntimeError("The model did not return a parseable JSON triage. Try qwen2.5 or Claude.")
    triaged = TriagedIssue(**data)
    return TriageResponse(triage=triaged, retrieved=retrieved, agent_searches=searches)


def _run_triage(provider, raw_text: str, source: str, reporter: Optional[str]) -> TriageResponse:
    # Cloud models (Claude) are fast enough for the full agentic multi-search loop;
    # local/CPU models use the single-pass path to stay responsive.
    if provider.name != "anthropic":
        return _triage_single(provider, raw_text, source, reporter)

    seed_text, seed_ctx = _run_search(raw_text, k=5)
    all_contexts: dict[str, RetrievedContext] = {c.id: c for c in seed_ctx}
    agent_searches: list[str] = []

    user_text = (
        f"Source: {source}\nReporter: {reporter or 'unknown'}\n\n"
        f"--- RAW REPORT ---\n{raw_text}\n\n"
        f"--- INITIAL SEARCH RESULTS (for grounding; search more if useful) ---\n{seed_text}"
    )
    messages = [user_msg(user_text)]
    triaged: Optional[TriagedIssue] = None

    # Phase 1 — agentic search: let the model look for duplicates/context (or submit early).
    for _ in range(MAX_ITERS - 1):
        turn = provider.chat(
            system=SYSTEM_PROMPT, messages=messages,
            tools=[SEARCH_TOOL, SUBMIT_TOOL], max_tokens=8000,
        )
        messages.append(turn.assistant_message)
        if not turn.tool_calls:
            break  # model stopped using tools — go force the submit below
        submitted = False
        for tc in turn.tool_calls:
            if tc.name == "submit_triage":
                triaged = TriagedIssue(**tc.arguments)
                submitted = True
                messages.append(tool_result_msg(tc.id, "Triage recorded."))
            elif tc.name == "search_existing_issues":
                q = tc.arguments.get("query", "")
                k = int(tc.arguments.get("k", 5) or 5)
                agent_searches.append(q)
                text, ctx = _run_search(q, k=k)
                for c in ctx:
                    all_contexts.setdefault(c.id, c)
                messages.append(tool_result_msg(tc.id, text))
            else:
                messages.append(tool_result_msg(tc.id, f"Unknown tool: {tc.name}"))
        if submitted:
            break

    # Phase 2 — guaranteed structured output: a dedicated call with ONLY the submit
    # tool, forced. Even small local models comply when it's the only option.
    if triaged is None:
        messages.append(user_msg(
            "Now produce the final triage. Call submit_triage with all fields filled in. "
            "Do not reply with prose."
        ))
        turn = provider.chat(
            system=SYSTEM_PROMPT, messages=messages,
            tools=[SUBMIT_TOOL], max_tokens=8000, force_tool="submit_triage",
        )
        messages.append(turn.assistant_message)
        for tc in turn.tool_calls:
            if tc.name == "submit_triage":
                triaged = TriagedIssue(**tc.arguments)

    if triaged is None:
        raise RuntimeError(
            "The model did not produce a triage even when forced. Your LLM provider may not "
            "support forced tool calls — try a tool-capable model (llama3.1 / qwen2.5) or Claude."
        )

    retrieved = sorted(all_contexts.values(), key=lambda c: -c.score)[:6]
    return TriageResponse(triage=triaged, retrieved=retrieved, agent_searches=agent_searches)


def triage(raw_text: str, source: str = "other", reporter: Optional[str] = None) -> TriageResponse:
    last_err: Optional[Exception] = None
    for _ in range(2):  # primary provider, then one failover
        provider = get_active_provider()
        if provider is None:
            raise RuntimeError(
                "No LLM provider available. Set ANTHROPIC_API_KEY, or OPENAI_COMPAT_* vars, "
                "or run a local Ollama model."
            )
        try:
            return _run_triage(provider, raw_text, source, reporter)
        except ProviderError as exc:
            last_err = exc
            if not exc.recoverable:
                raise RuntimeError(str(exc))
            invalidate()  # re-select a different provider and retry
    raise RuntimeError(f"All LLM providers failed: {last_err}")


# ---- Release notes generation (a normal grounded LLM call) ----
RELEASE_SYSTEM = """You write concise, professional release notes for an engineering team.
Group the resolved issues into sections: New Features, Bug Fixes, Improvements. Omit empty \
sections. Each line should be user-facing and reference the issue number like (#123). Output \
GitHub-flavored Markdown only — start with a '## Release Notes' heading."""


def generate_release_notes(items: list[dict]) -> str:
    listing = "\n".join(
        f"- #{it['number']} {it['title']} [labels: {', '.join(it.get('labels', [])) or 'none'}]"
        for it in items
    )
    last_err: Optional[Exception] = None
    for _ in range(2):
        provider = get_active_provider()
        if provider is None:
            raise RuntimeError("No LLM provider available for release notes.")
        try:
            turn = provider.chat(
                system=RELEASE_SYSTEM,
                messages=[user_msg(f"Resolved issues:\n{listing}")],
                max_tokens=2000,
            )
            return turn.text.strip()
        except ProviderError as exc:
            last_err = exc
            if not exc.recoverable:
                raise RuntimeError(str(exc))
            invalidate()
    raise RuntimeError(f"All LLM providers failed: {last_err}")

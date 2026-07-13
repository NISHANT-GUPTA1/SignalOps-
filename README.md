# AI Bug Triage — a Slack Agent

Paste any messy bug report into Slack and an AI agent turns it into a clean, **prioritized,
de-duplicated** engineering issue — with reproduction steps — then files it to **GitHub** (private
repos included) or a **Plane** board with one click. It also drafts **release notes** from resolved
issues.

The agent lives **inside Slack**: a slash command, a message shortcut, an `@mention`, and the native
**AI Assistant panel**. The brain is a genuine **agentic-RAG** loop (Claude Opus 4.8, hybrid
vector + keyword search over a Qdrant DB) grounded in your team's real issue history — no mocks, no
seed data.

> Built for the **Slack Agent Builder Challenge** · Track: **New Slack Agent**.

---

## What it does

- **Turns raw reports into proper issues** — title, summary, reproduction steps, expected vs. actual.
- **Prioritizes automatically** — P0–P3 + severity, each with a stated reason.
- **Catches duplicates before filing** — checks every report against your existing issues.
- **Files with one click** — a *File to GitHub / Plane* button creates the real issue; the card updates to ✅.
- **Writes release notes** — `/release-notes` turns resolved issues into a changelog.
- **Grounded in your real history** — every decision cites your team's own past issues.

## The four ways to use it in Slack

| You do this | It does this |
|---|---|
| `/triage <paste report>` | Posts an interactive triage card to the channel |
| Hover a message → **⚡ Triage this** | Triages that exact message in a thread |
| `@BugTriage <report>` | Triages in a channel thread |
| Open the **AI Assistant panel** and paste | Chat-style triage, back in the thread |

## Challenge technologies used (two of the three)

| Pillar | How it's used |
|---|---|
| **Slack AI capabilities** | Native **AI Assistant panel** (Agents & AI Apps) — suggested prompts, live status, threaded replies. |
| **MCP server integration** | The engine is published as an **MCP server** (`python -m app.mcp_server`) with tools `triage_report`, `search_issues`, `file_issue`, `release_notes`. |

---

## Quick start

### 1. Install
```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # then edit .env (see keys below)
```

### 2. Create the Slack app
- Go to <https://api.slack.com/apps> → **Create New App** → **From a manifest** → paste
  [`slack_manifest.yaml`](slack_manifest.yaml).
- **Basic Information → App-Level Tokens** → generate a token with scope `connections:write` →
  `SLACK_APP_TOKEN` (`xapp-…`).
- **Install App** → copy the **Bot User OAuth Token** → `SLACK_BOT_TOKEN` (`xoxb-…`).
- Ensure **Settings → Socket Mode** is **On**.

### 3. Configure `.env`
```ini
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
ANTHROPIC_API_KEY=sk-ant-...       # the AI brain (or OPENAI_COMPAT_* / local Ollama)
GITHUB_TOKEN=ghp_...               # to file issues (private-repo capable)
GITHUB_REPO_URL=owner/name         # default repo to file into
EMBEDDING_PROVIDER=voyage          # or ollama (free, local) or hash (offline)
VOYAGE_API_KEY=...                 # if using voyage
```

### 4. Seed the knowledge base (so dedup + grounding work)
Ingest a repo's issue history once via the web console:
```bash
uvicorn app.main:app --port 8077   # open http://127.0.0.1:8077/app → Ingest a repo
```

### 5. Run the Slack agent
```bash
python run_slack.py
```
When you see `⚡️ Slack agent connected`, invite the bot to a channel (`/invite @BugTriage`) and try
`/triage <a bug report>`.

> Only one `run_slack.py` at a time — embedded Qdrant locks the store to a single process.

---

## Architecture

```
   Slack:  /triage  ·  ⚡ Triage this  ·  @mention  ·  AI Assistant panel  ·  [File] buttons
                                   │  Socket Mode (WebSocket)
                                   ▼
             app/slack_agent.py  ──▶  app/slack_blocks.py  (Block Kit cards)
                                   │
                                   ▼
        ┌──────────────── app/agent.py — agentic RAG ─────────────────┐
        │  Claude Opus 4.8 tool loop                                  │
        │   ├─ search_existing_issues ─▶ Qdrant hybrid                │
        │   │                             (dense HNSW + BM25 → RRF)   │
        │   └─ submit_triage (strict structured output) ── ends loop  │
        └────────────────────────────┬────────────────────────────────┘
                                      ▼
                Triaged issue ─▶ GitHub (private-repo capable)  or  Plane

        Same engine also published over MCP:  app/mcp_server.py
        KB seeded via web console:  GitHub issues + docs ─▶ chunk ─▶ embed ─▶ Qdrant
```

A rendered version is in [`architecture.html`](architecture.html) (open in a browser).

---

## MCP server (standalone)

```bash
python -m app.mcp_server            # stdio transport
```
Tools: `triage_report`, `search_issues`, `file_issue`, `release_notes`. Point any MCP client at it.

---

## The engine

### LLM — fallback chain (paid when available, free otherwise)
The agent runs on the first working provider, shown in the status bar (`LLM: <provider> / <model>`):

1. **Anthropic (Claude)** — `ANTHROPIC_API_KEY`. Best quality; native tool use + structured output.
2. **OpenAI-compatible** — `OPENAI_COMPAT_BASE_URL` + `OPENAI_COMPAT_API_KEY` + `OPENAI_COMPAT_MODEL`.
3. **Local Ollama (free, no key)** — `ollama pull llama3.1 && ollama serve`.

Pin one with `LLM_PROVIDER=anthropic|openai-compat|ollama`.

### Embeddings (pluggable)
`EMBEDDING_PROVIDER=voyage|ollama|hash`. `voyage` = hosted semantic (free tier is rate-limited;
add a card at dash.voyageai.com for higher limits). `ollama` = free local semantic
(`ollama pull nomic-embed-text`). `hash` = offline lexical fallback, no API.

### Vector DB — Qdrant (embedded)
Real vector DB with HNSW ANN, no Docker. Set `QDRANT_URL` / `QDRANT_API_KEY` to use Qdrant Cloud/Docker.

### Other keys

| Key | Required? | Unlocks |
|---|---|---|
| `GITHUB_TOKEN` | For private repos / writes | Ingest at 5,000 req/hr **and** create issues. PAT: **Contents: Read** + **Issues: Read and write**. |
| `GITHUB_REPO_URL` | For one-click GitHub filing | Default repo the agent files into (or pass `repo:owner/name` per command). |
| `PLANE_*` (4 vars) | Optional | The *File to Plane* button files a real work item. |

---

## Web console (secondary surface)

The FastAPI app (`uvicorn app.main:app`) is used to **ingest** a repo into the knowledge base and to
inspect status. It's the admin surface; Slack is the primary one.

---

## Project layout

```
run_slack.py           Socket Mode entrypoint for the Slack agent
slack_manifest.yaml    one-paste Slack app definition
architecture.html      rendered architecture diagram
app/
  slack_agent.py       Slack Bolt app: commands, mention, shortcut, AI assistant, buttons
  slack_blocks.py      Block Kit triage-card renderers
  mcp_server.py        MCP server publishing the engine as tools
  formatting.py        shared issue-body Markdown (web + Slack)
  agent.py             Claude agentic triage loop + release notes
  vectorstore.py       Qdrant vector DB + hybrid retrieval (dense HNSW + BM25, RRF)
  embeddings.py        voyage | ollama | openai | hash providers
  chunking.py          heading-aware overlapping chunker
  config.py            env-driven settings + capability flags
  models.py            Pydantic schemas
  main.py              FastAPI app (ingestion + web console)
  integrations/
    github_client.py   real GitHub: issues, repo docs, create issue
    plane_client.py    real Plane issue creation
    slack_client.py    read-only Slack channel puller (web console)
  static/index.html    single-file web UI
```

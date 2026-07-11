#  AI Bug Triage & Release Operator

An **industry-internal engineering tool** and a **genuine RAG + agentic-AI** showcase. Drop a messy
bug report (Slack / email / form / GitHub) and an AI agent turns it into a clean, **prioritized,
de-duplicated** engineering issue — with reproduction steps — then files it to **GitHub** (incl.
private repos) or a **Plane** board, and drafts **release notes** from resolved issues.

Everything is real: real LLM (Claude Opus 4.8), real embeddings (Voyage), real vector DB (Qdrant),
real ingestion from your actual repo. **No mocks, no seed data, no hardcoded fixtures** — the
knowledge base is whatever you ingest.

---

## How it maps to the GenAI / RAG learning roadmap

| Phase | Concept | Where it lives |
|---|---|---|
| 1 — LLMs | Next-token model, tool use, structured output | `app/agent.py` (Claude Opus 4.8) |
| 2 — Embeddings & Vector DBs | Embeddings, similarity, **ANN/HNSW**, vector DB | `app/embeddings.py` (Voyage), `app/vectorstore.py` (**Qdrant**) |
| 3 — RAG | **Chunking**, indexing, retrievers, **BM25 vs semantic**, **hybrid search**, **RRF**, grounding | `app/chunking.py`, `VectorStore.search()` |
| 4 — Orchestration | Document loaders, splitters, retrieval pipeline | `/api/ingest` (GitHub issues + repo markdown → chunk → embed → index) |
| 5 — Agents | Tool use / function calling, **agentic RAG** (model issues its own searches), task decomposition | `app/agent.py` tool loop |
| 6 — Production | Serving, config, observability of retrieval trace | FastAPI app, `/api/status`, retrieval trace in UI |

---

## Architecture

```
Raw report ─▶ FastAPI ─▶ Agent (Claude Opus 4.8, adaptive thinking)
                          │  ├─ tool: search_existing_issues ─▶ Qdrant hybrid (dense HNSW + BM25, fused by RRF)
                          │  └─ tool: submit_triage (strict structured output)  ◀── ends the loop
                          ▼
                 Triaged issue ─▶ GitHub (real, private-repo capable)  or  Plane (real)

GitHub repo ─▶ /api/ingest ─▶ issues + markdown docs ─▶ chunk ─▶ Voyage embeddings ─▶ Qdrant
Closed issues ─▶ Agent ─▶ Release notes (Markdown)
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env                   # then edit .env (see keys below)
uvicorn app.main:app --port 8077
# open http://127.0.0.1:8077
```

The store starts **empty**. In the UI: **(1)** enter your repo and click *Ingest issues*, then
**(2)** paste a real bug report and *Triage*.

### LLM provider — fallback chain (paid-when-available, free otherwise)

The agent runs on the **first working** provider, auto-selected by a health check and shown in the
status bar (`LLM: <provider> / <model>`):

1. **Anthropic (Claude)** — set `ANTHROPIC_API_KEY`. Best quality; native tool use + structured output.
2. **OpenAI-compatible** — set `OPENAI_COMPAT_BASE_URL` + `OPENAI_COMPAT_API_KEY` + `OPENAI_COMPAT_MODEL`
   (OpenAI, OpenRouter, Groq, Together, …). Used when Anthropic has no working key.
3. **Local Ollama (free, no key)** — the fallback when no paid key is present. Setup:
   ```bash
   # install from https://ollama.com, then:
   ollama pull llama3.1        # a tool-capable model (or qwen2.5)
   ollama serve                # exposes http://localhost:11434
   ```
   Defaults: `OLLAMA_BASE_URL=http://localhost:11434/v1`, `OLLAMA_MODEL=llama3.1`.

Pin one tier with `LLM_PROVIDER=anthropic|openai-compat|ollama`. If none is available, triage returns
a clear error.

> **Cursor note:** a Cursor API key **cannot** be used here. Cursor's API is admin/analytics +
> cloud-agents only — it is not an LLM chat/completions endpoint, and Cursor blocks routing its own
> models to external callers. Use the OpenAI-compatible tier for a second paid provider.

### Other keys

| Key | Required? | Unlocks |
|---|---|---|
| `GITHUB_TOKEN` | For private repos / writes | Ingest private repos at 5,000 req/hr **and** create issues. Public-repo *reads* work tokenless. Fine-grained PAT: **Contents: Read** + **Issues: Read and write**. |
| `VOYAGE_API_KEY` | Recommended | Real semantic embeddings (`EMBEDDING_PROVIDER=voyage`). Free tier: https://dash.voyageai.com . Or `EMBEDDING_PROVIDER=ollama` for free local embeddings (no key), or `hash` (offline lexical, degraded). |
| `SLACK_BOT_TOKEN` | Optional | **Read-only.** Pull real messages from a Slack channel to triage (never posts). Bot scopes: `channels:read`, `channels:history`, `users:read`; invite the bot to the channel. |
| `PLANE_*` (3 vars) | Optional | The "Create in Plane" button files a real **work item** (create-only — never reads/edits/deletes). Otherwise use the GitHub target. |

> **Private repos:** fully supported. Provide a `GITHUB_TOKEN` with access to the repo — the same
> token both ingests (issues + docs) and creates issues.

---

## Demo script (~2 min)

1. **Ingest your repo** (box 1). The status bar shows `store: N chunks` (real issues + docs).
2. **Paste a messy report** (box 2) → **Triage with Claude**. You get a clean title, **P-level
   priority + reasoning**, reproduction steps, **flagged duplicates** (linked to real ingested
   issues), and a *Retrieval trace* showing the agent's own search queries (agentic RAG).
3. **→ Create in GitHub** (or Plane) — files the real issue and links it.
4. **Generate release notes** — Claude summarizes the repo's resolved/closed issues into Markdown.

---

## Vector DB note (Python 3.14)

The roadmap lists Chroma, Qdrant, FAISS, etc. This project uses **Qdrant in embedded local mode**
(`QdrantClient(path=...)`) — a real vector DB with HNSW ANN, no Docker required. (`chromadb` 1.5.x
ships Rust bindings that **segfault on Python 3.14**; Qdrant local mode does not.) To use **Qdrant
Cloud** (free tier) or a Docker container instead, just set `QDRANT_URL` / `QDRANT_API_KEY` — same code.

---

## Project layout

```
app/
  main.py            FastAPI app, endpoints, serves the UI
  agent.py           Claude agentic triage loop + release notes
  vectorstore.py     Qdrant vector DB + hybrid retrieval (dense HNSW + BM25, RRF)
  embeddings.py      voyage | openai | hash providers
  chunking.py        heading-aware overlapping chunker
  config.py          env-driven settings + capability flags
  models.py          Pydantic schemas
  integrations/
    github_client.py real GitHub: issues, repo markdown docs, create issue
    plane_client.py  real Plane issue creation
  static/index.html  single-file UI (no build step)
```

"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = _get("ANTHROPIC_API_KEY")
    claude_model: str = _get("CLAUDE_MODEL", "claude-opus-4-8")

    # LLM fallback chain: anthropic -> openai-compat -> ollama (local, free).
    # Optional pin: LLM_PROVIDER=anthropic|openai-compat|ollama
    llm_provider: str = _get("LLM_PROVIDER")
    openai_compat_base_url: str = _get("OPENAI_COMPAT_BASE_URL")
    openai_compat_api_key: str = _get("OPENAI_COMPAT_API_KEY")
    openai_compat_model: str = _get("OPENAI_COMPAT_MODEL")
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1")
    # Local CPU generation can be slow; allow a long read timeout for chat calls.
    llm_request_timeout: int = int(_get("LLM_TIMEOUT", "600"))

    github_token: str = _get("GITHUB_TOKEN")

    # Embeddings: voyage | openai | ollama (local, free, no rate limit) | hash
    embedding_provider: str = _get("EMBEDDING_PROVIDER", "voyage").lower()
    voyage_api_key: str = _get("VOYAGE_API_KEY")
    voyage_model: str = _get("VOYAGE_MODEL", "voyage-3.5")
    openai_api_key: str = _get("OPENAI_API_KEY")
    ollama_embed_model: str = _get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Vector DB: local embedded Qdrant by default; or point at Qdrant Cloud/Docker.
    qdrant_url: str = _get("QDRANT_URL")
    qdrant_api_key: str = _get("QDRANT_API_KEY")
    qdrant_path: str = _get("QDRANT_PATH", "data/qdrant")
    qdrant_collection: str = _get("QDRANT_COLLECTION", "kb")

    # Slack (read-only): a Bot token (xoxb-...) to pull channel messages.
    slack_bot_token: str = _get("SLACK_BOT_TOKEN")
    # Slack AGENT (read/write, Socket Mode): the bot posts triage cards, runs the
    # /triage command, the "Triage this" message shortcut, @mentions and the native
    # AI assistant panel. Needs a bot token (xoxb-, above) + an app-level token
    # (xapp-, with connections:write) for Socket Mode.
    slack_app_token: str = _get("SLACK_APP_TOKEN")
    slack_signing_secret: str = _get("SLACK_SIGNING_SECRET")  # only for HTTP mode
    # Default repo the Slack agent files issues into / draws release notes from.
    # Accepts 'owner/repo' or a full GitHub URL. Users can override per-command.
    default_repo_url: str = _get("GITHUB_REPO_URL")
    # Channel id the /release-notes digest posts to (optional).
    slack_digest_channel: str = _get("SLACK_DIGEST_CHANNEL")

    # Plane engineering board (real). PLANE_BASE_URL points at a self-hosted
    # instance; defaults to Plane Cloud. Trailing slash is stripped on read.
    plane_base_url: str = _get("PLANE_BASE_URL", "https://api.plane.so").rstrip("/")
    plane_api_key: str = _get("PLANE_API_KEY")
    plane_workspace_slug: str = _get("PLANE_WORKSPACE_SLUG")
    plane_project_id: str = _get("PLANE_PROJECT_ID")

    data_dir: str = _get("DATA_DIR", "data")

    # --- derived capability flags (used by the UI status banner) ---
    @property
    def has_claude(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def github_can_write(self) -> bool:
        return bool(self.github_token)

    @property
    def has_slack(self) -> bool:
        return bool(self.slack_bot_token)

    @property
    def slack_agent_ready(self) -> bool:
        """Socket-Mode agent needs a bot token AND an app-level token."""
        return bool(self.slack_bot_token and self.slack_app_token)

    @property
    def default_target(self) -> str:
        """Where the Slack agent files by default: github if a repo+token exist,
        else plane if configured, else github (which will prompt for a repo)."""
        if self.default_repo_url and self.github_token:
            return "github"
        if self.plane_is_live:
            return "plane"
        return "github"

    @property
    def plane_is_live(self) -> bool:
        return bool(self.plane_api_key and self.plane_workspace_slug and self.plane_project_id)

    @property
    def effective_embedding_provider(self) -> str:
        """Fall back to 'hash' only if the chosen hosted provider has no key.
        'ollama' needs no key (local) and 'hash' is offline — both pass through."""
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            return "hash"
        if self.embedding_provider == "openai" and not self.openai_api_key:
            return "hash"
        return self.embedding_provider

    @property
    def embeddings_are_real(self) -> bool:
        return self.effective_embedding_provider in ("voyage", "openai", "ollama")

    @property
    def qdrant_mode(self) -> str:
        return "server" if self.qdrant_url else "local"


settings = Settings()

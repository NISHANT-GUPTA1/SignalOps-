"""Start the Slack Bug Triage agent over Socket Mode.

    python run_slack.py

Needs (in .env):
  SLACK_BOT_TOKEN   xoxb-...   bot token with the scopes in slack_manifest.yaml
  SLACK_APP_TOKEN   xapp-...   app-level token with connections:write (Socket Mode)
plus at least one LLM provider (ANTHROPIC_API_KEY / OPENAI_COMPAT_* / local Ollama).

Socket Mode keeps a WebSocket to Slack, so no public URL / tunnel is required —
ideal for a developer sandbox and for the hackathon demo.
"""
from __future__ import annotations

import sys

from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.config import settings
from app.slack_agent import build_app, startup_banner


def main() -> int:
    if not settings.slack_bot_token:
        print("ERROR: SLACK_BOT_TOKEN is not set (needs a xoxb- bot token).", file=sys.stderr)
        return 1
    if not settings.slack_app_token:
        print("ERROR: SLACK_APP_TOKEN is not set (needs a xapp- app-level token "
              "with connections:write for Socket Mode).", file=sys.stderr)
        return 1

    print(startup_banner())
    # Warm the vector store so the first triage isn't slow.
    try:
        from app.vectorstore import get_store
        print(f"  store     : {get_store().stats()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  store     : (lazy — will open on first use) [{exc}]")

    app = build_app()
    print("⚡️ Slack agent connected. Try /triage, @mention, the 'Triage this' "
          "shortcut, or the AI assistant panel.")
    SocketModeHandler(app, settings.slack_app_token).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

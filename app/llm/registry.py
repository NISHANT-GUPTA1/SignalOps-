"""Provider registry + selection.

Priority order: Anthropic -> OpenAI-compatible -> local Ollama. The first
provider that is configured AND passes a health check becomes active. The active
choice is cached for the process and re-evaluated when invalidated (e.g. after a
mid-call auth/connection failure).
"""
from __future__ import annotations

from typing import Optional

from ..config import settings
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .openai_compat_provider import OpenAICompatProvider

_active: Optional[LLMProvider] = None
_resolved = False


def _build_chain() -> list[LLMProvider]:
    chain: list[LLMProvider] = []
    if settings.anthropic_api_key:
        chain.append(AnthropicProvider(settings.anthropic_api_key, settings.claude_model))
    if settings.openai_compat_base_url and settings.openai_compat_model:
        chain.append(
            OpenAICompatProvider(
                "openai-compat",
                settings.openai_compat_base_url,
                settings.openai_compat_api_key,
                settings.openai_compat_model,
                requires_key=True,
            )
        )
    # Free local tier (no key). Always a candidate; health() decides if it's up.
    chain.append(
        OpenAICompatProvider(
            "ollama",
            settings.ollama_base_url,
            "ollama",
            settings.ollama_model,
            requires_key=False,
        )
    )
    # Optional pin via LLM_PROVIDER
    if settings.llm_provider:
        chain = [p for p in chain if p.name == settings.llm_provider]
    return chain


def get_active_provider() -> Optional[LLMProvider]:
    global _active, _resolved
    if _resolved:
        return _active
    for provider in _build_chain():
        if provider.available() and provider.health():
            _active = provider
            _resolved = True
            return _active
    _active = None
    _resolved = True
    return None


def invalidate() -> None:
    """Force re-selection on the next get_active_provider() call."""
    global _active, _resolved
    _active = None
    _resolved = False


def describe() -> dict:
    """Status snapshot for /api/status (forces a fresh resolution)."""
    invalidate()
    active = get_active_provider()
    return {
        "active_provider": active.name if active else "none",
        "model": active.model if active else None,
        "tiers": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai_compat": bool(settings.openai_compat_base_url and settings.openai_compat_model),
            "ollama": True,  # candidate iff reachable; reflected by active_provider
        },
    }

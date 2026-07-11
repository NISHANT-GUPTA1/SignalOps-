"""Provider-agnostic LLM layer with a fallback chain:
Anthropic (Claude) -> OpenAI-compatible -> local Ollama.
"""
from .registry import describe, get_active_provider, invalidate

__all__ = ["get_active_provider", "invalidate", "describe"]

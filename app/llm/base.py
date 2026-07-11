"""Normalized LLM types + provider interface.

The internal message format is OpenAI-ish (the lingua franca that Ollama also
speaks); each provider converts to/from its native wire format. Assistant turns
may carry an opaque `_native` field so a provider can replay its own turns
losslessly (e.g. Anthropic thinking-block signatures).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool input
    strict: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # normalized assistant message to append to history for the next chat() call
    assistant_message: dict = field(default_factory=dict)


# ---- normalized message helpers ----
def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def tool_result_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ProviderError(Exception):
    """Raised by a provider on a recoverable failure (auth/connection) so the
    registry can fall through to the next provider."""

    def __init__(self, message: str, *, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def available(self) -> bool:
        """True if this provider is configured (keys/urls present)."""

    @abstractmethod
    def health(self) -> bool:
        """Cheap reachability/auth check. False -> skip this provider."""

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: Optional[list[ToolDef]] = None,
        max_tokens: int = 4096,
        force_tool: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> ChatTurn:
        """One model turn. `force_tool` requires the model to call that tool;
        `response_format` (OpenAI-compatible providers) constrains output to JSON."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model}

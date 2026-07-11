"""Anthropic (Claude) provider — wraps the Messages API with adaptive thinking.

Converts the normalized message format to/from Anthropic content blocks. Assistant
turns are replayed from their raw `_native` content so thinking-block signatures
round-trip losslessly on the same model.
"""
from __future__ import annotations

from typing import Optional

import anthropic

from .base import ChatTurn, LLMProvider, ProviderError, ToolCall, ToolDef


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.name = "anthropic"
        self.model = model
        self._api_key = api_key
        self._client: Optional[anthropic.Anthropic] = None

    def _client_lazy(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def available(self) -> bool:
        return bool(self._api_key)

    def health(self) -> bool:
        try:
            self._client_lazy().models.retrieve(self.model)  # cheap GET, no tokens
            return True
        except Exception:
            return False

    # ---- normalized -> Anthropic ----
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tools():
            nonlocal pending_tool_results
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for m in messages:
            role = m["role"]
            if role == "tool":
                pending_tool_results.append(
                    {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                )
                continue
            flush_tools()
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                if m.get("_native") is not None:
                    out.append({"role": "assistant", "content": m["_native"]})
                else:
                    blocks: list[dict] = []
                    if m.get("content"):
                        blocks.append({"type": "text", "text": m["content"]})
                    for tc in m.get("tool_calls", []):
                        blocks.append(
                            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                        )
                    out.append({"role": "assistant", "content": blocks})
        flush_tools()
        return out

    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: Optional[list[ToolDef]] = None,
        max_tokens: int = 4096,
        force_tool: Optional[str] = None,
        response_format: Optional[dict] = None,  # ignored (Anthropic uses tool-based output)
    ) -> ChatTurn:
        client = self._client_lazy()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"},
            "system": system,
            "messages": self._to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                    **({"strict": True} if t.strict else {}),
                }
                for t in tools
            ]
        if force_tool:
            # forcing a tool is incompatible with extended thinking
            kwargs.pop("thinking", None)
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        try:
            resp = client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            # Not every model supports adaptive thinking. If that's why the
            # request was rejected, retry once without it so triage still works
            # (e.g. when CLAUDE_MODEL is set to an older model in production).
            if "thinking" in str(exc).lower() and kwargs.get("thinking") is not None:
                kwargs.pop("thinking", None)
                try:
                    resp = client.messages.create(**kwargs)
                except anthropic.APIError as exc2:
                    raise ProviderError(f"anthropic request failed: {exc2}", recoverable=False)
            else:
                raise ProviderError(f"anthropic bad request: {exc}", recoverable=False)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ProviderError(f"anthropic auth failed: {exc}", recoverable=True)
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"anthropic connection failed: {exc}", recoverable=True)

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        assistant_message = {
            "role": "assistant",
            "content": text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
            "_native": resp.content,  # lossless replay (thinking blocks, signatures)
        }
        return ChatTurn(text=text, tool_calls=tool_calls, assistant_message=assistant_message)

"""OpenAI-compatible provider — one class for BOTH the paid OpenAI-compatible
tier (OpenAI / OpenRouter / Groq / Together / ...) and local Ollama.

Talks plain HTTP to `{base_url}/chat/completions` (no SDK dependency). Ollama is
just `base_url=http://localhost:11434/v1` with a dummy key.
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from ..config import settings
from .base import ChatTurn, LLMProvider, ProviderError, ToolCall, ToolDef


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, *, requires_key: bool):
        self.name = name
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._requires_key = requires_key

    def available(self) -> bool:
        if not (self.base_url and self.model):
            return False
        if self._requires_key and not self._api_key:
            return False
        return True

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    # ---- normalized -> OpenAI ----
    @staticmethod
    def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    **({"strict": True} if t.strict else {}),
                },
            }
            for t in tools
        ]

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
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._to_openai_messages(system, messages),
        }
        if tools:
            body["tools"] = self._to_openai_tools(tools)
        if force_tool:
            body["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        if response_format:
            body["response_format"] = response_format

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions", headers=self._headers(), json=body,
                timeout=settings.llm_request_timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"{self.name} connection failed: {exc}", recoverable=True)
        if r.status_code in (401, 403):
            raise ProviderError(f"{self.name} auth failed: {r.status_code}", recoverable=True)
        if r.status_code >= 400:
            # 4xx other than auth is a real request bug — not recoverable by failover
            raise ProviderError(f"{self.name} error {r.status_code}: {r.text[:300]}", recoverable=False)

        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id") or fn.get("name", ""), name=fn.get("name", ""), arguments=args))

        assistant_message = {
            "role": "assistant",
            "content": text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
        }
        return ChatTurn(text=text, tool_calls=tool_calls, assistant_message=assistant_message)

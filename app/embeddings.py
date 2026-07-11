"""Pluggable embedding providers: voyage | openai | ollama | hash.

- voyage : hosted, real semantic (free tier is heavily rate-limited).
- openai : hosted, real semantic.
- ollama : LOCAL, real semantic, FREE, no key, no rate limit (recommended when
           Voyage's free tier throttles). Run `ollama pull nomic-embed-text`.
- hash   : offline deterministic fallback (lexical only) so the app always boots.
The /api/status banner reports honestly whether embeddings are real or degraded.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Literal

import requests

from .config import settings

InputType = Literal["document", "query"]

# Batch size: small enough that LOCAL CPU embedding (Ollama) finishes a request
# well within the timeout, and under hosted caps (Voyage 128). Timeout is generous
# because local CPU embedding of a batch can take a while (model load + no GPU).
_EMBED_BATCH = 24
_EMBED_TIMEOUT = 300
_MAX_RETRIES = 6

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_DIM = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _HASH_DIM
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _HASH_DIM] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def _post_with_retry(url: str, headers: dict, payload: dict) -> dict:
    """POST with exponential backoff on 429 (rate limit), honoring Retry-After."""
    for attempt in range(_MAX_RETRIES):
        resp = requests.post(url, headers=headers, json=payload, timeout=_EMBED_TIMEOUT)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        wait = float(resp.headers.get("retry-after", 0) or 0) or min(2 ** attempt, 30)
        print(f"[embeddings] 429 rate-limited; retrying in {wait:.0f}s "
              f"(attempt {attempt + 1}/{_MAX_RETRIES})")
        time.sleep(wait)
    raise RuntimeError(
        "Embedding provider is rate-limiting (HTTP 429) even after retries. "
        "Voyage's no-card free tier allows only a few requests/min — add a payment method at "
        "dash.voyageai.com for much higher free limits (still has a large free quota), ingest "
        "fewer items at once, or set EMBEDDING_PROVIDER=hash to unblock without any API."
    )


def _batched(texts: list[str], fn) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        out.extend(fn(texts[i:i + _EMBED_BATCH]))
    return out


def _voyage_embed(texts: list[str], input_type: InputType) -> list[list[float]]:
    def one(batch: list[str]) -> list[list[float]]:
        data = _post_with_retry(
            "https://api.voyageai.com/v1/embeddings",
            {"Authorization": f"Bearer {settings.voyage_api_key}"},
            {"input": batch, "model": settings.voyage_model, "input_type": input_type},
        )
        return [d["embedding"] for d in data["data"]]

    return _batched(texts, one)


def _openai_embed(texts: list[str]) -> list[list[float]]:
    def one(batch: list[str]) -> list[list[float]]:
        data = _post_with_retry(
            "https://api.openai.com/v1/embeddings",
            {"Authorization": f"Bearer {settings.openai_api_key}"},
            {"input": batch, "model": "text-embedding-3-small"},
        )
        return [d["embedding"] for d in data["data"]]

    return _batched(texts, one)


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Local Ollama embeddings via its OpenAI-compatible endpoint. No rate limit."""
    base = settings.ollama_base_url.rstrip("/")

    def one(batch: list[str]) -> list[list[float]]:
        data = _post_with_retry(
            f"{base}/embeddings",
            {"Authorization": "Bearer ollama"},
            {"input": batch, "model": settings.ollama_embed_model},
        )
        # OpenAI-compatible responses preserve input order in data[]
        return [d["embedding"] for d in data["data"]]

    return _batched(texts, one)


def embed(texts: list[str], input_type: InputType = "document") -> list[list[float]]:
    if not texts:
        return []
    provider = settings.effective_embedding_provider
    if provider == "voyage":
        return _voyage_embed(texts, input_type)
    if provider == "openai":
        return _openai_embed(texts)
    if provider == "ollama":
        return _ollama_embed(texts)
    return _hash_embed(texts)


def embed_one(text: str, input_type: InputType = "query") -> list[float]:
    return embed([text], input_type)[0]


def embedding_dim() -> int:
    """Probe the active provider for its vector dimension (Qdrant needs it up front)."""
    return len(embed_one("dimension probe", input_type="document"))

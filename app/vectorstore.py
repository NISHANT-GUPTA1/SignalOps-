"""Real vector database (Qdrant) with HYBRID retrieval.

* Dense / semantic: Qdrant HNSW ANN over embeddings (Phase 2: ANN/HNSW, vector DB).
* Sparse / keyword: BM25 over document tokens (matters for code & function names).
* Fusion: Reciprocal Rank Fusion (RRF) — the multi-stage hybrid pipeline.

Qdrant runs in embedded LOCAL mode (no Docker) by default, or against Qdrant
Cloud/Docker if QDRANT_URL is set. No mocks, no seed data — the store is
populated only by real ingestion (`/api/ingest`).
"""
from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import re

from qdrant_client import QdrantClient, models

from .config import settings
from .embeddings import embed, embed_one, embedding_dim

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NS = uuid.UUID("b9d6f1a2-4c3e-4a1b-9f2d-000000000001")  # fixed namespace for deterministic ids


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _pid(logical_id: str) -> str:
    return str(uuid.uuid5(_NS, logical_id))


@dataclass
class Doc:
    id: str
    title: str
    text: str
    source: str  # "github_issue" | "doc"
    url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float


class VectorStore:
    def __init__(self) -> None:
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.qdrant_collection
        self.dim = embedding_dim()
        self._ensure_collection()
        # in-memory mirror of payloads for BM25 + reconstruction
        self.corpus: dict[str, Doc] = {}
        self._load_corpus()

    # ---------- collection lifecycle ----------
    def _ensure_collection(self) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists:
            info = self.client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore[union-attr]
            if size != self.dim:
                # embedding provider/dim changed — rebuild cleanly
                print(f"[store] vector dim changed {size}->{self.dim}; recreating collection")
                self.client.delete_collection(self.collection)
                exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    def _load_corpus(self) -> None:
        self.corpus.clear()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, with_payload=True, with_vectors=False, limit=256, offset=offset
            )
            for p in points:
                pl = p.payload or {}
                doc = Doc(
                    id=pl.get("_id", str(p.id)),
                    title=pl.get("title", ""),
                    text=pl.get("text", ""),
                    source=pl.get("source", ""),
                    url=pl.get("url"),
                    metadata=pl.get("metadata", {}) or {},
                )
                self.corpus[doc.id] = doc
            if offset is None:
                break

    # ---------- ingestion ----------
    def add(self, docs: list[Doc]) -> int:
        if not docs:
            return 0
        vectors = embed([f"{d.title}\n\n{d.text}" for d in docs], input_type="document")
        points = []
        new = 0
        for d, v in zip(docs, vectors):
            if d.id not in self.corpus:
                new += 1
            self.corpus[d.id] = d
            points.append(
                models.PointStruct(
                    id=_pid(d.id),
                    vector=v,
                    payload={
                        "_id": d.id, "title": d.title, "text": d.text,
                        "source": d.source, "url": d.url, "metadata": d.metadata,
                    },
                )
            )
        self.client.upsert(self.collection, points=points)
        return new

    # ---------- retrieval ----------
    def _dense_rank(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.corpus:
            return []
        qv = embed_one(query, input_type="query")
        res = self.client.query_points(self.collection, query=qv, limit=k, with_payload=True).points
        return [((p.payload or {}).get("_id", str(p.id)), float(p.score)) for p in res]

    def _bm25_rank(self, query: str, k: int, k1: float = 1.5, b: float = 0.75):
        ids = list(self.corpus.keys())
        if not ids:
            return []
        corpus = {i: _tok(f"{self.corpus[i].title} {self.corpus[i].text}") for i in ids}
        lengths = {i: len(corpus[i]) for i in ids}
        avgdl = (sum(lengths.values()) / len(ids)) or 1.0
        df: Counter = Counter()
        for toks in corpus.values():
            df.update(set(toks))
        N = len(ids)
        q_terms = _tok(query)
        scored: list[tuple[str, float]] = []
        for i in ids:
            tf = Counter(corpus[i])
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf[term] + k1 * (1 - b + b * lengths[i] / avgdl)
                s += idf * (tf[term] * (k1 + 1)) / denom
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[Hit]:
        pool = max(k * 3, 10)
        dense = self._dense_rank(query, pool)
        sparse = self._bm25_rank(query, pool)
        fused: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(dense):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(doc=self.corpus[i], score=round(s, 5)) for i, s in ranked if i in self.corpus]

    # ---------- introspection ----------
    def stats(self) -> dict:
        by_source: Counter = Counter(d.source for d in self.corpus.values())
        return {"total": len(self.corpus), "by_source": dict(by_source)}

    def closed_issues(self, repo: Optional[str] = None) -> list[Doc]:
        out = []
        for d in self.corpus.values():
            if d.source != "github_issue":
                continue
            if d.metadata.get("state") != "closed":
                continue
            if repo and d.metadata.get("repo") != repo:
                continue
            out.append(d)
        return out

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

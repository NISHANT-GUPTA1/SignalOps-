"""Pydantic request/response schemas for the API surface."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---- Triage input ----
class TriageRequest(BaseModel):
    raw_text: str = Field(..., description="The messy, raw bug report / feedback.")
    source: Literal["slack", "email", "github", "form", "other"] = "other"
    reporter: Optional[str] = None


# ---- Triage output (mirrors the submit_triage tool schema) ----
# NOTE: fields are lenient (defaults + coercion) so output from smaller local
# models (e.g. llama3.1 via Ollama) never 500s on a missing/odd field. The tool
# schema still guides the model toward the proper enums; this is just a safety net.
class DuplicateCandidate(BaseModel):
    id: str = ""
    title: str = ""
    confidence: str = "low"
    reason: str = ""


class TriagedIssue(BaseModel):
    title: str = "Untitled issue"
    type: str = "bug"
    priority: str = "P2"
    priority_reason: str = ""
    severity: str = "medium"
    components: list[str] = []
    labels: list[str] = []
    summary: str = ""
    reproduction_steps: list[str] = []
    expected_behavior: str = ""
    actual_behavior: str = ""
    duplicate_candidates: list[DuplicateCandidate] = []
    suggested_next_action: str = ""

    @field_validator("components", "labels", "reproduction_steps", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("duplicate_candidates", mode="before")
    @classmethod
    def _coerce_dups(cls, v):
        if not v:
            return []
        if isinstance(v, dict):
            return [v]
        return v


class RetrievedContext(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str
    url: Optional[str] = None


class TriageResponse(BaseModel):
    triage: TriagedIssue
    retrieved: list[RetrievedContext]
    agent_searches: list[str] = []  # queries Claude issued itself (agentic RAG trace)


# ---- Issue creation ----
class CreateIssueRequest(BaseModel):
    triage: TriagedIssue
    target: Literal["github", "plane"] = "plane"
    repo_url: Optional[str] = None  # required for github


class CreateIssueResponse(BaseModel):
    ok: bool
    target: str
    id: str
    url: Optional[str] = None
    message: str = ""


# ---- Ingestion ----
class IngestRequest(BaseModel):
    repo_url: str
    state: Literal["open", "closed", "all"] = "all"
    limit: int = 50


class IngestResponse(BaseModel):
    ingested: int
    total_in_store: int
    message: str


# ---- Release notes ----
class ReleaseNotesRequest(BaseModel):
    repo_url: str
    limit: int = 30


class ReleaseNotesResponse(BaseModel):
    markdown: str
    issue_count: int

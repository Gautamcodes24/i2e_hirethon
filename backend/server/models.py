"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /api/query — ask a question."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to retrieve")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense/BM25 weight (1.0 = pure dense)")
    use_parent: bool = Field(default=True, description="Resolve child → parent chunks for richer context")


class IngestRequest(BaseModel):
    """Configuration overrides for ingestion (optional)."""
    enable_ocr: bool = True
    enable_vision: bool = True
    enable_parent_child: bool = True
    enable_quality_filter: bool = True
    enable_dedup: bool = True
    chunk_max_tokens: int = 256


# ── Response Models ───────────────────────────────────────────────────────────

class Citation(BaseModel):
    index: int
    citation: str
    score: float
    page: int | str | None = None
    pdf_name: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []
    scores: list[float] = []


class IngestStatusResponse(BaseModel):
    task_id: str
    status: str                # queued | running | completed | failed
    error: str | None = None
    current_step: int = 0
    total_steps: int = 5
    step_message: str = ""
    step_detail: str = ""


class IngestResultResponse(BaseModel):
    task_id: str
    status: str
    n_chunks: int = 0
    n_parents: int = 0
    n_tables: int = 0
    n_images_described: int = 0
    embedding_dim: int = 0


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    n_chunks: int
    version: str

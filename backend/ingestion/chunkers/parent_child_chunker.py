"""Hierarchical parent-child chunking for small-to-big retrieval.

Strategy: create two tiers of chunks —
  - **Child chunks** (~256 tokens) — small, precise, used for retrieval
  - **Parent chunks** (~1024 tokens) — larger context windows, returned to the LLM

Each child chunk has a ``parent_id`` linking it to the parent chunk that
contains it.  At retrieval time, you search against child embeddings but
return the parent's full text for richer context.

This dramatically improves answer quality because:
  1. Small chunks have more precise embeddings (less topic dilution)
  2. Large context windows give the LLM enough info to answer well
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.ingestion.chunkers.chunker import recursive_chunk

# ── Defaults ──────────────────────────────────────────────────────────────────
PARENT_MAX_TOKENS = 1024
PARENT_OVERLAP_TOKENS = 100
CHILD_MAX_TOKENS = 256
CHILD_OVERLAP_TOKENS = 30


def create_parent_child_chunks(
    text: str,
    *,
    page_num: int,
    section_heading: str = "",
    source: str = "native",
    base_metadata: dict[str, Any] | None = None,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
    parent_overlap_tokens: int = PARENT_OVERLAP_TOKENS,
    child_max_tokens: int = CHILD_MAX_TOKENS,
    child_overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split text into parent and child chunks with linked IDs.

    Returns
    -------
    (parent_chunks, child_chunks)
        Parent chunks have ``chunk_id``, ``is_parent=True``.
        Child chunks have ``parent_id`` linking to the parent.
    """
    if not text or not text.strip():
        return [], []

    meta_base = base_metadata or {}
    parent_chunks: list[dict[str, Any]] = []
    child_chunks: list[dict[str, Any]] = []

    # Step 1: Create parent-level chunks (large context windows)
    parents_raw = recursive_chunk(
        text,
        max_tokens=parent_max_tokens,
        overlap_tokens=parent_overlap_tokens,
    )

    for p_raw in parents_raw:
        parent_id = str(uuid.uuid4())[:12]
        parent_content = p_raw["content"]

        parent_chunk = {
            "content": parent_content,
            "type": "text",
            "page": page_num,
            "section_heading": section_heading,
            "chunk_id": parent_id,
            "is_parent": True,
            "metadata": {
                **meta_base,
                "pages": [page_num],
                "start_char": p_raw["start_char"],
                "end_char": p_raw["end_char"],
                "source": source,
                "chunk_level": "parent",
            },
        }
        parent_chunks.append(parent_chunk)

        # Step 2: Sub-chunk each parent into child chunks (precise retrieval)
        children_raw = recursive_chunk(
            parent_content,
            max_tokens=child_max_tokens,
            overlap_tokens=child_overlap_tokens,
        )

        for c_raw in children_raw:
            child_chunk = {
                "content": c_raw["content"],
                "type": "text",
                "page": page_num,
                "section_heading": section_heading,
                "parent_id": parent_id,
                "is_parent": False,
                "metadata": {
                    **meta_base,
                    "pages": [page_num],
                    "start_char": p_raw["start_char"] + c_raw["start_char"],
                    "end_char": p_raw["start_char"] + c_raw["end_char"],
                    "source": source,
                    "chunk_level": "child",
                    "parent_id": parent_id,
                },
            }
            child_chunks.append(child_chunk)

    return parent_chunks, child_chunks


def build_parent_lookup(parent_chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a chunk_id → parent_chunk mapping for fast retrieval-time lookup."""
    return {p["chunk_id"]: p for p in parent_chunks if p.get("chunk_id")}


def resolve_parents(
    child_results: list[dict[str, Any]],
    parent_lookup: dict[str, dict[str, Any]],
    *,
    deduplicate_parents: bool = True,
) -> list[dict[str, Any]]:
    """Given retrieved child chunks, resolve to their parent chunks for context.

    Parameters
    ----------
    child_results : list
        Retrieved child chunk dicts (each should have ``parent_id``).
    parent_lookup : dict
        Mapping from chunk_id → parent chunk dict.
    deduplicate_parents : bool
        If True, return each parent only once even if multiple children match.

    Returns
    -------
    list of parent chunks (unique if deduplicate_parents=True)
    """
    seen: set[str] = set()
    parents: list[dict[str, Any]] = []

    for child in child_results:
        pid = child.get("parent_id") or child.get("metadata", {}).get("parent_id")
        if not pid:
            # No parent — return child as-is
            parents.append(child)
            continue

        if deduplicate_parents and pid in seen:
            continue

        parent = parent_lookup.get(pid)
        if parent:
            parents.append(parent)
            seen.add(pid)
        else:
            # Parent not found — fall back to child
            parents.append(child)

    return parents

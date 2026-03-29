"""Recursive character-level text chunking with overlap.

Splits long text blocks into retrieval-friendly chunks (default ~500 tokens)
using a hierarchy of separators while preserving a configurable overlap so
that sentences are not cut mid-thought.

Tables and image descriptions are **not** chunked here — they are kept at
their natural granularity (full table + per-row, or full description).
"""

from __future__ import annotations

from typing import Sequence

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _token_len(text: str) -> int:
    """Fast whitespace-based token count approximation."""
    return len(text.split())


def recursive_chunk(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[dict]:
    """Split *text* into chunks of roughly *max_tokens* words.

    Parameters
    ----------
    text : str
        The input text to chunk.
    max_tokens : int
        Target maximum tokens per chunk (using whitespace word count).
    overlap_tokens : int
        Number of trailing tokens from the previous chunk to prepend to the
        next chunk for context continuity.
    separators : sequence of str
        Ordered list of separators to try.  The algorithm uses the first
        separator that produces pieces small enough; if none do, it falls
        through to the next separator.

    Returns
    -------
    list[dict]
        Each dict has ``content``, ``start_char``, ``end_char``.
    """
    text = text.strip()
    if not text:
        return []

    if _token_len(text) <= max_tokens:
        return [{"content": text, "start_char": 0, "end_char": len(text)}]

    chunks = _split_recursive(text, max_tokens, overlap_tokens, list(separators))

    # Annotate character offsets
    result: list[dict] = []
    search_start = 0
    for chunk_text in chunks:
        idx = text.find(chunk_text[:80], search_start)  # 80-char prefix match
        if idx == -1:
            idx = search_start
        result.append({
            "content": chunk_text,
            "start_char": idx,
            "end_char": idx + len(chunk_text),
        })
        # Don't advance search_start past the overlap region
        search_start = idx + max(1, len(chunk_text) - overlap_tokens * 6)

    return result


def _split_recursive(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    separators: list[str],
) -> list[str]:
    """Core recursive splitting logic."""
    if _token_len(text) <= max_tokens:
        return [text]

    if not separators:
        # Last resort: hard split by word count
        return _hard_split(text, max_tokens, overlap_tokens)

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        return _hard_split(text, max_tokens, overlap_tokens)

    pieces = text.split(sep)

    # Merge small adjacent pieces until they reach max_tokens
    merged_chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = _token_len(piece)

        if current_len + piece_len > max_tokens and current_parts:
            merged_text = sep.join(current_parts)
            merged_chunks.append(merged_text)

            # Build overlap from tail of current_parts
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                p_len = _token_len(p)
                if overlap_len + p_len > overlap_tokens:
                    break
                overlap_parts.insert(0, p)
                overlap_len += p_len

            current_parts = overlap_parts + [piece]
            current_len = sum(_token_len(p) for p in current_parts)
        else:
            current_parts.append(piece)
            current_len += piece_len

    if current_parts:
        merged_chunks.append(sep.join(current_parts))

    # Recursively split any chunk that's still too large
    final: list[str] = []
    for chunk in merged_chunks:
        if _token_len(chunk) > max_tokens:
            final.extend(_split_recursive(chunk, max_tokens, overlap_tokens, remaining_seps))
        else:
            final.append(chunk)

    return final


def _hard_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split by word count when no separator works."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        end = min(i + max_tokens, len(words))
        chunks.append(" ".join(words[i:end]))
        i = end - overlap_tokens if end < len(words) else end
    return chunks

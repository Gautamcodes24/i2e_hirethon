"""Chunk quality scoring and filtering.

Scores each chunk on multiple quality dimensions and filters out noise,
garbled OCR output, boilerplate, and low-information chunks that would
pollute retrieval.

Quality dimensions:
  - **Length score**: penalises very short or very long chunks
  - **Entropy score**: detects repetitive / garbled text
  - **Alphanumeric ratio**: catches OCR garbage (high symbol density)
  - **Boilerplate score**: detects repeated headers, "Page X of Y", etc.
  - **Language coherence**: basic word-length and vocab checks
  - **Information density**: ratio of unique words to total words
"""

from __future__ import annotations

import math
import re
from typing import Any

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_QUALITY_SCORE = 0.30          # chunks below this are filtered out
MIN_TOKENS = 10                   # absolute minimum words in a chunk
MAX_TOKENS = 2000                 # chunks above this are suspicious
MIN_ALPHA_RATIO = 0.50            # minimum fraction of alphanumeric chars
MIN_ENTROPY = 2.0                 # minimum character-level entropy
MIN_UNIQUE_WORD_RATIO = 0.15      # minimum unique-word / total-word ratio

# Boilerplate patterns
_BOILERPLATE_PATTERNS = [
    re.compile(r"^(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?$", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\s*\.{3,}\s*\d+\s*$"),  # TOC dot leaders: "..... 42"
    re.compile(r"^(copyright|©|all rights reserved)", re.IGNORECASE),
    re.compile(r"^(this page intentionally left blank)", re.IGNORECASE),
    re.compile(r"^(draft|confidential|internal use only)$", re.IGNORECASE),
]


# ── Scoring functions ─────────────────────────────────────────────────────────

def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text.lower():
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphanumeric or whitespace."""
    if not text:
        return 0.0
    alnum = sum(1 for c in text if c.isalnum() or c.isspace())
    return alnum / len(text)


def _unique_word_ratio(text: str) -> float:
    """Ratio of unique words to total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _avg_word_length(text: str) -> float:
    """Average word length — very short or very long signals noise."""
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _is_boilerplate(text: str) -> bool:
    """Check if text matches known boilerplate patterns."""
    stripped = text.strip()
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _length_score(n_tokens: int) -> float:
    """Score based on token count — peaks at 50-500 tokens."""
    if n_tokens < MIN_TOKENS:
        return 0.0
    if n_tokens <= 50:
        return 0.5 + 0.5 * (n_tokens / 50)
    if n_tokens <= 500:
        return 1.0
    if n_tokens <= MAX_TOKENS:
        return 1.0 - 0.3 * ((n_tokens - 500) / (MAX_TOKENS - 500))
    return 0.3


# ── Public API ────────────────────────────────────────────────────────────────

def score_chunk(chunk: dict[str, Any]) -> dict[str, float]:
    """Score a single chunk across multiple quality dimensions.

    Returns a dict with individual scores and a composite ``quality_score``.
    """
    content = chunk.get("content", "")
    words = content.split()
    n_tokens = len(words)

    # Individual dimension scores (0.0-1.0)
    length = _length_score(n_tokens)
    entropy = min(_char_entropy(content) / 5.0, 1.0)  # normalize: 5 bits ≈ max
    alpha = _alpha_ratio(content)
    uniqueness = _unique_word_ratio(content)
    avg_wl = _avg_word_length(content)
    coherence = 1.0 if 2.5 <= avg_wl <= 12.0 else 0.5
    boilerplate = 0.0 if _is_boilerplate(content) else 1.0

    # Weighted composite
    composite = (
        0.15 * length
        + 0.20 * entropy
        + 0.15 * alpha
        + 0.15 * uniqueness
        + 0.10 * coherence
        + 0.25 * boilerplate
    )

    # Special types get a floor — tables and images are always valuable
    chunk_type = chunk.get("type", "text")
    if chunk_type in ("table", "image"):
        composite = max(composite, 0.60)

    return {
        "length_score": round(length, 3),
        "entropy_score": round(entropy, 3),
        "alpha_ratio": round(alpha, 3),
        "uniqueness_score": round(uniqueness, 3),
        "coherence_score": round(coherence, 3),
        "boilerplate_score": round(boilerplate, 3),
        "quality_score": round(composite, 3),
    }


def filter_low_quality(
    chunks: list[dict[str, Any]],
    *,
    min_score: float = MIN_QUALITY_SCORE,
    annotate: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score all chunks and split into kept and rejected.

    Parameters
    ----------
    chunks : list of chunk dicts
    min_score : float
        Minimum composite quality score to keep.
    annotate : bool
        If True, add ``quality_scores`` to each chunk's metadata.

    Returns
    -------
    (kept_chunks, rejected_chunks)
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for chunk in chunks:
        scores = score_chunk(chunk)
        if annotate:
            chunk.setdefault("metadata", {})["quality_scores"] = scores

        if scores["quality_score"] >= min_score:
            kept.append(chunk)
        else:
            rejected.append(chunk)

    return kept, rejected

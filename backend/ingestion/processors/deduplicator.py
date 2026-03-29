"""Near-duplicate chunk detection and removal.

Uses MinHash fingerprinting with Jaccard similarity to identify semantically
redundant chunks that waste retrieval slots.  Keeps the chunk with the
richest metadata (more fields, longer content) when duplicates are found.

Real-world scenarios handled:
  - Overlapping text from recursive chunking overlap regions
  - Table summary + row chunk duplication
  - OCR + native text overlap on partially-scanned pages
  - Header/footer text appearing in multiple page-level chunks
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ── MinHash parameters ────────────────────────────────────────────────────────
_NUM_HASHES = 128         # number of hash functions (more → more accurate)
_SHINGLE_SIZE = 3         # word n-gram size for shingling
_DEDUP_THRESHOLD = 0.85   # Jaccard similarity above which two chunks are "duplicates"


def _tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenization, stripping punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def _shingles(tokens: list[str], k: int = _SHINGLE_SIZE) -> set[str]:
    """Create a set of k-word shingles from token list."""
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhash_signature(shingle_set: set[str], num_hashes: int = _NUM_HASHES) -> list[int]:
    """Compute a MinHash signature vector for a set of shingles."""
    if not shingle_set:
        return [0] * num_hashes

    max_val = (1 << 32) - 1
    signature = [max_val] * num_hashes

    for shingle in shingle_set:
        shingle_bytes = shingle.encode("utf-8")
        for i in range(num_hashes):
            # Different hash function per slot: hash(shingle + salt)
            h = int(hashlib.md5(shingle_bytes + i.to_bytes(2, "big")).hexdigest()[:8], 16)
            if h < signature[i]:
                signature[i] = h

    return signature


def _jaccard_from_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _exact_jaccard(tokens_a: list[str], tokens_b: list[str], k: int = _SHINGLE_SIZE) -> float:
    """Exact Jaccard similarity using shingle sets (slower but precise)."""
    set_a = _shingles(tokens_a, k)
    set_b = _shingles(tokens_b, k)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _chunk_richness(chunk: dict[str, Any]) -> float:
    """Score a chunk's metadata richness — prefer to keep richer chunks."""
    score = len(chunk.get("content", ""))
    meta = chunk.get("metadata", {})
    if meta.get("caption"):
        score += 500
    if meta.get("caption_label"):
        score += 300
    if meta.get("references"):
        score += 100
    if meta.get("links"):
        score += 50 * len(meta["links"])
    if chunk.get("section_heading"):
        score += 200
    if chunk.get("parent_id"):
        score += 100
    return score


# ── Public API ────────────────────────────────────────────────────────────────

def deduplicate_chunks(
    chunks: list[dict[str, Any]],
    *,
    threshold: float = _DEDUP_THRESHOLD,
    use_minhash: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Remove near-duplicate chunks.

    Parameters
    ----------
    chunks : list of chunk dicts
        Each must have a ``content`` key.
    threshold : float
        Jaccard similarity above which two chunks are considered duplicates.
    use_minhash : bool
        If True, use MinHash for O(n^2) approximate comparison.
        If False, use exact Jaccard (slower but perfectly accurate).

    Returns
    -------
    (deduplicated_chunks, num_removed)
    """
    if not chunks:
        return [], 0

    n = len(chunks)
    # Pre-compute tokens and signatures
    tokens_list = [_tokenize(c.get("content", "")) for c in chunks]

    if use_minhash:
        signatures = [
            _minhash_signature(_shingles(tokens)) for tokens in tokens_list
        ]

    # Track which chunks to keep
    removed: set[int] = set()

    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue

            # Quick length filter — very different lengths = not duplicate
            len_i = len(tokens_list[i])
            len_j = len(tokens_list[j])
            if len_i == 0 or len_j == 0:
                continue
            ratio = min(len_i, len_j) / max(len_i, len_j)
            if ratio < 0.5:
                continue

            # Compute similarity
            if use_minhash:
                sim = _jaccard_from_minhash(signatures[i], signatures[j])
            else:
                sim = _exact_jaccard(tokens_list[i], tokens_list[j])

            if sim >= threshold:
                # Keep the chunk with richer metadata
                if _chunk_richness(chunks[i]) >= _chunk_richness(chunks[j]):
                    removed.add(j)
                else:
                    removed.add(i)
                    break  # i is removed, no need to compare further

    deduplicated = [c for idx, c in enumerate(chunks) if idx not in removed]
    return deduplicated, len(removed)


def exact_content_dedup(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove chunks with exactly identical content (fast, O(n))."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    removed = 0

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if content in seen:
            removed += 1
            continue
        seen.add(content)
        unique.append(chunk)

    return unique, removed

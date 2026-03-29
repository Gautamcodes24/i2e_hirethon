"""Text cleaning and normalization for PDF-extracted text.

Handles real-world PDF extraction artefacts:
  - Unicode normalization (ligatures, smart quotes, special chars)
  - Hyphenated word rejoining across line breaks
  - Repeated header / footer detection and removal
  - Page number stripping
  - Watermark text removal
  - Whitespace normalization
  - Control character removal
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

# ── Unicode replacements ──────────────────────────────────────────────────────
_UNICODE_MAP: dict[str, str] = {
    "\u2018": "'", "\u2019": "'",       # smart single quotes
    "\u201c": '"', "\u201d": '"',       # smart double quotes
    "\u2013": "-", "\u2014": "-",       # en/em dashes
    "\u2026": "...",                     # ellipsis
    "\ufb01": "fi", "\ufb02": "fl",     # fi/fl ligatures
    "\ufb00": "ff", "\ufb03": "ffi",    # ff/ffi ligatures
    "\ufb04": "ffl",                    # ffl ligature
    "\u00a0": " ",                       # non-breaking space
    "\u200b": "",                        # zero-width space
    "\u200c": "", "\u200d": "",         # zero-width non-joiner/joiner
    "\ufeff": "",                        # byte order mark
    "\u00ad": "",                        # soft hyphen
    "\u2022": "- ",                      # bullet → dash
    "\u25cf": "- ",                      # black circle → dash
    "\u25cb": "- ",                      # white circle → dash
    "\u25aa": "- ",                      # small black square → dash
    "\u2023": "- ",                      # triangular bullet → dash
    "\u00b7": "- ",                      # middle dot (bullet) → dash
}

_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_MAP))

# ── Regex patterns ────────────────────────────────────────────────────────────
_HYPHEN_REJOIN = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_NUMBER = re.compile(
    r"(?:^|\n)\s*(?:Page\s+)?\d{1,4}\s*(?:of\s+\d{1,4})?\s*(?:\n|$)",
    re.IGNORECASE,
)
_WATERMARK = re.compile(
    r"(?:DRAFT|CONFIDENTIAL|DO NOT DISTRIBUTE|INTERNAL USE ONLY|SAMPLE|UNCONTROLLED COPY)",
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply all text cleaning steps to *text*."""
    if not text:
        return ""

    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = rejoin_hyphenated_words(text)
    text = normalize_whitespace(text)
    text = strip_page_numbers(text)
    text = strip_watermarks(text)

    return text.strip()


def normalize_unicode(text: str) -> str:
    """NFKC normalize and replace common PDF-specific Unicode chars."""
    text = unicodedata.normalize("NFKC", text)
    text = _UNICODE_RE.sub(lambda m: _UNICODE_MAP[m.group()], text)
    return text


def remove_control_chars(text: str) -> str:
    """Remove ASCII control characters (except newline, tab, carriage return)."""
    return _CONTROL_CHARS.sub("", text)


def rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words split by a hyphen at line break: 'sys-\\ntem' → 'system'."""
    return _HYPHEN_REJOIN.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to one; cap consecutive newlines at 2."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text


def strip_page_numbers(text: str) -> str:
    """Remove standalone page number lines (e.g. 'Page 42', '42 of 300')."""
    return _PAGE_NUMBER.sub("\n", text)


def strip_watermarks(text: str) -> str:
    """Remove common watermark text patterns."""
    return _WATERMARK.sub("", text)


# ── Header / Footer removal ──────────────────────────────────────────────────

def detect_repeated_headers_footers(
    page_texts: Sequence[str],
    *,
    n_lines: int = 3,
    min_occurrences: float = 0.3,
) -> tuple[list[str], list[str]]:
    """Detect text lines that appear as headers/footers across many pages.

    Parameters
    ----------
    page_texts : sequence of str
        Raw text extracted from each page.
    n_lines : int
        Number of lines from the top/bottom of each page to inspect.
    min_occurrences : float
        Fraction of pages a line must appear on to be considered a header/footer
        (0.3 = 30% of pages).

    Returns
    -------
    tuple of (header_patterns, footer_patterns)
        Each is a list of regex-escaped strings to match and remove.
    """
    if not page_texts:
        return [], []

    threshold = max(3, int(len(page_texts) * min_occurrences))

    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()

    for text in page_texts:
        lines = text.strip().split("\n")
        top_lines = lines[:n_lines]
        bot_lines = lines[-n_lines:] if len(lines) > n_lines else []

        for line in top_lines:
            normalized = _normalize_header_line(line)
            if normalized and len(normalized) > 3:
                header_counter[normalized] += 1

        for line in bot_lines:
            normalized = _normalize_header_line(line)
            if normalized and len(normalized) > 3:
                footer_counter[normalized] += 1

    headers = [pat for pat, count in header_counter.items() if count >= threshold]
    footers = [pat for pat, count in footer_counter.items() if count >= threshold]

    return headers, footers


def remove_headers_footers(text: str, headers: list[str], footers: list[str]) -> str:
    """Remove detected header/footer text from a page."""
    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        normalized = _normalize_header_line(line)
        if normalized in headers or normalized in footers:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _normalize_header_line(line: str) -> str:
    """Normalize a line for header/footer matching (strip numbers, whitespace)."""
    line = line.strip()
    # Replace page numbers with placeholder to match across pages
    line = re.sub(r"\b\d{1,4}\b", "#", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line

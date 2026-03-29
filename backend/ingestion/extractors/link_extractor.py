"""Hyperlink and reference / citation extraction from PDF pages.

Uses PyMuPDF to pull embedded hyperlinks and regex patterns to detect
in-text citation markers such as ``[1]``, ``(Author, Year)``, and
footnote superscripts.
"""

from __future__ import annotations

import re
from typing import Any

import fitz  # PyMuPDF

# ── Citation / reference regex patterns ───────────────────────────────────────
_PATTERNS: dict[str, re.Pattern[str]] = {
    # Bracketed numeric: [1], [12], [1,2,3], [1-5]
    "bracket_numeric": re.compile(r"\[(\d+(?:[,\-–]\s*\d+)*)\]"),
    # Author-year: (Smith, 2020), (Smith & Jones, 2019), (Smith et al., 2021)
    "author_year": re.compile(
        r"\(([A-Z][a-z]+(?:\s(?:&|and|et\s+al\.?)\s+[A-Z][a-z]+)?,\s*\d{4}[a-z]?)\)"
    ),
    # Footnote superscript markers: a bare number at the start of a line that
    # looks like a footnote (1-3 digits, followed by text).
    "footnote": re.compile(r"(?:^|\s)(\d{1,3})\s+(?=[A-Z])"),
}


def extract_links(page: fitz.Page, page_num: int) -> list[dict[str, Any]]:
    """Return hyperlinks found on *page*.

    Each dict contains:
        uri  – the URL (for external links) or ``None``
        kind – link type string (``"uri"``, ``"goto"``, ``"named"``, ``"launch"``)
        dest_page – target page number for internal links, else ``None``
        page – source page number
        bbox – (x0, y0, x1, y1) link rectangle
    """
    results: list[dict[str, Any]] = []
    for link in page.get_links():
        kind_code = link.get("kind", -1)
        kind_map = {0: "none", 1: "goto", 2: "uri", 3: "launch", 5: "named"}
        kind_str = kind_map.get(kind_code, "unknown")

        uri = link.get("uri")
        dest_page = link.get("page")  # internal link target (0-based)
        rect = link.get("from")  # fitz.Rect

        bbox = None
        if rect:
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

        results.append({
            "uri": uri,
            "kind": kind_str,
            "dest_page": (dest_page + 1) if dest_page is not None and dest_page >= 0 else None,
            "page": page_num,
            "bbox": bbox,
        })
    return results


def extract_references(text: str) -> list[dict[str, str]]:
    """Detect in-text citation markers in *text*.

    Returns a list of dicts with ``pattern`` (name) and ``match`` (the
    captured group text).
    """
    refs: list[dict[str, str]] = []
    seen: set[str] = set()

    for name, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            val = m.group(1).strip()
            key = f"{name}:{val}"
            if key not in seen:
                seen.add(key)
                refs.append({"pattern": name, "match": val})
    return refs


def links_for_page_region(
    links: list[dict[str, Any]],
    bbox: tuple[float, float, float, float] | None = None,
) -> list[str]:
    """Return a flat list of URI strings from *links* that fall within *bbox*.

    If *bbox* is ``None`` all URIs are returned.
    """
    uris: list[str] = []
    for lnk in links:
        if lnk.get("uri") is None:
            continue
        if bbox is None:
            uris.append(lnk["uri"])
            continue
        lnk_bbox = lnk.get("bbox")
        if lnk_bbox and _bbox_overlaps(lnk_bbox, bbox):
            uris.append(lnk["uri"])
    return uris


def _bbox_overlaps(a: tuple, b: tuple) -> bool:
    """True if bounding boxes *a* and *b* overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

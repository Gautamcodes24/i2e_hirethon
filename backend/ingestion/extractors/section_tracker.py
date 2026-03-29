"""Section-heading detection and chunk → heading assignment.

Scans page text for common heading patterns (numbered sections, uppercase
titles, chapter markers) and maintains a running index so that every chunk
can be tagged with the most-recent section heading for citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Heading regex patterns (ordered by specificity) ───────────────────────────
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # Numbered sections: "1.2.3 Title …"
    re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$", re.MULTILINE),
    # Chapter markers: "Chapter 3 – Title"
    re.compile(r"^(Chapter\s+\d+)[:\s\-–—]*(.{0,80})$", re.MULTILINE | re.IGNORECASE),
    # Appendix markers: "Appendix A – Title"
    re.compile(r"^(Appendix\s+[A-Z0-9]+)[:\s\-–—]*(.{0,80})$", re.MULTILINE | re.IGNORECASE),
    # ALL-CAPS titles (≥ 4 words, ≤ 80 chars, not just a short label)
    re.compile(r"^([A-Z][A-Z\s,&\-]{8,80})$", re.MULTILINE),
]


@dataclass
class Heading:
    """A detected section heading."""
    page: int
    title: str
    number: str = ""           # e.g. "2.3.1" if available
    char_offset: int = 0       # character position on the page

    def label(self) -> str:
        """Human-readable heading label for citations."""
        if self.number:
            return f"§{self.number} {self.title}".strip()
        return self.title.strip()


@dataclass
class SectionIndex:
    """Accumulated heading index built page-by-page."""
    headings: list[Heading] = field(default_factory=list)

    # ── Building ──────────────────────────────────────────────────────────
    def add_page(self, page_text: str, page_num: int) -> list[Heading]:
        """Extract headings from *page_text* and append them to the index.

        Returns the headings found on this page.
        """
        found = extract_headings_from_text(page_text, page_num)
        self.headings.extend(found)
        return found

    # ── Querying ──────────────────────────────────────────────────────────
    def heading_for(self, page: int, char_offset: int = 0) -> str:
        """Return the label of the most-recent heading at or before *page*.

        If *char_offset* is given and headings exist on the same page, the
        most recent heading whose offset is ≤ *char_offset* is preferred.
        """
        best: Heading | None = None
        for h in self.headings:
            if h.page > page:
                break
            if h.page == page and h.char_offset > char_offset:
                continue
            best = h
        return best.label() if best else ""

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {"page": h.page, "number": h.number, "title": h.title, "label": h.label()}
            for h in self.headings
        ]


# ── Free functions ────────────────────────────────────────────────────────────

def extract_headings_from_text(page_text: str, page_num: int) -> list[Heading]:
    """Return all detected headings in *page_text*, sorted by character offset."""
    if not page_text:
        return []

    seen_offsets: set[int] = set()
    results: list[Heading] = []

    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(page_text):
            offset = m.start()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            groups = m.groups()
            if len(groups) == 2:
                number_or_label, title = groups
                # Numbered section?
                if re.match(r"^\d+(\.\d+)*$", number_or_label.strip()):
                    results.append(Heading(
                        page=page_num,
                        title=title.strip(),
                        number=number_or_label.strip(),
                        char_offset=offset,
                    ))
                else:
                    # Chapter / Appendix / other labelled heading
                    full_title = f"{number_or_label} {title}".strip() if title.strip() else number_or_label.strip()
                    results.append(Heading(
                        page=page_num,
                        title=full_title,
                        char_offset=offset,
                    ))
            else:
                # ALL-CAPS single-group match
                results.append(Heading(
                    page=page_num,
                    title=groups[0].strip().title(),  # Title-case for readability
                    char_offset=offset,
                ))

    results.sort(key=lambda h: h.char_offset)
    return results

"""Advanced PDF parser — universal ingestion pipeline (v2).

Orchestrates:
  - **PyMuPDF** (fitz) for native text, images, links, and page rendering
  - **pdfplumber** for layout-aware table detection and multi-page stitching
  - **Tesseract OCR** for scanned / image-only pages
  - **Groq Vision** for generating textual descriptions of diagrams/images
  - **Recursive chunking** with parent-child hierarchy for small-to-big retrieval
  - **Section heading tracking** for rich citation metadata
  - **Text cleaning** (Unicode, hyphen rejoining, header/footer removal)
  - **Caption detection** for Figure/Table captions
  - **Document structure** analysis (page classification, TOC parsing)
  - **Quality scoring** and filtering of low-quality chunks
  - **Near-duplicate removal** via MinHash fingerprinting
  - **Vector diagram detection** for PDF-path-drawn figures

Produces a flat list of chunks, each annotated with:
    type, page, section_heading, content, quality_scores, parent_id, and metadata dict.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from backend.ingestion.extractors.caption_extractor import (
    Caption,
    captions_to_list,
    extract_captions,
    link_captions_to_images,
    link_captions_to_tables,
)
from backend.ingestion.chunkers.chunker import recursive_chunk
from backend.ingestion.processors.deduplicator import deduplicate_chunks, exact_content_dedup
from backend.ingestion.processors.doc_structure import DocumentStructure, analyze_document_structure
from backend.ingestion.extractors.image_extractor import (
    describe_images_batch,
    detect_vector_diagrams,
    extract_images,
)
from backend.ingestion.extractors.link_extractor import extract_links, extract_references, links_for_page_region
from backend.ingestion.extractors.ocr import is_scanned_page, ocr_page
from backend.ingestion.chunkers.parent_child_chunker import create_parent_child_chunks
from backend.ingestion.parser import (
    COLUMN_SNAP_TOL,
    MIN_ROW_CELLS,
    MIN_TEXT_LENGTH,
    TableObject,
    _clean_cell,
    _columns_match,
    _extract_col_xs,
    _validate_row,
)
from backend.ingestion.processors.quality_scorer import filter_low_quality
from backend.ingestion.extractors.section_tracker import SectionIndex
from backend.ingestion.processors.text_cleaner import (
    clean_text,
    detect_repeated_headers_footers,
    remove_headers_footers,
)

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
ENABLE_OCR = True
ENABLE_VISION = True
ENABLE_PARENT_CHILD = True
ENABLE_QUALITY_FILTER = True
ENABLE_DEDUP = True
ENABLE_TEXT_CLEANING = True
ENABLE_CAPTIONS = True
ENABLE_DOC_STRUCTURE = True
ENABLE_VECTOR_DETECTION = True
QUALITY_MIN_SCORE = 0.30


@dataclass
class ParseStats:
    """Counters collected during a parse run."""
    total_pages: int = 0
    native_text_pages: int = 0
    ocr_pages: int = 0
    skipped_pages: int = 0            # non-content pages (cover, TOC, blank)
    tables_detected: int = 0
    tables_with_captions: int = 0
    images_extracted: int = 0
    images_described: int = 0
    vector_diagrams_detected: int = 0
    figures_with_captions: int = 0
    links_found: int = 0
    captions_found: int = 0
    headers_removed: int = 0          # repeated headers/footers stripped
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    ocr_chunks: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    total_before_dedup: int = 0
    exact_dupes_removed: int = 0
    near_dupes_removed: int = 0
    low_quality_removed: int = 0
    total_chunks: int = 0


# ── Main parser ───────────────────────────────────────────────────────────────

class AdvancedPDFParser:
    """Universal PDF ingestion pipeline (v2).

    Usage::

        parser = AdvancedPDFParser(groq_api_key="…")
        result = parser.ingest("path/to/doc.pdf")
        chunks        = result["chunks"]           # ready for embedding
        parent_chunks = result["parent_chunks"]    # large context chunks (if enabled)
        stats         = result["stats"]            # ParseStats summary
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        *,
        enable_ocr: bool = ENABLE_OCR,
        enable_vision: bool = ENABLE_VISION,
        enable_parent_child: bool = ENABLE_PARENT_CHILD,
        enable_quality_filter: bool = ENABLE_QUALITY_FILTER,
        enable_dedup: bool = ENABLE_DEDUP,
        enable_text_cleaning: bool = ENABLE_TEXT_CLEANING,
        enable_captions: bool = ENABLE_CAPTIONS,
        enable_doc_structure: bool = ENABLE_DOC_STRUCTURE,
        enable_vector_detection: bool = ENABLE_VECTOR_DETECTION,
        quality_min_score: float = QUALITY_MIN_SCORE,
        chunk_max_tokens: int = CHUNK_MAX_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        ocr_dpi: int = 300,
        ocr_lang: str = "eng",
        vision_delay: float = 1.0,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.enable_ocr = enable_ocr
        self.enable_vision = enable_vision and bool(self.groq_api_key)
        self.enable_parent_child = enable_parent_child
        self.enable_quality_filter = enable_quality_filter
        self.enable_dedup = enable_dedup
        self.enable_text_cleaning = enable_text_cleaning
        self.enable_captions = enable_captions
        self.enable_doc_structure = enable_doc_structure
        self.enable_vector_detection = enable_vector_detection
        self.quality_min_score = quality_min_score
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang
        self.vision_delay = vision_delay

        # Internal state (reset per ingest call)
        self._tables: list[TableObject] = []
        self._page_texts: list[dict[str, Any]] = []
        self._image_chunks: list[dict[str, Any]] = []
        self._all_links: list[dict[str, Any]] = []
        self._all_captions: list[Caption] = []
        self._section_index = SectionIndex()
        self._doc_structure: DocumentStructure | None = None
        self._raw_page_texts: list[str] = []      # for header/footer detection
        self._header_patterns: list[str] = []
        self._footer_patterns: list[str] = []
        self._parent_chunks: list[dict[str, Any]] = []
        self._table_counter = 0
        self._stats = ParseStats()

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str, on_progress=None) -> dict[str, Any]:
        """Parse *pdf_path* and return a dict with ``chunks``, ``parent_chunks``,
        ``tables``, ``stats``, ``section_index``, ``captions``, ``doc_structure``.
        """
        self._reset()

        fitz_doc = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(pdf_path)

        num_pages = len(fitz_doc)
        self._stats.total_pages = num_pages
        logger.info("Parsing %d pages from %s", num_pages, pdf_path)

        # ── Phase 0: Quick first pass — extract raw text for structure analysis
        for page_idx in range(num_pages):
            fitz_page = fitz_doc[page_idx]
            raw_text = (fitz_page.get_text("text") or "").strip()
            self._raw_page_texts.append(raw_text)
            if on_progress:
                on_progress(f"Scanning page {page_idx + 1} of {num_pages}")

        # ── Phase 1: Document structure analysis
        if self.enable_doc_structure:
            self._doc_structure = analyze_document_structure(self._raw_page_texts)
            skipped = sum(1 for p in self._doc_structure.pages if not p.is_content)
            self._stats.skipped_pages = skipped
            logger.info("Document structure: %s", self._doc_structure.summary())

        # ── Phase 2: Detect repeated headers/footers
        if self.enable_text_cleaning:
            self._header_patterns, self._footer_patterns = detect_repeated_headers_footers(
                self._raw_page_texts
            )
            self._stats.headers_removed = len(self._header_patterns) + len(self._footer_patterns)
            if self._header_patterns or self._footer_patterns:
                logger.info(
                    "Detected %d header patterns, %d footer patterns",
                    len(self._header_patterns), len(self._footer_patterns),
                )

        # ── Phase 3: Per-page extraction
        all_page_images: list[dict[str, Any]] = []

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            fitz_page = fitz_doc[page_idx]
            plumber_page = plumber_doc.pages[page_idx] if page_idx < len(plumber_doc.pages) else None

            # Skip non-content pages if doc structure enabled
            if self.enable_doc_structure and self._doc_structure:
                if not self._doc_structure.is_content_page(page_num):
                    logger.debug("Skipping non-content page %d", page_num)
                    continue

            self._process_page(fitz_page, plumber_page, page_num, fitz_doc, all_page_images)
            if on_progress:
                on_progress(f"Extracting page {page_num} of {num_pages}")

        # ── Phase 4: Vision descriptions (batched)
        if self.enable_vision and all_page_images:
            logger.info("Describing %d images via Groq Vision …", len(all_page_images))
            if on_progress:
                on_progress(f"Describing {len(all_page_images)} images via Vision...")
            describe_images_batch(
                all_page_images,
                self.groq_api_key,
                delay=self.vision_delay,
            )
            self._stats.images_described = sum(1 for img in all_page_images if img.get("description"))
            for img in all_page_images:
                desc = img.get("description", "")
                if desc:
                    # Enrich with caption if available
                    caption_label = img.get("caption_label", "")
                    if caption_label:
                        desc = f"{caption_label}\n\n{desc}"

                    self._image_chunks.append({
                        "content": desc,
                        "type": "image",
                        "page": img["page"],
                        "metadata": {
                            "pages": [img["page"]],
                            "image_b64": img.get("image_b64", ""),
                            "bbox": img.get("bbox"),
                            "source": "vision_llm",
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "is_vector": img.get("is_vector", False),
                            "caption": img.get("caption", ""),
                            "caption_label": img.get("caption_label", ""),
                            "figure_number": img.get("figure_number", ""),
                        },
                    })

        # ── Phase 5: Link captions to tables
        if self.enable_captions and self._all_captions:
            table_dicts = [t.to_dict() for t in self._tables]
            link_captions_to_tables(self._all_captions, table_dicts)
            # Propagate captions back to TableObjects
            for td, tbl in zip(table_dicts, self._tables):
                if td.get("caption"):
                    tbl.metadata["caption"] = td["caption"]
                    tbl.metadata["caption_label"] = td.get("caption_label", "")
                    tbl.metadata["table_number"] = td.get("table_number", "")
                    self._stats.tables_with_captions += 1

            link_captions_to_images(self._all_captions, all_page_images)
            self._stats.figures_with_captions = sum(1 for img in all_page_images if img.get("caption"))

        plumber_doc.close()
        fitz_doc.close()

        # ── Phase 6: Build chunks
        chunks = self._build_chunks()
        self._stats.total_before_dedup = len(chunks)

        # ── Phase 7: Quality filtering
        rejected: list[dict[str, Any]] = []
        if self.enable_quality_filter:
            chunks, rejected = filter_low_quality(chunks, min_score=self.quality_min_score)
            self._stats.low_quality_removed = len(rejected)
            logger.info("Quality filter: kept %d, rejected %d", len(chunks), len(rejected))

        # ── Phase 8: Deduplication
        if self.enable_dedup:
            chunks, exact_removed = exact_content_dedup(chunks)
            self._stats.exact_dupes_removed = exact_removed

            chunks, near_removed = deduplicate_chunks(chunks)
            self._stats.near_dupes_removed = near_removed
            logger.info("Dedup: removed %d exact + %d near-duplicates", exact_removed, near_removed)

        self._stats.total_chunks = len(chunks)

        return {
            "chunks": chunks,
            "parent_chunks": self._parent_chunks,
            "tables": [t.to_dict() for t in self._tables],
            "stats": self._stats,
            "section_index": self._section_index.to_list(),
            "captions": captions_to_list(self._all_captions),
            "doc_structure": self._doc_structure.summary() if self._doc_structure else {},
            "rejected_chunks": rejected,
        }

    # ── Per-page processing ───────────────────────────────────────────────

    def _process_page(
        self,
        fitz_page: fitz.Page,
        plumber_page,
        page_num: int,
        fitz_doc: fitz.Document,
        all_page_images: list[dict[str, Any]],
    ):
        """Process a single page: OCR check → tables → images → text → links → headings → captions."""

        # 1  OCR check
        is_scanned = False
        ocr_text = ""
        if self.enable_ocr and is_scanned_page(fitz_page):
            is_scanned = True
            self._stats.ocr_pages += 1
            ocr_text = ocr_page(fitz_page, dpi=self.ocr_dpi, lang=self.ocr_lang)
            logger.info("OCR page %d: %d chars", page_num, len(ocr_text))
        else:
            self._stats.native_text_pages += 1

        # 2  Table detection via pdfplumber
        table_bboxes: list[tuple] = []
        if plumber_page is not None and not is_scanned:
            table_bboxes = self._detect_tables(plumber_page, page_num)

        # 3  Image extraction via PyMuPDF
        if not is_scanned:
            page_images = extract_images(fitz_page, page_num, fitz_doc)
            self._stats.images_extracted += len(page_images)
            all_page_images.extend(page_images)

            # 3b  Vector diagram detection
            if self.enable_vector_detection:
                vector_imgs = detect_vector_diagrams(
                    fitz_page, page_num,
                    min_drawings=5,
                    min_raster_images=len(page_images),
                )
                if vector_imgs:
                    self._stats.vector_diagrams_detected += len(vector_imgs)
                    all_page_images.extend(vector_imgs)

        # 4  Text extraction (non-table, non-image regions)
        if is_scanned:
            page_text = ocr_text
        else:
            page_text = self._extract_text_outside_tables(plumber_page, table_bboxes) if plumber_page else ""

        # 4b  Text cleaning
        if self.enable_text_cleaning and page_text:
            page_text = clean_text(page_text)
            if self._header_patterns or self._footer_patterns:
                page_text = remove_headers_footers(page_text, self._header_patterns, self._footer_patterns)

        if page_text and len(page_text.strip()) >= MIN_TEXT_LENGTH:
            source = "ocr" if is_scanned else "native"
            self._page_texts.append({
                "page": page_num,
                "text": page_text.strip(),
                "source": source,
                "has_tables": bool(table_bboxes),
            })

        # 5  Link extraction
        page_links = extract_links(fitz_page, page_num)
        self._all_links.extend(page_links)
        self._stats.links_found += len(page_links)

        # 6  Section heading tracking
        heading_text = page_text if page_text else (ocr_text if ocr_text else "")
        self._section_index.add_page(heading_text, page_num)

        # 7  Caption extraction
        if self.enable_captions:
            caption_text = page_text if page_text else (ocr_text if ocr_text else "")
            page_captions = extract_captions(caption_text, page_num)
            self._all_captions.extend(page_captions)
            self._stats.captions_found += len(page_captions)

    # ── Table detection (reuses existing pdfplumber logic) ────────────────

    def _detect_tables(self, plumber_page, page_num: int) -> list[tuple]:
        """Detect tables on a pdfplumber page, stitch multi-page tables."""
        detected = plumber_page.find_tables()
        detected = sorted(detected, key=lambda t: t.bbox[1])
        table_bboxes: list[tuple] = []

        for pl_table in detected:
            raw_rows = pl_table.extract()
            if not raw_rows:
                continue

            col_xs = _extract_col_xs(pl_table)
            rows = [[_clean_cell(c) for c in row] for row in raw_rows]
            header = rows[0] if rows else []
            data_rows = [r for r in rows[1:] if _validate_row(r)]

            if not data_rows:
                continue

            table_bottom = pl_table.bbox[3]
            table_bboxes.append(pl_table.bbox)
            self._stats.tables_detected += 1

            continuation = self._find_continuation(col_xs, page_num)
            if continuation:
                continuation.rows.extend(data_rows)
                if page_num not in continuation.pages:
                    continuation.pages.append(page_num)
                continuation.last_bottom = table_bottom
            else:
                self._table_counter += 1
                table_id = f"p{page_num}_t{self._table_counter}"
                new_table = TableObject(
                    table_id=table_id,
                    pages=[page_num],
                    header=header,
                    rows=data_rows,
                    col_xs=col_xs,
                    last_bottom=table_bottom,
                    metadata={"bbox_first_page": pl_table.bbox},
                )
                self._tables.append(new_table)

        return table_bboxes

    def _find_continuation(self, col_xs: list[float], current_page: int) -> TableObject | None:
        for tbl in reversed(self._tables):
            last_page = tbl.pages[-1]
            if last_page < current_page - 1:
                break
            if _columns_match(
                tbl.col_signature(),
                [round(x / COLUMN_SNAP_TOL) * COLUMN_SNAP_TOL for x in col_xs],
            ):
                return tbl
        return None

    # ── Text extraction outside table bboxes ──────────────────────────────

    @staticmethod
    def _extract_text_outside_tables(plumber_page, table_bboxes: list[tuple]) -> str:
        if not table_bboxes:
            return (plumber_page.extract_text() or "").strip()

        page_height = plumber_page.height
        page_width = plumber_page.width

        boundaries = sorted(
            {0.0}
            | {bb[1] for bb in table_bboxes}
            | {bb[3] for bb in table_bboxes}
            | {page_height}
        )
        text_parts: list[str] = []

        for i in range(len(boundaries) - 1):
            top = boundaries[i]
            bottom = boundaries[i + 1]
            covered = any(bb[1] <= top and bb[3] >= bottom for bb in table_bboxes)
            if covered or (bottom - top) < 4:
                continue
            try:
                cropped = plumber_page.crop((0, top, page_width, bottom))
                txt = (cropped.extract_text() or "").strip()
                if txt:
                    text_parts.append(txt)
            except Exception:
                pass

        return "\n".join(text_parts)

    # ── Chunk assembly ────────────────────────────────────────────────────

    def _build_chunks(self) -> list[dict[str, Any]]:
        """Assemble all extracted content into flat, enriched chunks."""
        chunks: list[dict[str, Any]] = []

        # ── Table chunks (full-table summary + per-row) ───────────────────
        for tbl in self._tables:
            section = self._section_index.heading_for(tbl.pages[0])
            page_links = [l for l in self._all_links if l["page"] in tbl.pages]
            link_uris = [l["uri"] for l in page_links if l.get("uri")]
            doc_region = self._get_doc_region(tbl.pages[0])

            caption = tbl.metadata.get("caption", "")
            caption_label = tbl.metadata.get("caption_label", "")

            # Full-table summary chunk (header + all rows as markdown-style table)
            full_content = self._table_to_text(tbl)
            if caption_label:
                full_content = f"{caption_label}\n\n{full_content}"

            chunks.append({
                "content": full_content,
                "type": "table",
                "page": tbl.pages[0],
                "section_heading": section,
                "metadata": {
                    "table_id": tbl.table_id,
                    "header": tbl.header,
                    "pages": tbl.pages,
                    "row": -1,  # -1 signals full-table summary
                    "links": link_uris[:5],
                    "source": "native",
                    "caption": caption,
                    "caption_label": caption_label,
                    "table_number": tbl.metadata.get("table_number", ""),
                    "doc_region": doc_region,
                },
            })
            self._stats.table_chunks += 1

            # Per-row chunks (for fine-grained retrieval)
            for row_idx, row in enumerate(tbl.rows):
                row_content = " | ".join(row)
                # Prepend header for context
                header_line = " | ".join(tbl.header)
                content_with_header = f"{header_line}\n{row_content}"
                if caption_label:
                    content_with_header = f"{caption_label}\n{content_with_header}"
                chunks.append({
                    "content": content_with_header,
                    "type": "table",
                    "page": tbl.pages[0],
                    "section_heading": section,
                    "metadata": {
                        "table_id": tbl.table_id,
                        "header": tbl.header,
                        "row": row_idx,
                        "pages": tbl.pages,
                        "links": [],
                        "source": "native",
                        "caption": caption,
                        "doc_region": doc_region,
                    },
                })
                self._stats.table_chunks += 1

        # ── Text chunks (parent-child or flat recursive) ──────────────────
        for page_info in self._page_texts:
            page_num = page_info["page"]
            text = page_info["text"]
            source = page_info.get("source", "native")
            section = self._section_index.heading_for(page_num)
            doc_region = self._get_doc_region(page_num)

            page_links = [l for l in self._all_links if l["page"] == page_num]
            link_uris = [l["uri"] for l in page_links if l.get("uri")]
            references = extract_references(text)

            base_meta = {
                "links": link_uris[:5],
                "references": references[:10],
                "doc_region": doc_region,
            }

            chunk_type = "ocr_text" if source == "ocr" else "text"

            if self.enable_parent_child:
                # Hierarchical: parent (large) + child (small) chunks
                parents, children = create_parent_child_chunks(
                    text,
                    page_num=page_num,
                    section_heading=section,
                    source=source,
                    base_metadata=base_meta,
                    parent_max_tokens=1024,
                    parent_overlap_tokens=100,
                    child_max_tokens=self.chunk_max_tokens,
                    child_overlap_tokens=self.chunk_overlap_tokens,
                )

                for p in parents:
                    p["type"] = chunk_type
                    self._parent_chunks.append(p)
                    self._stats.parent_chunks += 1

                for c in children:
                    c["type"] = chunk_type
                    chunks.append(c)
                    if chunk_type == "ocr_text":
                        self._stats.ocr_chunks += 1
                    else:
                        self._stats.text_chunks += 1
                    self._stats.child_chunks += 1

            else:
                # Flat recursive chunking (original behavior)
                sub_chunks = recursive_chunk(
                    text,
                    max_tokens=self.chunk_max_tokens,
                    overlap_tokens=self.chunk_overlap_tokens,
                )

                for sc in sub_chunks:
                    sc_section = self._section_index.heading_for(page_num, sc["start_char"])
                    chunks.append({
                        "content": sc["content"],
                        "type": chunk_type,
                        "page": page_num,
                        "section_heading": sc_section or section,
                        "metadata": {
                            "pages": [page_num],
                            "start_char": sc["start_char"],
                            "end_char": sc["end_char"],
                            **base_meta,
                            "source": source,
                        },
                    })
                    if chunk_type == "ocr_text":
                        self._stats.ocr_chunks += 1
                    else:
                        self._stats.text_chunks += 1

        # ── Image description chunks ──────────────────────────────────────
        for img_chunk in self._image_chunks:
            page_num = img_chunk["page"]
            section = self._section_index.heading_for(page_num)
            doc_region = self._get_doc_region(page_num)
            img_chunk["section_heading"] = section
            img_chunk["metadata"]["doc_region"] = doc_region
            chunks.append(img_chunk)
            self._stats.image_chunks += 1

        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_text(tbl: TableObject) -> str:
        """Convert a TableObject to a readable markdown-like text block."""
        lines: list[str] = []
        header_line = " | ".join(tbl.header)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for row in tbl.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    def _get_doc_region(self, page_num: int) -> str:
        """Return the document structural region for a page (body, appendix, etc.)."""
        if self._doc_structure:
            return self._doc_structure.category_for(page_num)
        return "body"

    def _reset(self):
        """Clear state between ingest calls."""
        self._tables = []
        self._page_texts = []
        self._image_chunks = []
        self._all_links = []
        self._all_captions = []
        self._section_index = SectionIndex()
        self._doc_structure = None
        self._raw_page_texts = []
        self._header_patterns = []
        self._footer_patterns = []
        self._parent_chunks = []
        self._table_counter = 0
        self._stats = ParseStats()

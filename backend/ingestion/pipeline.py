"""Top-level ingestion pipeline — orchestrates parsing, embedding, and indexing.

This is the single entry point for ingesting a PDF end-to-end.
It wraps the AdvancedPDFParser from `app/` and adds:
  - Centralized config (no hardcoded values)
  - Structured step logging with progress indicators
  - Embedding + FAISS index creation
  - BM25 sparse index creation
  - Artefact persistence (faiss.index, chunks.pkl, parent_chunks.pkl, bm25_corpus.pkl)

Usage:
    from backend.ingestion import IngestPipeline
    result = IngestPipeline().run("data/my_doc.pdf")
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.config import settings
from backend.logger import get_logger, progress_bar, step_logger

logger = get_logger(__name__)


class IngestPipeline:
    """End-to-end PDF → FAISS pipeline with structured logging."""

    def __init__(self, **overrides: Any) -> None:
        """Accepts any Settings field name as a keyword override."""
        self._cfg = settings
        self._overrides = overrides
        self._on_step = overrides.pop("on_step", None)
        self._on_detail = overrides.pop("on_detail", None)

    # ──────────────────────────────────────────────────────────────────────
    def run(self, pdf_path: str) -> dict[str, Any]:
        """Ingest *pdf_path* and persist all artefacts. Returns result dict."""
        tracker = step_logger(logger, total_steps=5, on_step=self._on_step)

        # ── Step 1: Parse PDF ─────────────────────────────────────────────
        tracker.step("Parsing PDF with AdvancedPDFParser")
        from backend.ingestion.advanced_parser import AdvancedPDFParser  # deferred to avoid slow startup

        parser = AdvancedPDFParser(
            groq_api_key=self._overrides.get("groq_api_key", self._cfg.GROQ_API_KEY),
            enable_ocr=self._overrides.get("enable_ocr", self._cfg.ENABLE_OCR),
            enable_vision=self._overrides.get("enable_vision", self._cfg.ENABLE_VISION),
            enable_parent_child=self._overrides.get("enable_parent_child", self._cfg.ENABLE_PARENT_CHILD),
            enable_quality_filter=self._overrides.get("enable_quality_filter", self._cfg.ENABLE_QUALITY_FILTER),
            enable_dedup=self._overrides.get("enable_dedup", self._cfg.ENABLE_DEDUP),
            enable_text_cleaning=self._overrides.get("enable_text_cleaning", self._cfg.ENABLE_TEXT_CLEANING),
            enable_captions=self._overrides.get("enable_captions", self._cfg.ENABLE_CAPTIONS),
            enable_doc_structure=self._overrides.get("enable_doc_structure", self._cfg.ENABLE_DOC_STRUCTURE),
            enable_vector_detection=self._overrides.get("enable_vector_detection", self._cfg.ENABLE_VECTOR_DETECTION),
            quality_min_score=self._overrides.get("quality_min_score", self._cfg.QUALITY_MIN_SCORE),
            chunk_max_tokens=self._overrides.get("chunk_max_tokens", self._cfg.CHUNK_MAX_TOKENS),
            chunk_overlap_tokens=self._overrides.get("chunk_overlap_tokens", self._cfg.CHUNK_OVERLAP_TOKENS),
            vision_delay=self._overrides.get("vision_delay", self._cfg.VISION_DELAY),
            ocr_dpi=self._overrides.get("ocr_dpi", self._cfg.OCR_DPI),
            ocr_lang=self._overrides.get("ocr_lang", self._cfg.OCR_LANG),
        )

        result = parser.ingest(pdf_path, on_progress=self._on_detail)
        chunks = result["chunks"]
        parent_chunks = result["parent_chunks"]
        stats = result["stats"]
        logger.info("Parsed %d pages → %d chunks (%d parents)", stats.total_pages, stats.total_chunks, stats.parent_chunks)

        if not chunks:
            logger.error("No chunks extracted — aborting")
            return {"error": "No chunks extracted from PDF", "stats": stats}

        # ── Step 2: Build embeddings ──────────────────────────────────────
        tracker.step(f"Building embeddings ({self._cfg.EMBEDDING_PROVIDER}, {self._cfg.active_embedding_model})")
        from backend.utils.embeddings import Embeddings

        embedder = Embeddings()
        texts = [c["content"] for c in chunks]
        if self._on_detail:
            self._on_detail(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed(texts, show_progress=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings /= norms

        logger.info("Embeddings: shape=%s, memory=%.1f MB", embeddings.shape, embeddings.nbytes / 1024 / 1024)

        # ── Step 3: Build FAISS index ─────────────────────────────────────
        tracker.step("Building FAISS IndexFlatIP")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        logger.info("FAISS index: %d vectors × %d dims", index.ntotal, dimension)

        # ── Step 4: Build BM25 index ─────────────────────────────────────
        tracker.step("Building BM25 sparse index")
        from rank_bm25 import BM25Okapi

        bm25_corpus = [self._tokenize_bm25(c["content"]) for c in chunks]
        bm25_index = BM25Okapi(bm25_corpus)
        logger.info("BM25 index: %d docs, avg_dl=%.1f", len(bm25_corpus), bm25_index.avgdl)

        # ── Step 5: Save artefacts ────────────────────────────────────────
        tracker.step(f"Saving artefacts to {self._cfg.DATA_DIR}")
        self._save_artefacts(index, chunks, parent_chunks, bm25_corpus)

        tracker.done("Ingestion complete")

        return {
            "chunks": chunks,
            "parent_chunks": parent_chunks,
            "stats": stats,
            "tables": result["tables"],
            "captions": result.get("captions", []),
            "doc_structure": result.get("doc_structure", {}),
            "section_index": result.get("section_index", []),
            "n_chunks": len(chunks),
            "n_parents": len(parent_chunks),
            "embedding_dim": dimension,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize_bm25(text: str) -> list[str]:
        """Tokenizer for BM25 — lowercase, alpha-only, stopword removal."""
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "so", "yet", "both", "either", "neither", "each", "every",
            "all", "any", "few", "more", "most", "other", "some", "such", "no",
            "only", "own", "same", "than", "too", "very", "this", "that", "these",
            "those", "it", "its", "he", "she", "they", "them", "their", "we",
            "us", "our", "you", "your", "i", "my", "me",
        }
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in _STOP and len(t) > 1]

    def _save_artefacts(
        self,
        index: faiss.Index,
        chunks: list[dict],
        parent_chunks: list[dict],
        bm25_corpus: list[list[str]],
    ) -> None:
        faiss.write_index(index, str(self._cfg.INDEX_PATH))

        with self._cfg.CHUNKS_PATH.open("wb") as f:
            pickle.dump(chunks, f)

        with self._cfg.PARENT_CHUNKS_PATH.open("wb") as f:
            pickle.dump(parent_chunks, f)

        with self._cfg.BM25_PATH.open("wb") as f:
            pickle.dump({"corpus": bm25_corpus, "chunks_len": len(chunks)}, f)

        logger.info(
            "Saved: index=%.1fMB, chunks=%.1fMB, parents=%.1fMB, bm25=%.1fMB",
            self._cfg.INDEX_PATH.stat().st_size / 1024 / 1024,
            self._cfg.CHUNKS_PATH.stat().st_size / 1024 / 1024,
            self._cfg.PARENT_CHUNKS_PATH.stat().st_size / 1024 / 1024,
            self._cfg.BM25_PATH.stat().st_size / 1024 / 1024,
        )

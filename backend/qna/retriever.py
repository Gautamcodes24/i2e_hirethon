"""Hybrid retriever — combines dense (FAISS) and sparse (BM25) search.

Fuses scores: final_score = α * dense_cosine + (1-α) * bm25_normalized
Then resolves child chunks → parent chunks for richer LLM context.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Loads saved artefacts and performs hybrid search with parent resolution."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._parent_chunks: list[dict[str, Any]] = []
        self._parent_lookup: dict[str, dict[str, Any]] = {}
        self._bm25_index = None
        self._bm25_corpus: list[list[str]] = []
        self._embedder = None
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._index is not None

    def load(self) -> None:
        """Load FAISS index, chunks, parents, and BM25 from disk."""
        if self._loaded:
            return

        required = [settings.INDEX_PATH, settings.CHUNKS_PATH]
        for path in required:
            if not path.exists():
                logger.warning("Missing artefact: %s — run ingestion first", path)
                return

        logger.info("Loading retrieval artefacts from %s", settings.DATA_DIR)

        self._index = faiss.read_index(str(settings.INDEX_PATH))
        logger.info("FAISS index loaded: %d vectors", self._index.ntotal)

        with settings.CHUNKS_PATH.open("rb") as f:
            self._chunks = pickle.load(f)

        if settings.PARENT_CHUNKS_PATH.exists():
            with settings.PARENT_CHUNKS_PATH.open("rb") as f:
                self._parent_chunks = pickle.load(f)
            # Build parent lookup: chunk_id → parent dict
            self._parent_lookup = {
                p["chunk_id"]: p for p in self._parent_chunks if p.get("chunk_id")
            }
            logger.info("Parent chunks loaded: %d", len(self._parent_lookup))

        if settings.BM25_PATH.exists():
            with settings.BM25_PATH.open("rb") as f:
                bm25_data = pickle.load(f)
            self._bm25_corpus = bm25_data.get("corpus", [])
            from rank_bm25 import BM25Okapi
            self._bm25_index = BM25Okapi(self._bm25_corpus)
            logger.info("BM25 index loaded: %d docs", len(self._bm25_corpus))

        self._loaded = True
        logger.info("All retrieval artefacts ready (%d chunks)", len(self._chunks))

    def _get_embedder(self):
        """Lazy-load embedding model (avoids startup cost until first query)."""
        if self._embedder is None:
            from backend.utils.embeddings import Embeddings
            self._embedder = Embeddings()
        return self._embedder

    # ── Hybrid search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        alpha: float | None = None,
        use_parent: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval: fuse dense + BM25 scores, resolve parents.

        Parameters
        ----------
        query : str
        k : int
            Number of results to return.
        alpha : float
            Weight for dense score (1-alpha = BM25 weight).
        use_parent : bool
            Whether to resolve child → parent chunks.

        Returns
        -------
        list of result dicts with: chunk, score, parent_content (if resolved)
        """
        self.load()

        if not self.is_loaded:
            logger.error("Cannot search — artefacts not loaded")
            return []

        k = k or settings.RETRIEVAL_TOP_K
        alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        use_parent = use_parent if use_parent is not None else settings.USE_PARENT_RESOLUTION

        embedder = self._get_embedder()

        # Dense retrieval via FAISS
        q_emb = np.asarray(embedder.embed([query]), dtype="float32")
        q_norms = np.linalg.norm(q_emb, keepdims=True)
        q_norms[q_norms == 0.0] = 1.0
        q_emb /= q_norms

        n_dense = min(settings.RETRIEVAL_SEARCH_K, self._index.ntotal)
        dense_scores, dense_idxs = self._index.search(q_emb, n_dense)
        dense_scores = dense_scores[0]
        dense_idxs = dense_idxs[0]

        # BM25 retrieval
        fused: dict[int, float] = {}
        if self._bm25_index is not None:
            q_tokens = self._tokenize(query)
            bm25_scores_all = self._bm25_index.get_scores(q_tokens)
            bm25_max = max(bm25_scores_all.max(), 1e-6)
            bm25_norm = bm25_scores_all / bm25_max

            # Fuse dense + BM25 scores
            for score, idx in zip(dense_scores, dense_idxs):
                if idx >= 0:
                    fused[int(idx)] = alpha * float(score) + (1 - alpha) * float(bm25_norm[int(idx)])

            # Add BM25-only top hits not already in fused
            bm25_top = np.argsort(bm25_scores_all)[::-1][:settings.RETRIEVAL_SEARCH_K]
            for idx in bm25_top:
                idx_int = int(idx)
                if idx_int not in fused:
                    fused[idx_int] = (1 - alpha) * float(bm25_norm[idx_int])
        else:
            # Dense-only fallback
            for score, idx in zip(dense_scores, dense_idxs):
                if idx >= 0:
                    fused[int(idx)] = float(score)

        # Rank by fused score
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            if idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            entry: dict[str, Any] = {
                "chunk": chunk,
                "score": round(score, 4),
                "index": idx,
            }

            # Parent resolution: resolve child → parent for richer LLM context
            if use_parent and self._parent_lookup:
                pid = chunk.get("parent_id") or chunk.get("metadata", {}).get("parent_id")
                parent = self._parent_lookup.get(pid) if pid else None
                if parent:
                    entry["parent_content"] = parent["content"]

            results.append(entry)

        return results

    # ── Tokenizer ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "and", "but", "or", "not", "so",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "we", "us", "you", "your", "i", "my", "me", "he", "she",
        }
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in _STOP and len(t) > 1]

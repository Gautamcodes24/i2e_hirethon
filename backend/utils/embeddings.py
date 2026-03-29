"""Embedding creation logic — supports local (HuggingFace) and OpenAI providers.

Provider is selected via ``backend.config.settings.EMBEDDING_PROVIDER``:
  - ``"local"`` / ``"huggingface"`` → SentenceTransformer (BGE, MiniLM, etc.)
  - ``"openai"`` → OpenAI Embeddings API (text-embedding-3-small, etc.)

Usage:
    from backend.utils.embeddings import Embeddings
    embedder = Embeddings()            # reads provider from config
    vectors  = embedder.embed(texts)
"""

import os
from typing import Sequence

# reduce transformer / HF hub noise during import and setup
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)

# BGE instruction prefix (recommended by the model authors for retrieval)
_BGE_INSTRUCTION = "Represent this sentence: "

# Batch size for OpenAI API calls (API limit: 2048 inputs per request)
_OPENAI_BATCH_SIZE = 512


class Embeddings:
    """Unified embedding interface for local and OpenAI models."""

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
    ):
        self._provider = (provider or settings.EMBEDDING_PROVIDER).lower()

        if self._provider == "openai":
            self._model_name = model_name or settings.OPENAI_EMBEDDING_MODEL
            self._dimension = settings.OPENAI_EMBEDDING_DIM
            self._client = None  # lazy
            logger.info("Embedding provider: OpenAI (%s, %dd)", self._model_name, self._dimension)
        else:
            # local / huggingface
            self._model_name = model_name or settings.EMBEDDING_MODEL
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._model_name)
            self._is_bge = "bge" in self._model_name.lower()
            self._dimension = self._local_model.get_sentence_embedding_dimension()
            logger.info("Embedding provider: local (%s, %dd)", self._model_name, self._dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── OpenAI helpers ────────────────────────────────────────────────────

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            if not settings.has_openai_key:
                raise EnvironmentError("OPENAI_API_KEY is required for OpenAI embeddings")
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _embed_openai(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """Call OpenAI Embeddings API in batches."""
        client = self._get_openai_client()
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[start : start + _OPENAI_BATCH_SIZE]
            response = client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            # Sort by index to preserve order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

            if show_progress:
                done = min(start + _OPENAI_BATCH_SIZE, len(texts))
                logger.info("OpenAI embeddings: %d/%d", done, len(texts))

        return np.asarray(all_embeddings, dtype="float32")

    # ── Local helpers ─────────────────────────────────────────────────────

    def _prepare_texts(self, texts: Sequence[str]) -> list[str]:
        if self._is_bge:
            return [f"{_BGE_INSTRUCTION}{t}" for t in texts]
        return list(texts)

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts and return (N, D) float32 array."""
        if self._provider == "openai":
            return self._embed_openai(list(texts), show_progress=show_progress)

        prepared = self._prepare_texts(texts)
        return self._local_model.encode(
            prepared,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )


def get_embeddings(texts: Sequence[str], model_name: str | None = None) -> np.ndarray:
    """Convenience function — creates an Embeddings instance and calls embed."""
    return Embeddings(model_name=model_name).embed(texts)

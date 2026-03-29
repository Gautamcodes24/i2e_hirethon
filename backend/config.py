"""Centralized configuration — single source of truth for all settings.

All config is loaded from environment variables (with .env support).
No hardcoded secrets, API keys, or paths anywhere else in the codebase.

Usage:
    from backend.config import settings
    print(settings.GROQ_API_KEY)
    print(settings.DATA_DIR)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""

    # ── Paths ─────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    UPLOAD_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "uploads")
    INDEX_PATH: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "faiss.index")
    CHUNKS_PATH: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "chunks.pkl")
    PARENT_CHUNKS_PATH: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "parent_chunks.pkl")
    BM25_PATH: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "bm25_corpus.pkl")
    FRONTEND_BUILD_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "frontend" / "build")

    # ── API Keys (from env) ───────────────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── Provider Selection ("groq" or "openai") ───────────────────────────
    EMBEDDING_PROVIDER: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    LLM_PROVIDER: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    VISION_PROVIDER: str = field(default_factory=lambda: os.getenv("VISION_PROVIDER", "openai"))

    # ── Model Settings ────────────────────────────────────────────────────
    # Local / HuggingFace embedding (used when EMBEDDING_PROVIDER="local")
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    EMBEDDING_DIM: int = 1024
    # OpenAI embedding (used when EMBEDDING_PROVIDER="openai")
    OPENAI_EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    OPENAI_EMBEDDING_DIM: int = field(default_factory=lambda: int(os.getenv("OPENAI_EMBEDDING_DIM", "1536")))
    # Groq models
    GROQ_VISION_MODEL: str = field(default_factory=lambda: os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))
    GROQ_LLM_MODEL: str = field(default_factory=lambda: os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"))
    # OpenAI models
    OPENAI_LLM_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    OPENAI_VISION_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))
    # Shared
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0

    # ── Ingestion Pipeline ────────────────────────────────────────────────
    CHUNK_MAX_TOKENS: int = 256          # child chunk size (for retrieval)
    CHUNK_OVERLAP_TOKENS: int = 30
    PARENT_MAX_TOKENS: int = 1024        # parent chunk size (for LLM context)
    PARENT_OVERLAP_TOKENS: int = 100
    ENABLE_OCR: bool = True
    ENABLE_VISION: bool = True
    ENABLE_PARENT_CHILD: bool = True
    ENABLE_QUALITY_FILTER: bool = True
    ENABLE_DEDUP: bool = True
    ENABLE_TEXT_CLEANING: bool = True
    ENABLE_CAPTIONS: bool = True
    ENABLE_DOC_STRUCTURE: bool = True
    ENABLE_VECTOR_DETECTION: bool = True
    QUALITY_MIN_SCORE: float = 0.30
    VISION_DELAY: float = 1.0            # delay between Groq Vision API calls
    OCR_DPI: int = 300
    OCR_LANG: str = "eng"

    # ── QnA / Retrieval ───────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_SEARCH_K: int = 50         # larger candidate pool before rerank
    HYBRID_ALPHA: float = 0.7            # weight: dense=0.7, BM25=0.3
    USE_PARENT_RESOLUTION: bool = True   # resolve child→parent for LLM context

    # ── Server ────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ])

    # ── Tesseract ─────────────────────────────────────────────────────────
    TESSERACT_PATH: str = field(default_factory=lambda: os.path.expandvars(
        os.getenv("TESSERACT_PATH", r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
    ))

    def __post_init__(self) -> None:
        """Ensure required directories exist."""
        object.__setattr__(self, "DATA_DIR", Path(self.DATA_DIR))
        object.__setattr__(self, "UPLOAD_DIR", Path(self.UPLOAD_DIR))
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def has_groq_key(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def active_embedding_model(self) -> str:
        """Return the model name for the configured embedding provider."""
        if self.EMBEDDING_PROVIDER == "openai":
            return self.OPENAI_EMBEDDING_MODEL
        return self.EMBEDDING_MODEL

    @property
    def active_embedding_dim(self) -> int:
        if self.EMBEDDING_PROVIDER == "openai":
            return self.OPENAI_EMBEDDING_DIM
        return self.EMBEDDING_DIM

    @property
    def active_llm_model(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_LLM_MODEL
        return self.GROQ_LLM_MODEL

    @property
    def active_vision_model(self) -> str:
        if self.VISION_PROVIDER == "openai":
            return self.OPENAI_VISION_MODEL
        return self.GROQ_VISION_MODEL

    @property
    def vision_enabled(self) -> bool:
        return self.ENABLE_VISION and self.has_groq_key


# ── Singleton instance ────────────────────────────────────────────────────────
settings = Settings()

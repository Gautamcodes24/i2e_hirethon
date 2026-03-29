"""Unified LLM client — abstracts Groq and OpenAI behind a single interface.

Reads PROVIDER from backend.config.settings to decide which SDK to use.
Both providers expose the same OpenAI-compatible chat completions API,
so the wrapper is thin.

Usage:
    from backend.utils.llm_client import get_llm_client
    client = get_llm_client()                     # uses LLM_PROVIDER
    client = get_llm_client(provider="openai")    # explicit override
"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


def get_llm_client(*, provider: str | None = None) -> Any:
    """Return a chat-completions client for the requested provider.

    Both Groq and OpenAI SDKs expose ``client.chat.completions.create()``,
    so callers can use the returned object identically.
    """
    provider = (provider or settings.LLM_PROVIDER).lower()

    if provider == "openai":
        if not settings.has_openai_key:
            raise EnvironmentError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from openai import OpenAI
        logger.info("Using OpenAI LLM client (model=%s)", settings.OPENAI_LLM_MODEL)
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    if provider == "groq":
        if not settings.has_groq_key:
            raise EnvironmentError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from groq import Groq
        logger.info("Using Groq LLM client (model=%s)", settings.GROQ_LLM_MODEL)
        return Groq(api_key=settings.GROQ_API_KEY)

    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'groq' or 'openai'.")


def get_vision_client(*, provider: str | None = None) -> Any:
    """Return a vision-capable chat client for the requested provider."""
    provider = (provider or settings.VISION_PROVIDER).lower()

    if provider == "openai":
        if not settings.has_openai_key:
            raise EnvironmentError("OPENAI_API_KEY is required when VISION_PROVIDER=openai")
        from openai import OpenAI
        logger.info("Using OpenAI Vision client (model=%s)", settings.OPENAI_VISION_MODEL)
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    if provider == "groq":
        if not settings.has_groq_key:
            raise EnvironmentError("GROQ_API_KEY is required when VISION_PROVIDER=groq")
        from groq import Groq
        logger.info("Using Groq Vision client (model=%s)", settings.GROQ_VISION_MODEL)
        return Groq(api_key=settings.GROQ_API_KEY)

    raise ValueError(f"Unknown vision provider: {provider!r}. Use 'groq' or 'openai'.")

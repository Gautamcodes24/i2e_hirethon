"""Centralized logging with progress bar support.

Provides:
  - Structured console logging with timestamps and module names
  - tqdm-based progress bars for long operations
  - A helper to create module-specific loggers

Usage:
    from backend.logger import get_logger, progress_bar

    logger = get_logger(__name__)
    logger.info("Starting ingestion")

    for item in progress_bar(items, desc="Processing pages"):
        process(item)
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, TypeVar

T = TypeVar("T")

# ── Log format ────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Track whether root has been configured to avoid duplicate handlers
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler with clean formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in [
        "httpx", "urllib3", "transformers", "huggingface_hub",
        "sentence_transformers", "filelock", "httpcore",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Create a named logger. Auto-configures root logging on first call."""
    setup_logging()
    return logging.getLogger(name)


# ── Progress bar ──────────────────────────────────────────────────────────────

def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
    disable: bool = False,
) -> Iterable[T]:
    """Wrap an iterable with a tqdm progress bar.

    Falls back to the plain iterable if tqdm is not installed.
    """
    try:
        from tqdm import tqdm
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            disable=disable,
            ncols=100,
            bar_format="{l_bar}{bar}│ {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    except ImportError:
        return iterable


def step_logger(logger: logging.Logger, total_steps: int, on_step=None) -> "StepTracker":
    """Create a step tracker that logs progress like '[3/8] Building embeddings...'"""
    return StepTracker(logger, total_steps, on_step=on_step)


class StepTracker:
    """Logs numbered pipeline steps: [1/8] Phase description..."""

    def __init__(self, logger: logging.Logger, total: int, on_step=None):
        self._logger = logger
        self._total = total
        self._current = 0
        self._on_step = on_step

    def step(self, message: str) -> None:
        self._current += 1
        self._logger.info("[%d/%d] %s", self._current, self._total, message)
        if self._on_step:
            self._on_step(self._current, self._total, message)

    def done(self, message: str = "Pipeline complete") -> None:
        self._logger.info("✓ %s (%d steps)", message, self._current)
        if self._on_step:
            self._on_step(self._total, self._total, message)

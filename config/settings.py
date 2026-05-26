"""
Centralized configuration for the Intelligent Document Q&A Engine.

All settings are loaded from environment variables (via .env file)
with sensible defaults. Use `get_settings()` to access the singleton.
"""

from functools import lru_cache
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import List


try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        """Application-wide configuration backed by .env file (pydantic)."""

        # ── Grok (xAI) LLM ─────────────────────────────────────────────────
        XAI_API_KEY: str = ""
        XAI_MODEL: str = "grok-4.3"
        XAI_API_BASE: str = "https://api.x.ai/v1"

        # ── Embeddings (local, free via sentence-transformers) ──────────────
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        EMBEDDING_DIMENSIONS: int = 384  # MiniLM-L6-v2 outputs 384 dims
        EMBEDDING_BATCH_SIZE: int = 64  # chunks per batch

        # ── Chunking – Recursive ────────────────────────────────────────────
        CHUNK_SIZE: int = 512
        CHUNK_OVERLAP: int = 50

        # ── Chunking – Sliding Window ───────────────────────────────────────
        SLIDING_WINDOW_SIZE: int = 512
        SLIDING_WINDOW_STEP: int = 256  # overlap = window_size - step

        # ── Chunking – Semantic ─────────────────────────────────────────────
        SEMANTIC_BREAKPOINT_TYPE: str = "percentile"  # percentile | standard_deviation | interquartile

        # ── Paths ───────────────────────────────────────────────────────────
        EMBEDDINGS_OUTPUT_DIR: str = "./output/embeddings"

        # ── Supported file extensions ───────────────────────────────────────
        SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt"]

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

except Exception:  # pragma: no cover - fallback when pydantic_settings isn't available
    @dataclass
    class Settings:
        """Lightweight settings fallback using environment variables."""

        XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
        XAI_MODEL: str = os.getenv("XAI_MODEL", "grok-4.3")
        XAI_API_BASE: str = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")

        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
        EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

        CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
        CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

        SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "512"))
        SLIDING_WINDOW_STEP: int = int(os.getenv("SLIDING_WINDOW_STEP", "256"))

        SEMANTIC_BREAKPOINT_TYPE: str = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")

        EMBEDDINGS_OUTPUT_DIR: str = os.getenv("EMBEDDINGS_OUTPUT_DIR", "./output/embeddings")

        SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".txt"])


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()

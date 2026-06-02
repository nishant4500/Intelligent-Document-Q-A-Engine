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

        # ── LLM Configuration ──────────────────────────────────────────────
        LLM_PROVIDER: str = "xai"  # "xai" | "openai" | "groq" | "mock"
        XAI_API_KEY: str = ""
        XAI_MODEL: str = "grok-2"  # standard xai model
        XAI_API_BASE: str = "https://api.x.ai/v1"
        OPENAI_API_KEY: str = ""
        OPENAI_MODEL: str = "gpt-4o-mini"
        OPENAI_API_BASE: str = "https://api.openai.com/v1"
        GROQ_API_KEY: str = ""
        GROQ_MODEL: str = "llama-3.1-8b-instant"  # llama3-8b-8192 was decommissioned
        GROQ_API_BASE: str = "https://api.groq.com/openai/v1"

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

        # ── Vector Store & Search Service ──────────────────────────────────
        FAISS_INDEX_DIR: str = "./output/faiss_index"
        BM25_INDEX_PATH: str = "./output/bm25_index.pkl"
        HYBRID_ALPHA: float = 0.5  # Weight for dense vector similarity (1-alpha is for BM25)
        RERANK_TOP_K: int = 15     # Number of candidates to rerank
        RERANK_FINAL_N: int = 4    # Number of final results to feed to LLM
        RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ENABLE_RERANKING: bool = True

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

        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "xai")
        XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
        XAI_MODEL: str = os.getenv("XAI_MODEL", "grok-2")
        XAI_API_BASE: str = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        GROQ_API_BASE: str = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
        EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

        CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
        CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

        SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "512"))
        SLIDING_WINDOW_STEP: int = int(os.getenv("SLIDING_WINDOW_STEP", "256"))

        SEMANTIC_BREAKPOINT_TYPE: str = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")

        FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "./output/faiss_index")
        BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./output/bm25_index.pkl")
        HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
        RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "15"))
        RERANK_FINAL_N: int = int(os.getenv("RERANK_FINAL_N", "4"))
        RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "True").lower() == "true"

        EMBEDDINGS_OUTPUT_DIR: str = os.getenv("EMBEDDINGS_OUTPUT_DIR", "./output/embeddings")

        SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".txt"])


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()

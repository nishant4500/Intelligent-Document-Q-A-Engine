"""
Base chunker interface and shared data models for text chunking.

Every chunking strategy must subclass `BaseChunker` and implement
the `chunk()` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ingestion.base_reader import Document


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""

    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        return (
            f"Chunk(index={self.chunk_index}, "
            f"chars={self.char_count}, words={self.word_count})"
        )


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: The ingested Document to chunk.

        Returns:
            List of Chunk objects with text and metadata.
        """
        ...

    def _enrich_metadata(
        self,
        chunks: list[Chunk],
        document: Document,
    ) -> list[Chunk]:
        """Add source document metadata and strategy info to each chunk."""
        for chunk in chunks:
            chunk.metadata["source"] = document.metadata.get("source", "unknown")
            chunk.metadata["filename"] = document.metadata.get("filename", "unknown")
            chunk.metadata["file_type"] = document.metadata.get("file_type", "unknown")
            chunk.metadata["chunking_strategy"] = self.strategy_name
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

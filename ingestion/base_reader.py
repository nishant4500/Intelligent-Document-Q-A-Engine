"""
Base reader interface and shared data models for document ingestion.

Every file-type reader (PDF, DOCX, TXT) must subclass `BaseReader`
and implement the `read()` method.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """Represents a parsed document with its extracted text and metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        return f"Document(source={source!r}, chars={self.char_count}, words={self.word_count})"


# ── Exceptions ──────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Raised when a document cannot be ingested."""
    pass


# ── Abstract Base Reader ────────────────────────────────────────────────────

class BaseReader(ABC):
    """Abstract base class that all document readers must implement."""

    # Subclasses should declare which file extensions they handle
    supported_extensions: list[str] = []

    def validate_file(self, file_path: str) -> Path:
        """
        Validate that the file exists and has a supported extension.

        Args:
            file_path: Path to the file to validate.

        Returns:
            Resolved Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
            IngestionError: If the file extension is not supported.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise IngestionError(f"Path is not a file: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise IngestionError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {self.supported_extensions}"
            )

        return path

    def _base_metadata(self, path: Path) -> dict[str, Any]:
        """Generate common metadata fields for any file."""
        stat = path.stat()
        return {
            "source": str(path),
            "filename": path.name,
            "file_extension": path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    @abstractmethod
    def read(self, file_path: str) -> Document:
        """
        Parse a document and return its text and metadata.

        Args:
            file_path: Path to the document file.

        Returns:
            Document object containing the extracted text and metadata.

        Raises:
            IngestionError: If the file cannot be parsed.
        """
        ...

"""
Plain text reader — US-103.

Reads .txt files with automatic encoding detection.
Handles BOM stripping, whitespace normalization, and edge cases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chardet

from ingestion.base_reader import BaseReader, Document, IngestionError

logger = logging.getLogger(__name__)


class TxtReader(BaseReader):
    """Reader for plain text (.txt) files."""

    supported_extensions = [".txt"]

    def read(self, file_path: str) -> Document:
        """
        Parse a TXT file with automatic encoding detection.

        Preprocessing steps:
        1. Detect encoding via chardet
        2. Decode content
        3. Strip BOM (Byte Order Mark) if present
        4. Normalize line endings to \\n
        5. Strip leading/trailing whitespace

        Args:
            file_path: Path to the .txt file.

        Returns:
            Document with extracted text and metadata.

        Raises:
            IngestionError: If the file cannot be read or decoded.
        """
        path = self.validate_file(file_path)
        logger.info(f"Reading TXT file: {path.name}")

        # ── Read raw bytes ──────────────────────────────────────────────
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise IngestionError(f"Failed to read file '{path.name}': {e}")

        if not raw_bytes:
            logger.warning(f"TXT file '{path.name}' is empty.")
            metadata = self._base_metadata(path)
            metadata.update({
                "file_type": "txt",
                "encoding": "utf-8",
                "line_count": 0,
                "char_count": 0,
            })
            return Document(text="", metadata=metadata)

        # ── Detect encoding ─────────────────────────────────────────────
        detection = chardet.detect(raw_bytes)
        encoding = detection.get("encoding", "utf-8") or "utf-8"
        confidence = detection.get("confidence", 0.0)

        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

        # ── Decode content ──────────────────────────────────────────────
        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"Encoding '{encoding}' failed, falling back to utf-8: {e}")
            text = raw_bytes.decode("utf-8", errors="replace")
            encoding = "utf-8 (fallback)"

        # ── Preprocessing ───────────────────────────────────────────────
        # Strip BOM
        text = text.lstrip("\ufeff")

        # Normalize line endings (CRLF → LF, CR → LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip leading/trailing whitespace
        text = text.strip()

        # ── Build metadata ──────────────────────────────────────────────
        lines = text.split("\n") if text else []
        metadata = self._base_metadata(path)
        metadata.update({
            "file_type": "txt",
            "encoding": encoding,
            "encoding_confidence": round(confidence, 2),
            "line_count": len(lines),
            "char_count": len(text),
        })

        logger.info(f"Extracted {len(lines)} lines, {len(text)} chars from '{path.name}'")

        return Document(text=text, metadata=metadata)

"""
PDF document reader — Bonus.

Extracts text from PDF files using PyPDF2.
Handles multi-page extraction and metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from ingestion.base_reader import BaseReader, Document, IngestionError

logger = logging.getLogger(__name__)


class PdfReader_(BaseReader):
    """Reader for PDF (.pdf) files."""

    supported_extensions = [".pdf"]

    def read(self, file_path: str) -> Document:
        """
        Parse a PDF file and extract text from all pages.

        Args:
            file_path: Path to the .pdf file.

        Returns:
            Document with extracted text and metadata.

        Raises:
            IngestionError: If the file cannot be parsed.
        """
        path = self.validate_file(file_path)
        logger.info(f"Reading PDF file: {path.name}")

        try:
            reader = PdfReader(str(path))
        except PdfReadError:
            raise IngestionError(f"Invalid or corrupted PDF file: {path.name}")
        except Exception as e:
            raise IngestionError(f"Failed to open PDF file '{path.name}': {e}")

        # ── Check for encrypted PDFs ────────────────────────────────────
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise IngestionError(
                    f"PDF file '{path.name}' is encrypted and cannot be read."
                )

        # ── Extract text from each page ─────────────────────────────────
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            logger.warning(
                f"PDF file '{path.name}' is empty or contains no extractable text. "
                "It may be a scanned/image-only PDF."
            )

        # ── Build metadata ──────────────────────────────────────────────
        metadata = self._base_metadata(path)
        metadata.update(self._extract_pdf_metadata(reader))
        metadata["page_count"] = len(reader.pages)
        metadata["pages_with_text"] = len(pages_text)
        metadata["file_type"] = "pdf"

        logger.info(
            f"Extracted text from {len(pages_text)}/{len(reader.pages)} pages "
            f"in '{path.name}'"
        )

        return Document(text=full_text, metadata=metadata)

    @staticmethod
    def _extract_pdf_metadata(reader: PdfReader) -> dict[str, Any]:
        """Extract PDF document info as metadata."""
        result = {}
        info = reader.metadata

        if info is None:
            return result

        field_mapping = {
            "/Title": "title",
            "/Author": "author",
            "/Subject": "subject",
            "/Creator": "creator",
            "/Producer": "producer",
        }

        for pdf_key, meta_key in field_mapping.items():
            value = info.get(pdf_key)
            if value:
                result[meta_key] = str(value)

        return result

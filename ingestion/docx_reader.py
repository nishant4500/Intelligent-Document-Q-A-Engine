"""
DOCX document reader — US-102.

Extracts text from Word documents using python-docx.
Handles paragraphs, tables, headers, and document properties.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from ingestion.base_reader import BaseReader, Document, IngestionError

logger = logging.getLogger(__name__)


class DocxReader(BaseReader):
    """Reader for Microsoft Word (.docx) files."""

    supported_extensions = [".docx"]

    def read(self, file_path: str) -> Document:
        """
        Parse a DOCX file and extract all text content.

        Extracts text from:
        - All paragraphs (body text, headings, lists)
        - All table cells
        - Document core properties (author, title, etc.)

        Args:
            file_path: Path to the .docx file.

        Returns:
            Document with extracted text and metadata.

        Raises:
            IngestionError: If the file cannot be parsed.
        """
        path = self.validate_file(file_path)
        logger.info(f"Reading DOCX file: {path.name}")

        try:
            doc = DocxDocument(str(path))
        except PackageNotFoundError:
            raise IngestionError(f"Invalid or corrupted DOCX file: {path.name}")
        except Exception as e:
            raise IngestionError(f"Failed to open DOCX file '{path.name}': {e}")

        # ── Extract paragraph text ──────────────────────────────────────
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # ── Extract table text ──────────────────────────────────────────
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_cells.append(cell_text)
                if row_cells:
                    table_texts.append(" | ".join(row_cells))

        # ── Combine all text ────────────────────────────────────────────
        all_sections = []
        if paragraphs:
            all_sections.append("\n\n".join(paragraphs))
        if table_texts:
            all_sections.append("\n".join(table_texts))

        full_text = "\n\n".join(all_sections)

        if not full_text.strip():
            logger.warning(f"DOCX file '{path.name}' is empty or contains no extractable text.")

        # ── Build metadata ──────────────────────────────────────────────
        metadata = self._base_metadata(path)
        metadata.update(self._extract_properties(doc))
        metadata["paragraph_count"] = len(paragraphs)
        metadata["table_count"] = len(doc.tables)
        metadata["file_type"] = "docx"

        logger.info(
            f"Extracted {len(paragraphs)} paragraphs, "
            f"{len(doc.tables)} tables from '{path.name}'"
        )

        return Document(text=full_text, metadata=metadata)

    @staticmethod
    def _extract_properties(doc: DocxDocument) -> dict[str, Any]:
        """Extract document core properties as metadata."""
        props = doc.core_properties
        result = {}

        if props.author:
            result["author"] = props.author
        if props.title:
            result["title"] = props.title
        if props.subject:
            result["subject"] = props.subject
        if props.created:
            result["doc_created"] = props.created.isoformat()
        if props.modified:
            result["doc_modified"] = props.modified.isoformat()

        return result

"""
Document ingestion package.

Provides readers for PDF, DOCX, and TXT files, plus a factory function
to automatically select the right reader based on file extension.
"""

from ingestion.base_reader import BaseReader, Document, IngestionError

# Readers are imported lazily in `get_reader` to avoid requiring optional
# dependencies at module import time (e.g., `python-docx`, `PyPDF2`).

# ── Reader Registry ─────────────────────────────────────────────────────────

def get_reader(file_path: str) -> BaseReader:
    """Return an appropriate reader instance for the given file path.

    Readers are imported on demand to keep optional heavy dependencies
    out of the import path when they're not needed.
    """
    from pathlib import Path

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from ingestion.pdf_reader import PdfReader_

        return PdfReader_()
    if ext == ".docx":
        from ingestion.docx_reader import DocxReader

        return DocxReader()
    if ext == ".txt":
        from ingestion.txt_reader import TxtReader

        return TxtReader()

    supported = [".pdf", ".docx", ".txt"]
    raise IngestionError(
        f"No reader available for extension '{ext}'. Supported formats: {supported}"
    )


def ingest(file_path: str) -> Document:
    """
    Convenience function: auto-detect file type and extract text.

    Args:
        file_path: Path to the document file.

    Returns:
        Document with extracted text and metadata.
    """
    reader = get_reader(file_path)
    return reader.read(file_path)


__all__ = [
    "BaseReader",
    "Document",
    "IngestionError",
    "DocxReader",
    "PdfReader_",
    "TxtReader",
    "get_reader",
    "ingest",
]

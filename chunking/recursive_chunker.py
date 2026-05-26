"""
Recursive text chunker — US-104.

Wraps LangChain's RecursiveCharacterTextSplitter to split documents
using a hierarchy of separators (paragraphs → sentences → words).
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for environments without langchain
    RecursiveCharacterTextSplitter = None

from chunking.base_chunker import BaseChunker, Chunk
from config.settings import get_settings
from ingestion.base_reader import Document

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based text splitter.

    Splits text by progressively smaller separators, trying to keep
    semantically related text together (paragraphs → lines → sentences → words).
    """

    strategy_name = "recursive"

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to config.
            chunk_overlap: Overlap between consecutive chunks. Defaults to config.
            separators: List of separators to split on, in priority order.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        if RecursiveCharacterTextSplitter is not None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            # Lightweight fallback splitter when LangChain isn't available.
            class _SimpleSplitter:
                def __init__(self, chunk_size, chunk_overlap, separators):
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap
                    self.separators = separators

                def create_documents(self, texts, metadatas=None):
                    docs = []
                    for text, meta in zip(texts, metadatas or [{}]):
                        i = 0
                        n = len(text)
                        while i < n:
                            end = min(i + self.chunk_size, n)
                            chunk_text = text[i:end]
                            docs.append(type("Doc", (), {"page_content": chunk_text, "metadata": meta}))
                            i = end - self.chunk_overlap if end - self.chunk_overlap > i else end
                    return docs

            self._splitter = _SimpleSplitter(self.chunk_size, self.chunk_overlap, self.separators)

        logger.info(
            f"RecursiveChunker initialized: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks using recursive character splitting.

        Args:
            document: The ingested Document to chunk.

        Returns:
            List of Chunk objects with text and metadata.
        """
        if not document.text.strip():
            logger.warning("Document text is empty, returning no chunks.")
            return []

        # Use LangChain to split the text
        lc_docs = self._splitter.create_documents(
            texts=[document.text],
            metadatas=[document.metadata],
        )

        chunks = []
        for idx, lc_doc in enumerate(lc_docs):
            chunk = Chunk(
                text=lc_doc.page_content,
                chunk_index=idx,
                metadata={
                    "chunk_size_config": self.chunk_size,
                    "chunk_overlap_config": self.chunk_overlap,
                },
            )
            chunks.append(chunk)

        # Enrich with source metadata
        chunks = self._enrich_metadata(chunks, document)

        logger.info(
            f"Recursive chunking produced {len(chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars each)"
        )

        return chunks

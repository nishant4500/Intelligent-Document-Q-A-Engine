"""
Sliding-window text chunker — US-106.

Custom implementation of overlapping sliding-window chunking with
configurable window size and step size.
"""

from __future__ import annotations

import logging
from typing import Optional

from chunking.base_chunker import BaseChunker, Chunk
from config.settings import get_settings
from ingestion.base_reader import Document

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """
    Overlapping sliding-window chunker.

    Slides a fixed-size window across the text with a configurable step.
    The overlap between consecutive chunks is:  window_size - step_size.
    """

    strategy_name = "sliding_window"

    def __init__(
        self,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
    ):
        """
        Initialize the sliding-window chunker.

        Args:
            window_size: Number of characters per window/chunk.
                Defaults to config SLIDING_WINDOW_SIZE.
            step_size: Number of characters to advance each step.
                Defaults to config SLIDING_WINDOW_STEP.
                Must be <= window_size.

        Raises:
            ValueError: If step_size > window_size.
        """
        settings = get_settings()
        self.window_size = window_size or settings.SLIDING_WINDOW_SIZE
        self.step_size = step_size or settings.SLIDING_WINDOW_STEP

        if self.step_size > self.window_size:
            raise ValueError(
                f"step_size ({self.step_size}) must be <= "
                f"window_size ({self.window_size})"
            )

        self.overlap = self.window_size - self.step_size

        logger.info(
            f"SlidingWindowChunker initialized: "
            f"window={self.window_size}, step={self.step_size}, "
            f"overlap={self.overlap}"
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into overlapping chunks using a sliding window.

        Args:
            document: The ingested Document to chunk.

        Returns:
            List of Chunk objects with text and metadata.
        """
        text = document.text
        if not text.strip():
            logger.warning("Document text is empty, returning no chunks.")
            return []

        chunks = []
        text_length = len(text)
        start = 0
        idx = 0

        while start < text_length:
            end = min(start + self.window_size, text_length)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=idx,
                    metadata={
                        "window_size": self.window_size,
                        "step_size": self.step_size,
                        "overlap_chars": self.overlap,
                        "start_char": start,
                        "end_char": end,
                    },
                )
                chunks.append(chunk)
                idx += 1

            start += self.step_size

            # Avoid duplicate tail chunk
            if end == text_length:
                break

        # Enrich with source metadata
        chunks = self._enrich_metadata(chunks, document)

        logger.info(
            f"Sliding-window chunking produced {len(chunks)} chunks "
            f"(window={self.window_size}, overlap={self.overlap})"
        )

        return chunks

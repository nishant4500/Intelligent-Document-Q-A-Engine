"""
Tests for chunking strategies — US-104, US-105, US-106.
"""

import os
import tempfile

import pytest

from chunking import get_chunker, STRATEGY_MAP
from chunking.base_chunker import Chunk
from chunking.recursive_chunker import RecursiveChunker
from chunking.sliding_window_chunker import SlidingWindowChunker
from ingestion.base_reader import Document


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_document(text: str = None) -> Document:
    """Create a test Document with sample text."""
    if text is None:
        text = (
            "Artificial intelligence is transforming the world. "
            "Machine learning algorithms can now process vast amounts of data. "
            "Deep learning has revolutionized computer vision and NLP.\n\n"
            "Natural language processing enables machines to understand text. "
            "Transformers architecture changed the landscape of AI research. "
            "Large language models can generate human-like text.\n\n"
            "Retrieval-Augmented Generation combines search with generation. "
            "RAG systems improve accuracy by grounding answers in documents. "
            "Vector databases enable efficient semantic search at scale.\n\n"
            "The future of AI includes multimodal systems and agentic workflows. "
            "Responsible AI development requires careful attention to safety. "
            "AI alignment research aims to ensure beneficial outcomes for humanity."
        )
    return Document(
        text=text,
        metadata={
            "source": "test_document.txt",
            "filename": "test_document.txt",
            "file_type": "txt",
        },
    )


# ── US-104: Recursive Chunker Tests ────────────────────────────────────────

class TestRecursiveChunker:
    """Tests for the recursive character text splitter."""

    def test_basic_chunking(self):
        """Verify chunks are produced from a document."""
        doc = _make_document()
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self):
        """Verify no chunk exceeds the configured size (with small tolerance)."""
        doc = _make_document()
        chunk_size = 200
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            # Allow small tolerance for separator edge cases
            assert chunk.char_count <= chunk_size + 10, (
                f"Chunk {chunk.chunk_index} has {chunk.char_count} chars, "
                f"exceeds limit of {chunk_size}"
            )

    def test_no_text_lost(self):
        """Verify that all original text content is preserved across chunks."""
        text = "Word " * 100  # 500 characters
        doc = _make_document(text.strip())
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        combined = " ".join(c.text for c in chunks)
        # All original words should be in the combined output
        for word in text.strip().split():
            assert word in combined

    def test_chunk_indices_sequential(self):
        """Verify chunk indices are sequential starting from 0."""
        doc = _make_document()
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_enriched(self):
        """Verify source metadata is added to chunks."""
        doc = _make_document()
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test_document.txt"
            assert chunk.metadata["chunking_strategy"] == "recursive"
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_empty_document(self):
        """Verify empty documents produce no chunks."""
        doc = _make_document("")
        chunker = RecursiveChunker()
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_custom_separators(self):
        """Verify custom separators are respected."""
        doc = _make_document()
        chunker = RecursiveChunker(
            chunk_size=300,
            chunk_overlap=0,
            separators=["\n\n"],
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0


# ── US-106: Sliding Window Chunker Tests ────────────────────────────────────

class TestSlidingWindowChunker:
    """Tests for the sliding-window chunker."""

    def test_basic_sliding_window(self):
        """Verify chunks are produced with overlap."""
        doc = _make_document()
        chunker = SlidingWindowChunker(window_size=200, step_size=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_overlap_between_chunks(self):
        """Verify consecutive chunks have overlapping content."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20  # 520 chars
        doc = _make_document(text)
        chunker = SlidingWindowChunker(window_size=100, step_size=50)
        chunks = chunker.chunk(doc)

        # Check that consecutive chunks overlap
        for i in range(len(chunks) - 1):
            start_i = chunks[i].metadata["start_char"]
            end_i = chunks[i].metadata["end_char"]
            start_next = chunks[i + 1].metadata["start_char"]

            # The next chunk should start before the current one ends
            assert start_next < end_i, (
                f"No overlap between chunk {i} and {i+1}"
            )

    def test_configurable_overlap(self):
        """Verify overlap matches (window_size - step_size)."""
        window_size = 200
        step_size = 150
        expected_overlap = window_size - step_size

        chunker = SlidingWindowChunker(
            window_size=window_size,
            step_size=step_size,
        )
        assert chunker.overlap == expected_overlap

    def test_window_size_respected(self):
        """Verify no chunk exceeds window size."""
        doc = _make_document()
        window_size = 200
        chunker = SlidingWindowChunker(window_size=window_size, step_size=100)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.char_count <= window_size

    def test_metadata_has_positions(self):
        """Verify chunks have start/end character positions."""
        doc = _make_document()
        chunker = SlidingWindowChunker(window_size=200, step_size=100)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "start_char" in chunk.metadata
            assert "end_char" in chunk.metadata
            assert "overlap_chars" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "sliding_window"

    def test_step_larger_than_window_raises(self):
        """Verify ValueError when step_size > window_size."""
        with pytest.raises(ValueError, match="step_size"):
            SlidingWindowChunker(window_size=100, step_size=200)

    def test_empty_document(self):
        """Verify empty documents produce no chunks."""
        doc = _make_document("")
        chunker = SlidingWindowChunker(window_size=200, step_size=100)
        chunks = chunker.chunk(doc)

        assert chunks == []


# ── Factory Tests ───────────────────────────────────────────────────────────

class TestChunkerFactory:
    """Test the strategy factory function."""

    def test_get_recursive(self):
        """Verify recursive chunker creation."""
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_get_sliding_window(self):
        """Verify sliding_window chunker creation."""
        chunker = get_chunker("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)

    def test_unknown_strategy(self):
        """Verify ValueError for unknown strategies."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_all_strategies_registered(self):
        """Verify all three strategies are in the registry."""
        assert "recursive" in STRATEGY_MAP
        assert "semantic" in STRATEGY_MAP
        assert "sliding_window" in STRATEGY_MAP

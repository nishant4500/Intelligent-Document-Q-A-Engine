"""
Tests for embedding generation — US-107.

Uses sentence-transformers (local, free). No API key needed.
"""

import os
import tempfile

import pytest

from chunking.base_chunker import Chunk


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_chunks(n: int = 3) -> list[Chunk]:
    """Create a list of test chunks."""
    return [
        Chunk(
            text=f"This is test chunk number {i}. It contains sample text for embedding.",
            chunk_index=i,
            metadata={"source": "test.txt", "file_type": "txt"},
        )
        for i in range(n)
    ]


# ── Unit Tests ──────────────────────────────────────────────────────────────

class TestEmbeddedChunkDataclass:
    """Tests for the EmbeddedChunk dataclass."""

    def test_properties(self):
        """Test EmbeddedChunk properties."""
        from embeddings.embedding_generator import EmbeddedChunk

        ec = EmbeddedChunk(
            text="Test text",
            chunk_index=0,
            embedding=[0.1] * 384,
            metadata={"source": "test.txt"},
        )

        assert ec.dimensions == 384
        assert "dims=384" in repr(ec)

    def test_empty_embedding(self):
        """Test EmbeddedChunk with empty embedding."""
        from embeddings.embedding_generator import EmbeddedChunk

        ec = EmbeddedChunk(
            text="Test",
            chunk_index=0,
            embedding=[],
            metadata={},
        )
        assert ec.dimensions == 0


# ── Integration Tests (local model — no API key needed) ─────────────────────

class TestEmbeddingGeneratorIntegration:
    """Integration tests using sentence-transformers locally."""

    def test_generate_embeddings(self):
        """Test actual embedding generation with local model."""
        from embeddings.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator()
        chunks = _make_chunks(3)
        embedded = generator.generate(chunks, show_progress=False)

        assert len(embedded) == 3
        for ec in embedded:
            assert ec.dimensions == generator.dimensions
            assert len(ec.embedding) == generator.dimensions
            assert ec.metadata["embedding_model"] == generator.model_name

    def test_empty_chunks_returns_empty(self):
        """Verify empty input produces empty output."""
        from embeddings.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator()
        result = generator.generate([])
        assert result == []

    def test_save_and_load(self):
        """Test saving and loading embeddings to/from disk."""
        from embeddings.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator()
        chunks = _make_chunks(2)
        embedded = generator.generate(chunks, show_progress=False)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = generator.save(embedded, output_path=tmpdir)
            assert os.path.exists(save_path)

            # Load
            loaded = EmbeddingGenerator.load(save_path)
            assert len(loaded) == 2
            assert loaded[0].dimensions == embedded[0].dimensions

    def test_embedding_matrix(self):
        """Test numpy matrix conversion."""
        from embeddings.embedding_generator import EmbeddingGenerator
        import numpy as np

        generator = EmbeddingGenerator()
        chunks = _make_chunks(3)
        embedded = generator.generate(chunks, show_progress=False)

        matrix = generator.get_embedding_matrix(embedded)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, generator.dimensions)
        assert matrix.dtype == np.float32

    def test_different_texts_different_embeddings(self):
        """Verify different texts produce different embeddings."""
        from embeddings.embedding_generator import EmbeddingGenerator
        import numpy as np

        generator = EmbeddingGenerator()
        chunks = [
            Chunk(text="The cat sat on the mat.", chunk_index=0, metadata={}),
            Chunk(text="Quantum physics explores subatomic particles.", chunk_index=1, metadata={}),
        ]
        embedded = generator.generate(chunks, show_progress=False)

        # Embeddings should NOT be identical
        assert not np.allclose(embedded[0].embedding, embedded[1].embedding)

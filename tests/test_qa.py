"""
Tests for RAG Prompting & LLM completion streaming — US-109, US-110, US-111.

Verifies CoT prompting, streaming citations injection, and unified LLM client fallbacks.
"""

import pytest

from embeddings.embedding_generator import EmbeddedChunk
from qa.llm_client import LLMClient
from qa.rag_engine import RAGEngine
from indexing.faiss_index import FAISSIndexManager
from indexing.hybrid_search import BM25Searcher, HybridSearcher


# ── Helpers ─────────────────────────────────────────────────────────────────

class MockEmbeddingGenerator:
    """Mock generator to avoid local transformer overhead in QA tests."""
    def __init__(self, dimensions=384):
        self.dimensions = dimensions

    def generate(self, chunks, show_progress=False):
        # returns dummy embeddings
        class MockEmbedded:
            def __init__(self, text, idx):
                self.text = text
                self.chunk_index = idx
                self.embedding = [0.1] * 384
                self.metadata = {"filename": "test.txt", "file_type": ".txt"}
        return [MockEmbedded(c.text, c.chunk_index) for c in chunks]


# ── Unit Tests ──────────────────────────────────────────────────────────────

class TestLLMClient:
    """Tests for the LLMClient class."""

    def test_mock_fallback_on_key_absence(self):
        """Verify client gracefully falls back to mock generator when API keys are absent."""
        client = LLMClient(provider="openai", api_key="")
        assert client.provider == "mock"
        assert client.model == "local-mock"

    def test_mock_stream_relevance(self):
        """Test mock response text content and citation inclusion."""
        client = LLMClient(provider="mock")
        messages = [
            {"role": "user", "content": "Query: rate limit. Retrieved Contexts: [Source: gateway.pdf, Chunk: 0]"}
        ]
        
        # Consume stream
        tokens = list(client.stream_chat(messages))
        full_text = "".join(tokens)
        
        # Verify it has standard offline response structure and extracted source file
        assert "Local RAG Offline Response" in full_text
        assert "gateway.pdf" in full_text
        assert "[Source: gateway.pdf, Chunk: 0]" in full_text


class TestRAGEngine:
    """Tests for the RAGEngine orchestrator."""

    def test_format_contexts(self):
        """Test search results prompt context formatting."""
        # Mock search result tuple list
        chunk = EmbeddedChunk(
            text="Context text",
            chunk_index=3,
            embedding=[],
            metadata={"filename": "specs.txt"}
        )
        results = [(chunk, 0.85)]

        # We construct a RAGEngine with dummy values to test internal formatting
        generator = MockEmbeddingGenerator()
        engine = RAGEngine(
            hybrid_searcher=None,  # type: ignore
            embedding_generator=generator,  # type: ignore
            llm_client=LLMClient(provider="mock")
        )

        formatted = engine._format_contexts(results)
        assert "specs.txt" in formatted
        assert "Chunk Index: 3" in formatted
        assert "Context text" in formatted

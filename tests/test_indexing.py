"""
Tests for indexing and hybrid search — US-108, US-115, US-116.

Verifies FAISS vector index, BM25 keyword matching, and HybridSearcher score fusion.
"""

import os
import tempfile
import pytest
import numpy as np

from embeddings.embedding_generator import EmbeddedChunk
from indexing.faiss_index import FAISSIndexManager
from indexing.hybrid_search import BM25Searcher, HybridSearcher


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_embedded_chunks() -> list[EmbeddedChunk]:
    """Create a list of sample embedded chunks."""
    # 384 dimensions matching all-MiniLM-L6-v2
    # Create simple orthogonal vectors to make test similarity checks deterministic
    emb0 = [0.0] * 384
    emb0[0] = 1.0  # active dimension 0
    emb1 = [0.0] * 384
    emb1[1] = 1.0  # active dimension 1
    emb2 = [0.0] * 384
    emb2[2] = 1.0  # active dimension 2
    emb3 = [0.0] * 384
    emb3[3] = 1.0  # active dimension 3

    return [
        EmbeddedChunk(
            text="The API Gateway implements RS256 JWT key rate limiting specifications.",
            chunk_index=0,
            embedding=emb0,
            metadata={"source": "api_spec.txt", "filename": "api_spec.txt", "file_type": ".txt"},
        ),
        EmbeddedChunk(
            text="PostgreSQL transaction cluster shards writes across 16 shards using account_uuid.",
            chunk_index=1,
            embedding=emb1,
            metadata={"source": "db_spec.docx", "filename": "db_spec.docx", "file_type": ".docx"},
        ),
        EmbeddedChunk(
            text="Confluent Schema Registry manages Apache Avro schemas ensuring strict backward compatibility.",
            chunk_index=2,
            embedding=emb2,
            metadata={"source": "registry.txt", "filename": "registry.txt", "file_type": ".txt"},
        ),
        EmbeddedChunk(
            text="Disaster recovery point objective RPO target is set to exactly 5 minutes.",
            chunk_index=3,
            embedding=emb3,
            metadata={"source": "dr.txt", "filename": "dr.txt", "file_type": ".txt"},
        )
    ]


# ── Unit Tests ──────────────────────────────────────────────────────────────

class TestFAISSIndexManager:
    """Tests for the FAISSIndexManager class."""

    def test_add_and_search(self):
        """Test adding vectors and performing semantic search."""
        chunks = _make_embedded_chunks()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_mgr = FAISSIndexManager(index_dir=tmpdir, dimensions=384)
            faiss_mgr.add_chunks(chunks, persist=True)
            
            # Query close to chunk 0
            query_emb = [0.0] * 384
            query_emb[0] = 1.0  # matches chunk 0
            
            results = faiss_mgr.search(query_emb, k=1)
            assert len(results) == 1
            matched_chunk, score = results[0]
            assert matched_chunk.chunk_index == 0
            assert "api_spec.txt" in matched_chunk.metadata["filename"]
            assert score > 0.9  # exact match similarity is ~1.0

    def test_metadata_filtering(self):
        """Test search with metadata post-filtering constraints."""
        chunks = _make_embedded_chunks()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_mgr = FAISSIndexManager(index_dir=tmpdir, dimensions=384)
            faiss_mgr.add_chunks(chunks, persist=False)
            
            # Query pointing to chunk 0 but filter for DOCX
            query_emb = [0.0] * 384
            query_emb[0] = 1.0
            
            # Search with filter matching db_spec.docx
            results = faiss_mgr.search(query_emb, k=1, metadata_filter={"file_type": ".docx"})
            assert len(results) == 1
            matched_chunk, _ = results[0]
            # Should match chunk 1 because chunk 0 was filtered out!
            assert matched_chunk.chunk_index == 1
            assert matched_chunk.metadata["file_type"] == ".docx"

    def test_save_and_load(self):
        """Test index persistence on disk."""
        chunks = _make_embedded_chunks()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_mgr = FAISSIndexManager(index_dir=tmpdir, dimensions=384)
            faiss_mgr.add_chunks(chunks, persist=True)
            
            # Instantiate second manager and load
            faiss_mgr2 = FAISSIndexManager(index_dir=tmpdir, dimensions=384)
            assert faiss_mgr2._index.ntotal == 4
            assert len(faiss_mgr2._metadata_map) == 4


class TestBM25Searcher:
    """Tests for the BM25Searcher class."""

    def test_build_and_search(self):
        """Test BM25 keyword matching."""
        chunks = _make_embedded_chunks()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bm25_path = os.path.join(tmpdir, "bm25.pkl")
            searcher = BM25Searcher(index_path=bm25_path)
            searcher.build_index(chunks, persist=True)
            
            # Search for keyword specific to chunk 1
            results = searcher.search("shards writes PostgreSQL", k=1)
            assert len(results) == 1
            matched_chunk, score = results[0]
            assert matched_chunk.chunk_index == 1
            assert score > 0.0

    def test_save_and_load(self):
        """Test BM25 disk persistence."""
        chunks = _make_embedded_chunks()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bm25_path = os.path.join(tmpdir, "bm25.pkl")
            searcher = BM25Searcher(index_path=bm25_path)
            searcher.build_index(chunks, persist=True)
            
            searcher2 = BM25Searcher(index_path=bm25_path)
            assert len(searcher2._chunks) == 4


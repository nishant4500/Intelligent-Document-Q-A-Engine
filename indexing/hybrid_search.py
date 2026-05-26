"""
Hybrid Search & Re-ranking — US-115, US-116.

Combines FAISS dense retrieval and BM25 sparse retrieval with score normalization,
and applies a Cross-Encoder model to re-rank the top candidates.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from chunking.base_chunker import Chunk
from config.settings import get_settings
from embeddings.embedding_generator import EmbeddedChunk, EmbeddingGenerator
from indexing.faiss_index import FAISSIndexManager

logger = logging.getLogger(__name__)


class BM25Searcher:
    """
    Sparse retrieval engine using BM25.
    
    Provides exact keyword and lexical match capabilities.
    """

    def __init__(self, index_path: Optional[str] = None):
        """Initialize the BM25 searcher."""
        settings = get_settings()
        self.index_path = Path(index_path or settings.BM25_INDEX_PATH)
        self._bm25 = None
        self._chunks: list[EmbeddedChunk] = []

        if self.index_path.exists():
            self.load()

    def _tokenize(self, text: str) -> list[str]:
        """Simple English tokenizer for BM25: lowercase, strip punctuation, split."""
        text = text.lower()
        # Remove punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, embedded_chunks: list[EmbeddedChunk], persist: bool = True) -> None:
        """
        Build BM25 index from a list of chunks.

        Args:
            embedded_chunks: The chunks to index.
            persist: Whether to save the index to disk.
        """
        from rank_bm25 import BM25Okapi

        if not embedded_chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self._chunks = embedded_chunks
        
        # Tokenize documents
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in embedded_chunks]
        
        logger.info(f"Building BM25 index over {len(embedded_chunks)} documents...")
        self._bm25 = BM25Okapi(tokenized_corpus)

        if persist:
            self.save()

    def save(self) -> None:
        """Persist BM25 index and chunk mapping to disk."""
        if self._bm25 is None:
            logger.warning("No BM25 index to save.")
            return

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "bm25_object": self._bm25,
                "chunks": self._chunks,
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(payload, f)
            logger.info(f"BM25 index saved to {self.index_path}")
        except Exception as e:
            logger.error(
                f"Graceful Disk Write Bypass: Failed to save BM25 index to disk: {e}. "
                f"The active BM25 index remains 100% operational in-memory (RAM)."
            )

    def load(self) -> None:
        """Load BM25 index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found at {self.index_path}")

        logger.info(f"Loading BM25 index from {self.index_path}...")
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)

        self._bm25 = payload["bm25_object"]
        self._chunks = payload["chunks"]
        logger.info(f"BM25 index loaded successfully with {len(self._chunks)} documents.")

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Perform BM25 search and return top-k results.

        Args:
            query: The natural-language query string.
            k: Number of results to return.
            metadata_filter: Metadata filtering options.

        Returns:
            List of (EmbeddedChunk, score) tuples.
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("Search called on an uninitialized BM25 index.")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Zip chunks and scores, and filter by metadata
        results = []
        for chunk, score in zip(self._chunks, scores):
            if score <= 0.0:
                continue

            # Apply metadata post-filtering
            if metadata_filter:
                match = True
                for key, val in metadata_filter.items():
                    if chunk.metadata.get(key) != val:
                        match = False
                        break
                if not match:
                    continue

            results.append((chunk, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def clear(self) -> None:
        """Reset the BM25 searcher and remove index file."""
        self._bm25 = None
        self._chunks = []
        if self.index_path.exists():
            self.index_path.unlink()
        logger.info("BM25 index cleared.")


class CrossEncoderReRanker:
    """
    Re-ranking engine using a HuggingFace Cross-Encoder.
    
    Predicts a score for the query and a document context together,
    yielding significantly higher relevance accuracy compared to dual-encoders alone.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the Cross-Encoder re-ranker."""
        settings = get_settings()
        self.model_name = model_name or settings.RERANK_MODEL
        self._model = None

    def _lazy_load(self):
        """Lazily load the cross-encoder model to save memory during startup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-Encoder loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[EmbeddedChunk],
        top_n: int = 4,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Re-rank a list of candidate chunks for a query.

        Args:
            query: The user query.
            candidates: List of candidate EmbeddedChunk objects.
            top_n: Number of final results to return.

        Returns:
            List of (EmbeddedChunk, cross_encoder_score) tuples, sorted descending.
        """
        if not candidates:
            return []

        self._lazy_load()

        # Build sentence pairs
        pairs = [[query, chunk.text] for chunk in candidates]

        # Get scores
        scores = self._model.predict(pairs)

        # Match back to candidates
        scored_candidates = []
        for chunk, score in zip(candidates, scores):
            # Sigmoid normalization if needed, or raw logits. Standard ms-marco-MiniLM outputs raw logits.
            # Convert float32 numpy float to python float
            scored_candidates.append((chunk, float(score)))

        # Sort descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_n]


class HybridSearcher:
    """
    Orchestrates Hybrid Search combining FAISS vector search and BM25 keyword search,
    followed by an optional Cross-Encoder re-ranking phase.
    """

    def __init__(
        self,
        faiss_manager: FAISSIndexManager,
        bm25_searcher: BM25Searcher,
        re_ranker: Optional[CrossEncoderReRanker] = None,
    ):
        """
        Initialize the Hybrid Searcher.

        Args:
            faiss_manager: An initialized FAISSIndexManager instance.
            bm25_searcher: An initialized BM25Searcher instance.
            re_ranker: A CrossEncoderReRanker instance.
        """
        self.faiss = faiss_manager
        self.bm25 = bm25_searcher
        self.re_ranker = re_ranker or CrossEncoderReRanker()

    def search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
        alpha: Optional[float] = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Execute hybrid search with optional re-ranking.

        Args:
            query: Natural language query string.
            query_embedding: Dense embedding vector for the query.
            k: Final number of results to return.
            metadata_filter: Optional metadata filtering constraints.
            alpha: Weight for dense vector search (1-alpha is for BM25). Defaults to settings.

        Returns:
            List of (EmbeddedChunk, score) tuples, sorted by score descending.
        """
        settings = get_settings()
        a = alpha if alpha is not None else settings.HYBRID_ALPHA
        
        # Determine internal limits based on re-ranking config
        top_k_candidates = settings.RERANK_TOP_K if settings.ENABLE_RERANKING else k
        
        # 1. Fetch candidates from both search methods
        # Search a bit wider in individual indexes to ensure diverse candidates
        search_pool_k = max(top_k_candidates * 2, 20)
        
        dense_results = self.faiss.search(
            query_embedding,
            k=search_pool_k,
            metadata_filter=metadata_filter,
        )
        sparse_results = self.bm25.search(
            query,
            k=search_pool_k,
            metadata_filter=metadata_filter,
        )

        if not dense_results and not sparse_results:
            return []

        # 2. Fuse scores using Normalized Weighted Fusion
        # Create candidate maps
        dense_map = {ec.text: (ec, score) for ec, score in dense_results}
        sparse_map = {ec.text: (ec, score) for ec, score in sparse_results}

        # Normalize dense scores (already cosine similarity, typically [0, 1])
        # If there are dense scores, normalize them in their active range
        dense_scores_raw = [score for _, score in dense_results]
        min_d = min(dense_scores_raw) if dense_scores_raw else 0.0
        max_d = max(dense_scores_raw) if dense_scores_raw else 1.0
        range_d = max_d - min_d
        
        # Normalize sparse scores (BM25 raw scores, [0, inf])
        sparse_scores_raw = [score for _, score in sparse_results]
        min_s = min(sparse_scores_raw) if sparse_scores_raw else 0.0
        max_s = max(sparse_scores_raw) if sparse_scores_raw else 1.0
        range_s = max_s - min_s

        # Union of all candidate keys
        all_text_keys = set(dense_map.keys()).union(sparse_map.keys())
        
        fused_candidates = []
        for text in all_text_keys:
            ec = None
            
            # Normalize Dense Score
            if text in dense_map:
                ec, score = dense_map[text]
                norm_dense = (score - min_d) / range_d if range_d > 0 else score
            else:
                norm_dense = 0.0

            # Normalize Sparse Score
            if text in sparse_map:
                ec_s, score = sparse_map[text]
                ec = ec or ec_s  # use sparse's EmbeddedChunk if dense wasn't there
                norm_sparse = (score - min_s) / range_s if range_s > 0 else score
            else:
                norm_sparse = 0.0

            # Compute combined score
            fused_score = a * norm_dense + (1 - a) * norm_sparse
            fused_candidates.append((ec, fused_score))

        # Sort and take top candidate pool
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates_to_keep = [ec for ec, _ in fused_candidates[:top_k_candidates] if ec is not None]

        # 3. Apply Re-ranking (if enabled)
        if settings.ENABLE_RERANKING and candidates_to_keep:
            logger.info(f"Re-ranking {len(candidates_to_keep)} candidates with {settings.RERANK_MODEL}...")
            reranked_results = self.re_ranker.rerank(
                query,
                candidates_to_keep,
                top_n=k,
            )
            return reranked_results
        
        # Otherwise return fused candidate scores directly
        return fused_candidates[:k]

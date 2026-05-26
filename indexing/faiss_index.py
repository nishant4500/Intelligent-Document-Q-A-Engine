"""
FAISS Index Manager — US-108.

Manages building, saving, loading, and searching a local FAISS index
with metadata filtering support.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import get_settings
from embeddings.embedding_generator import EmbeddedChunk

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manages a local FAISS index for high-performance dense vector search.

    Uses faiss.IndexFlatIP (Inner Product) under the hood. Since we normalize
    vectors before indexing, Inner Product search yields exact Cosine Similarity.
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the FAISS index manager.

        Args:
            index_dir: Directory where the index is saved/loaded. Defaults to config.
            dimensions: Dimensionality of embeddings. Defaults to config dimensions.
        """
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.FAISS_INDEX_DIR)
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "metadata.pkl"

        import faiss
        self._faiss = faiss

        self._index: Any = None
        self._metadata_map: list[dict[str, Any]] = []
        self._chunk_texts: list[str] = []
        self._chunk_indices: list[int] = []

        # Load existing index if it exists
        if self.index_path.exists() and self.meta_path.exists():
            self.load()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Initialize an empty FAISS IndexFlatIP."""
        logger.info(f"Creating a new empty FAISS IndexFlatIP with {self.dimensions} dimensions.")
        self._index = self._faiss.IndexFlatIP(self.dimensions)
        self._metadata_map = []
        self._chunk_texts = []
        self._chunk_indices = []

    def save(self) -> None:
        """Persist the FAISS index and metadata mappings to disk."""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self._faiss.write_index(self._index, str(self.index_path))

            # Save metadata mapping
            metadata_payload = {
                "metadata_map": self._metadata_map,
                "chunk_texts": self._chunk_texts,
                "chunk_indices": self._chunk_indices,
                "dimensions": self.dimensions,
            }
            with open(self.meta_path, "wb") as f:
                pickle.dump(metadata_payload, f)

            logger.info(
                f"Successfully saved FAISS index to {self.index_path} "
                f"({self._index.ntotal} total vectors)"
            )
        except Exception as e:
            logger.error(
                f"Graceful Disk Write Bypass: Failed to save FAISS index to disk: {e}. "
                f"The active index remains 100% operational in-memory (RAM)."
            )

    def load(self) -> None:
        """Load the FAISS index and metadata from disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS index or metadata files do not exist.")

        logger.info(f"Loading FAISS index from {self.index_dir}...")
        self._index = self._faiss.read_index(str(self.index_path))

        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)

        self._metadata_map = meta.get("metadata_map", [])
        self._chunk_texts = meta.get("chunk_texts", [])
        self._chunk_indices = meta.get("chunk_indices", [])
        self.dimensions = meta.get("dimensions", self.dimensions)

        logger.info(
            f"FAISS index loaded. Total indexed chunks: {self._index.ntotal} "
            f"({self.dimensions} dimensions)"
        )

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors to compute exact cosine similarity using Inner Product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def add_chunks(self, embedded_chunks: list[EmbeddedChunk], persist: bool = True) -> None:
        """
        Add a list of embedded chunks to the FAISS index.

        Args:
            embedded_chunks: Chunks to index.
            persist: Whether to immediately save changes to disk.
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks provided to add.")
            return

        # Extract embeddings matrix
        embeddings = np.array(
            [ec.embedding for ec in embedded_chunks],
            dtype=np.float32,
        )

        # Validate dimensions
        if embeddings.shape[1] != self.dimensions:
            raise ValueError(
                f"Embedding dimensions mismatch. Expected {self.dimensions}, "
                f"got {embeddings.shape[1]}"
            )

        # L2 normalize embeddings for cosine similarity
        norm_embeddings = self._normalize_vectors(embeddings)

        # Add to index
        self._index.add(norm_embeddings)

        # Save metadata and content mappings
        for ec in embedded_chunks:
            self._metadata_map.append(ec.metadata)
            self._chunk_texts.append(ec.text)
            self._chunk_indices.append(ec.chunk_index)

        logger.info(
            f"Added {len(embedded_chunks)} vectors to FAISS. "
            f"New total size: {self._index.ntotal}"
        )

        if persist:
            self.save()

    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        k: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Search the FAISS index for the most similar chunks.

        Supports metadata post-filtering.

        Args:
            query_embedding: Embedding vector of the query.
            k: Number of final results to return.
            metadata_filter: Dictionary of metadata key-value constraints.
                             Example: {"source": "report.pdf", "file_type": "pdf"}

        Returns:
            List of (EmbeddedChunk, similarity_score) tuples, sorted descending by score.
        """
        if self._index.ntotal == 0:
            logger.warning("Search called on an empty FAISS index.")
            return []

        # Convert query to 2D numpy array
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # L2 normalize for cosine similarity
        query_vec = self._normalize_vectors(query_vec)

        # If a metadata filter is provided, we fetch a larger pool (e.g., k * 10 or min(100, total))
        # to ensure we have enough matches after post-filtering.
        search_k = min(self._index.ntotal, k * 10 if metadata_filter else k)
        search_k = max(search_k, k)

        # Execute FAISS search
        scores, indices = self._index.search(query_vec, search_k)
        
        # Flatten results (we only queried 1 vector)
        flat_scores = scores[0]
        flat_indices = indices[0]

        results = []
        for score, idx in zip(flat_scores, flat_indices):
            if idx == -1:
                continue

            metadata = self._metadata_map[idx]
            
            # Apply metadata post-filtering if specified
            if metadata_filter:
                match = True
                for key, val in metadata_filter.items():
                    if metadata.get(key) != val:
                        match = False
                        break
                if not match:
                    continue

            # Reconstruct EmbeddedChunk
            chunk = EmbeddedChunk(
                text=self._chunk_texts[idx],
                chunk_index=self._chunk_indices[idx],
                embedding=[],  # We omit embedding in returned object to save memory/speed
                metadata=metadata,
            )
            # FAISS scores are inner products (cosine similarity because vectors are normalized)
            # Clip between -1.0 and 1.0
            similarity = float(np.clip(score, -1.0, 1.0))
            results.append((chunk, similarity))

            # Stop once we have reached the requested k
            if len(results) == k:
                break

        return results

    def clear(self) -> None:
        """Reset the FAISS index and clear disk files."""
        self._create_new_index()
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        logger.info("FAISS index and metadata cleared.")

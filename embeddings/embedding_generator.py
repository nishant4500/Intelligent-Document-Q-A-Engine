"""
Embedding generator — US-107.

Generates vector embeddings for text chunks using sentence-transformers
(free, local — no API key required).

Default model: all-MiniLM-L6-v2 (384 dimensions, ~90MB)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from chunking.base_chunker import Chunk
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk enriched with its vector embedding."""

    text: str
    chunk_index: int
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.embedding)

    def __repr__(self) -> str:
        return (
            f"EmbeddedChunk(index={self.chunk_index}, "
            f"dims={self.dimensions}, chars={len(self.text)})"
        )


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers (local, free).

    Default model: all-MiniLM-L6-v2 (384 dimensions).
    No API key required — everything runs locally on your machine.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model name. Defaults to config.
            batch_size: Number of chunks per batch. Defaults to config.
            device: Device to run on ('cpu', 'cuda'). Auto-detected if None.
        """
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        logger.info(f"Loading embedding model: {self.model_name}...")
        self._model = SentenceTransformer(self.model_name, device=device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

        logger.info(
            f"EmbeddingGenerator ready: model={self.model_name}, "
            f"dims={self._dimensions}, device={self._model.device}"
        )

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality of the loaded model."""
        return self._dimensions

    def generate(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of Chunk objects to embed.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            List of EmbeddedChunk objects with embedding vectors.
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation.")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        texts = [chunk.text for chunk in chunks]

        # sentence-transformers handles batching internally
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                embedding=embedding.tolist(),
                metadata={
                    **chunk.metadata,
                    "embedding_model": self.model_name,
                    "embedding_dimensions": len(embedding),
                },
            )
            embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Successfully generated {len(embedded_chunks)} embeddings "
            f"({self._dimensions} dimensions each)"
        )

        return embedded_chunks

    def save(
        self,
        embedded_chunks: list[EmbeddedChunk],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save embedded chunks to disk as a pickle file.

        Args:
            embedded_chunks: List of EmbeddedChunk objects to save.
            output_path: Directory to save the file. Defaults to config dir.

        Returns:
            Path to the saved file.
        """
        settings = get_settings()
        output_dir = Path(output_path or settings.EMBEDDINGS_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / "embeddings.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(embedded_chunks, f)

        logger.info(f"Saved {len(embedded_chunks)} embedded chunks to {file_path}")
        return str(file_path)

    @staticmethod
    def load(file_path: str) -> list[EmbeddedChunk]:
        """
        Load embedded chunks from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            List of EmbeddedChunk objects.
        """
        with open(file_path, "rb") as f:
            embedded_chunks = pickle.load(f)

        logger.info(f"Loaded {len(embedded_chunks)} embedded chunks from {file_path}")
        return embedded_chunks

    def get_embedding_matrix(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> np.ndarray:
        """
        Convert embedded chunks to a numpy matrix for FAISS indexing.

        Args:
            embedded_chunks: List of EmbeddedChunk objects.

        Returns:
            Numpy array of shape (n_chunks, embedding_dimensions).
        """
        matrix = np.array(
            [ec.embedding for ec in embedded_chunks],
            dtype=np.float32,
        )
        logger.info(f"Created embedding matrix with shape {matrix.shape}")
        return matrix

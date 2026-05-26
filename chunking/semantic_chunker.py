"""
Semantic text chunker — US-105.

Uses embedding similarity to split text at semantic boundaries.
Powered by sentence-transformers (free, local — no API key needed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from chunking.base_chunker import BaseChunker, Chunk
from config.settings import get_settings
from ingestion.base_reader import Document

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Embedding-based semantic chunker.

    Calculates sentence-level embeddings using sentence-transformers
    and identifies semantic breakpoints where the meaning shifts
    significantly. This produces more coherent chunks compared to
    fixed-size splitting.

    Runs entirely locally — no API key required.
    """

    strategy_name = "semantic"

    def __init__(
        self,
        breakpoint_threshold_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the semantic chunker.

        Args:
            breakpoint_threshold_type: Method to detect breakpoints.
                One of: 'percentile', 'standard_deviation', 'interquartile'.
                Defaults to config setting.
            model_name: HuggingFace model for embeddings. Defaults to config.
        """
        settings = get_settings()
        self.breakpoint_type = (
            breakpoint_threshold_type or settings.SEMANTIC_BREAKPOINT_TYPE
        )
        self.model_name = model_name or settings.EMBEDDING_MODEL

        logger.info(
            f"SemanticChunker initialized: "
            f"breakpoint_type={self.breakpoint_type}, model={self.model_name}"
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple heuristics."""
        import re
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_breakpoints(
        self, embeddings: np.ndarray
    ) -> list[int]:
        """
        Find indices where semantic similarity drops significantly.

        Args:
            embeddings: Matrix of sentence embeddings (n_sentences, dims).

        Returns:
            List of sentence indices where splits should occur.
        """
        if len(embeddings) < 2:
            return []

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            similarities.append(cos_sim)

        similarities = np.array(similarities)

        # Compute distances (1 - similarity)
        distances = 1 - similarities

        # Determine threshold based on breakpoint type
        if self.breakpoint_type == "percentile":
            threshold = np.percentile(distances, 75)
        elif self.breakpoint_type == "standard_deviation":
            threshold = np.mean(distances) + np.std(distances)
        elif self.breakpoint_type == "interquartile":
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        else:
            threshold = np.percentile(distances, 75)

        # Find breakpoints where distance exceeds threshold
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        return breakpoints

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into semantically coherent chunks.

        Falls back to simple paragraph splitting if embedding fails.

        Args:
            document: The ingested Document to chunk.

        Returns:
            List of Chunk objects with text and metadata.
        """
        if not document.text.strip():
            logger.warning("Document text is empty, returning no chunks.")
            return []

        try:
            from sentence_transformers import SentenceTransformer

            # Split text into sentences
            sentences = self._split_into_sentences(document.text)

            if len(sentences) <= 1:
                # Single sentence or unsplittable — return as one chunk
                chunks = [
                    Chunk(
                        text=document.text.strip(),
                        chunk_index=0,
                        metadata={"breakpoint_type": self.breakpoint_type},
                    )
                ]
                return self._enrich_metadata(chunks, document)

            # Generate sentence-level embeddings
            logger.info(f"Encoding {len(sentences)} sentences with {self.model_name}...")
            model = SentenceTransformer(self.model_name)
            embeddings = model.encode(sentences, show_progress_bar=False)

            # Find semantic breakpoints
            breakpoints = self._compute_breakpoints(embeddings)

            # Group sentences into chunks at breakpoints
            chunks = []
            start_idx = 0
            chunk_index = 0

            split_points = breakpoints + [len(sentences)]
            for end_idx in split_points:
                chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
                if chunk_text:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        metadata={
                            "breakpoint_type": self.breakpoint_type,
                            "sentence_range": f"{start_idx}-{end_idx}",
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                start_idx = end_idx

        except Exception as e:
            logger.error(
                f"Semantic chunking failed: {e}. "
                f"Falling back to paragraph-based splitting."
            )
            chunks = self._fallback_chunk(document)

        # Enrich with source metadata
        chunks = self._enrich_metadata(chunks, document)

        logger.info(
            f"Semantic chunking produced {len(chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars each)"
        )

        return chunks

    @staticmethod
    def _fallback_chunk(document: Document) -> list[Chunk]:
        """Simple paragraph-based fallback when semantic chunking fails."""
        paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
        return [
            Chunk(
                text=para,
                chunk_index=idx,
                metadata={"fallback": True},
            )
            for idx, para in enumerate(paragraphs)
        ]

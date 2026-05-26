"""
RAG Engine Orchestrator — US-110, US-111.

Combines retrieval (dense/sparse) and LLM orchestration to answer user queries,
incorporating Chain-of-Thought reasoning and strict inline source citation injection.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Optional

from chunking.base_chunker import Chunk
from config.settings import get_settings
from embeddings.embedding_generator import EmbeddedChunk, EmbeddingGenerator
from indexing.hybrid_search import HybridSearcher
from qa.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) orchestrator.
    
    Connects the hybrid search index, embedding model, and LLM to answer Q&A queries
    with streaming completions and precise source citations.
    """

    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        embedding_generator: EmbeddingGenerator,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the RAG Engine.

        Args:
            hybrid_searcher: Configured HybridSearcher instance.
            embedding_generator: Configured EmbeddingGenerator instance.
            llm_client: Unified LLMClient instance. Defaults to config provider.
        """
        self.searcher = hybrid_searcher
        self.embedding_generator = embedding_generator
        self.llm = llm_client or LLMClient()

    def _get_system_prompt(self) -> str:
        """Return the standard system instructions for CoT and citation generation."""
        return (
            "You are an expert AI Document Q&A assistant. You will be provided with a set of retrieved text contexts.\n"
            "Your task is to answer the user's query truthfully, accurately, and strictly using ONLY the provided contexts.\n\n"
            "Guidelines:\n"
            "1. Professional Tone: Keep your language precise, professional, and helpful.\n"
            "2. Chain-of-Thought (CoT) Reasoning: You MUST explain your analytical step-by-step reasoning in a separate section first. "
            "Think out loud: analyze the retrieved contexts, assess how they address the query, identify key facts, and draft your plan before writing the final response.\n"
            "3. Source Citations: You MUST back up every claim, fact, or statement by citing the specific source document and chunk index in brackets, "
            "e.g., [Source: document_name.pdf, Chunk: 3]. Never say 'according to source 1'. Always use the exact filename and chunk index in the brackets.\n"
            "4. Strict Groundedness: If the provided contexts do not contain enough information to answer the query, state clearly that you do not "
            "possess enough information based on the uploaded documents. Under no circumstances should you generate answers outside of the contexts.\n"
            "5. Formatting: Structure your response clearly. Use '### 🧠 Chain-of-Thought Analysis' for your reasoning and '### 💬 Response' for your final answer."
        )

    def _format_contexts(self, results: list[tuple[EmbeddedChunk, float]]) -> str:
        """Format the search results into a clean string for the LLM prompt."""
        formatted_blocks = []
        for idx, (chunk, score) in enumerate(results):
            source = chunk.metadata.get("filename", "unknown")
            chunk_idx = chunk.chunk_index
            block = (
                f"--- Context {idx + 1} ---\n"
                f"Source File: {source}\n"
                f"Chunk Index: {chunk_idx}\n"
                f"Similarity Match Score: {score:.3f}\n"
                f"Text Content:\n{chunk.text.strip()}\n"
            )
            formatted_blocks.append(block)
        return "\n".join(formatted_blocks)

    def answer_query(
        self,
        query_text: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Orchestrate the complete Q&A pipeline for a user query.

        Args:
            query_text: Natural language user query.
            k: Number of contexts to retrieve. Defaults to settings.RERANK_FINAL_N.
            metadata_filter: Optional metadata filtering constraints.
            temperature: Sampling temperature for the LLM.

        Returns:
            Dictionary containing:
                - 'stream': Generator of streaming tokens.
                - 'retrieved_chunks': List of EmbeddedChunk objects retrieved.
        """
        settings = get_settings()
        retrieve_k = k or settings.RERANK_FINAL_N

        logger.info(f"Processing query: '{query_text}'")

        # 1. Embed query
        # Create a temporary Chunk to pass to the embedding generator
        temp_chunk = Chunk(text=query_text, chunk_index=0, metadata={})
        embedded_query_list = self.embedding_generator.generate([temp_chunk], show_progress=False)
        query_embedding = embedded_query_list[0].embedding

        # 2. Retrieve matched contexts using Hybrid Searcher
        results = self.searcher.search(
            query_text,
            query_embedding,
            k=retrieve_k,
            metadata_filter=metadata_filter,
        )

        logger.info(f"Retrieved {len(results)} candidate chunks for query.")

        # Extract only the chunks
        retrieved_chunks = [chunk for chunk, _ in results]

        # 3. Format Prompt
        contexts_str = self._format_contexts(results)
        
        system_prompt = self._get_system_prompt()
        user_prompt = (
            f"User Query:\n{query_text}\n\n"
            f"Retrieved Document Contexts:\n"
            f"============================================================\n"
            f"{contexts_str}\n"
            f"============================================================\n\n"
            f"Provide your step-by-step Chain-of-Thought analysis and final answered response:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 4. Stream response
        stream = self.llm.stream_chat(
            messages=messages,
            temperature=temperature,
        )

        return {
            "stream": stream,
            "retrieved_chunks": retrieved_chunks,
        }

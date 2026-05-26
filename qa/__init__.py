"""
QA package for Project 1.

Provides LLM interfaces and RAG generation orchestration with streaming citation injection.
"""

from qa.llm_client import LLMClient
from qa.rag_engine import RAGEngine

__all__ = [
    "LLMClient",
    "RAGEngine",
]

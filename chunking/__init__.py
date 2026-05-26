"""
Chunking strategies package.

Provides three chunking approaches plus a factory function
to select a strategy by name.
"""

from chunking.base_chunker import BaseChunker, Chunk
from chunking.recursive_chunker import RecursiveChunker
from chunking.sliding_window_chunker import SlidingWindowChunker

try:
    from chunking.semantic_chunker import SemanticChunker
except Exception:  # pragma: no cover - optional dependency
    class SemanticChunker(BaseChunker):
        strategy_name = "semantic"

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "SemanticChunker requires additional dependencies (numpy, sentence-transformers)."
            )

# ── Strategy Registry ───────────────────────────────────────────────────────

STRATEGY_MAP: dict[str, type[BaseChunker]] = {
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "sliding_window": SlidingWindowChunker,
}


def get_chunker(strategy: str, **kwargs) -> BaseChunker:
    """
    Factory function to create a chunker by strategy name.

    Args:
        strategy: One of 'recursive', 'semantic', 'sliding_window'.
        **kwargs: Additional keyword arguments passed to the chunker constructor.

    Returns:
        An instance of the requested chunker.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    chunker_cls = STRATEGY_MAP.get(strategy.lower())
    if chunker_cls is None:
        available = list(STRATEGY_MAP.keys())
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available strategies: {available}"
        )
    return chunker_cls(**kwargs)


__all__ = [
    "BaseChunker",
    "Chunk",
    "RecursiveChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "get_chunker",
    "STRATEGY_MAP",
]

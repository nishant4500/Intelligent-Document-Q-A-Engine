"""
Intelligent Document Q&A Engine — CLI Pipeline Runner.

Usage:
    python main.py --file <path> --chunker recursive|semantic|sliding_window [--embed]
    python main.py --index --file <path> [--chunker recursive|semantic|sliding_window]
    python main.py --query "your question"
    python main.py --eval
    python main.py --serve

Examples:
    python main.py --file docs/report.pdf --chunker recursive --embed
    python main.py --index -f docs/paper.pdf --chunker semantic
    python main.py --query "What is the rate limit for standard accounts?"
    python main.py --serve
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before importing our modules
load_dotenv()

from chunking import get_chunker, STRATEGY_MAP
from ingestion import ingest
from config.settings import get_settings

# ── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Q&A Engine — Pipeline Runner & Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Mode selectors
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--index",
        action="store_true",
        help="Run index mode: ingest, chunk, embed, and store document in FAISS/BM25 database.",
    )
    mode_group.add_argument(
        "--query",
        type=str,
        help="Run query mode: search hybrid index and generate streaming answer for question.",
    )
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation mode: execute benchmark suite comparing chunking strategies.",
    )
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="Run serve mode: launch FastAPI REST API and beautiful SPA dashboard.",
    )

    # Ingestion arguments
    parser.add_argument(
        "--file", "-f",
        required=False,
        help="Path to the document file (PDF, DOCX, or TXT). Required for --index or basic pipeline.",
    )
    parser.add_argument(
        "--chunker", "-c",
        choices=list(STRATEGY_MAP.keys()),
        default="recursive",
        help="Chunking strategy to use (default: recursive).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size from config.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap from config.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate local embeddings for chunks in basic pipeline mode.",
    )
    parser.add_argument(
        "--save-embeddings",
        type=str,
        default=None,
        help="Directory to save embeddings (default: from config).",
    )
    return parser


def print_summary(title: str, items: dict):
    """Pretty-print a summary section."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    for key, value in items.items():
        print(f"  {key:<30} {value}")
    print()


def run_basic_pipeline(args: argparse.Namespace):
    """Execute the basic document parsing & optional embedding generation pipeline."""
    if not args.file:
        logger.error("Error: --file is required for basic pipeline processing.")
        sys.exit(1)

    file_path = Path(args.file).resolve()
    logger.info(f"Starting basic pipeline for: {file_path.name}")

    # ── Step 1: Ingest ──
    print("\n[STEP 1] Document Ingestion")
    document = ingest(str(file_path))

    print_summary("Document Info", {
        "Source": document.metadata.get("filename", "unknown"),
        "Type": document.metadata.get("file_type", "unknown"),
        "Characters": f"{document.char_count:,}",
        "Words": f"{document.word_count:,}",
    })

    # ── Step 2: Chunk ──
    print(f"[STEP 2] Chunking (strategy: {args.chunker})")

    chunker_kwargs = {}
    if args.chunker == "recursive":
        if args.chunk_size:
            chunker_kwargs["chunk_size"] = args.chunk_size
        if args.chunk_overlap:
            chunker_kwargs["chunk_overlap"] = args.chunk_overlap
    elif args.chunker == "sliding_window":
        if args.chunk_size:
            chunker_kwargs["window_size"] = args.chunk_size
        if args.chunk_overlap:
            chunker_kwargs["step_size"] = args.chunk_size - args.chunk_overlap

    chunker = get_chunker(args.chunker, **chunker_kwargs)
    chunks = chunker.chunk(document)

    if chunks:
        avg_size = sum(c.char_count for c in chunks) / len(chunks)
        print_summary("Chunking Results", {
            "Strategy": args.chunker,
            "Total Chunks": len(chunks),
            "Avg Chunk Size (chars)": f"{avg_size:.0f}",
            "Min Chunk Size (chars)": min(c.char_count for c in chunks),
            "Max Chunk Size (chars)": max(c.char_count for c in chunks),
        })

        # Show first 3 chunks as preview
        print("  Preview (first 3 chunks):")
        for chunk in chunks[:3]:
            preview = chunk.text[:120].replace("\n", " ")
            print(f"    [{chunk.chunk_index}] {preview}...")
        print()
    else:
        print("  [WARNING] No chunks generated.\n")
        return

    # ── Step 3: Embed (optional) ──
    if args.embed:
        print("[STEP 3] Embedding Generation")
        from embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embedded_chunks = generator.generate(chunks)

        if embedded_chunks:
            print_summary("Embedding Results", {
                "Chunks Embedded": len(embedded_chunks),
                "Dimensions": embedded_chunks[0].dimensions,
                "Model": generator.model_name,
            })

            # Save to disk
            save_path = generator.save(
                embedded_chunks,
                output_path=args.save_embeddings,
            )
            print(f"  [SAVE] Embeddings saved to: {save_path}\n")
    else:
        print("  [INFO] Embedding generation skipped (use --embed to enable).\n")

    print("[OK] Pipeline complete!")


def run_index_mode(args: argparse.Namespace):
    """Ingest, partition, embed, and store document in FAISS and BM25 database."""
    if not args.file:
        logger.error("Error: --file is required for --index mode.")
        sys.exit(1)

    file_path = Path(args.file).resolve()
    logger.info(f"Indexing document: {file_path.name}")

    from embeddings import EmbeddingGenerator
    from indexing import FAISSIndexManager, BM25Searcher

    # 1. Ingest
    document = ingest(str(file_path))
    print(f"[INGEST] Ingested {document.metadata.get('filename')} ({document.word_count:,} words)")

    # 2. Chunk
    chunker_kwargs = {}
    if args.chunker == "recursive":
        if args.chunk_size:
            chunker_kwargs["chunk_size"] = args.chunk_size
        if args.chunk_overlap:
            chunker_kwargs["chunk_overlap"] = args.chunk_overlap
    elif args.chunker == "sliding_window":
        if args.chunk_size:
            chunker_kwargs["window_size"] = args.chunk_size
        if args.chunk_overlap:
            chunker_kwargs["step_size"] = args.chunk_size - args.chunk_overlap

    chunker = get_chunker(args.chunker, **chunker_kwargs)
    chunks = chunker.chunk(document)
    print(f"[CHUNK] Generated {len(chunks)} chunks using strategy '{args.chunker}'")

    # 3. Embed
    generator = EmbeddingGenerator()
    embedded_chunks = generator.generate(chunks, show_progress=True)

    # 4. Dense Vector Index (FAISS)
    faiss_mgr = FAISSIndexManager()
    faiss_mgr.add_chunks(embedded_chunks, persist=True)

    # 5. Sparse Keyword Index (BM25)
    bm25_searcher = BM25Searcher()
    # Rebuild by combining new chunks with any existing chunks in BM25 searcher cache
    combined_chunks = bm25_searcher._chunks + embedded_chunks
    bm25_searcher.build_index(combined_chunks, persist=True)

    print(f"[OK] Document successfully indexed in vector database! Total indexed chunks: {faiss_mgr._index.ntotal}")


def run_query_mode(question: str):
    """Perform hybrid search and stream Q&A answers with citations to standard output."""
    print(f"[QUERY] Question: {question}")
    print("[SEARCH] Searching index and generating response...\n")

    from embeddings import EmbeddingGenerator
    from indexing import FAISSIndexManager, BM25Searcher, HybridSearcher
    from qa import LLMClient, RAGEngine

    # Initialize RAG Pipeline components
    generator = EmbeddingGenerator()
    faiss_mgr = FAISSIndexManager()
    
    if faiss_mgr._index.ntotal == 0:
        print("[WARNING] The vector index is empty! Please index a document first using:")
        print("    python main.py --index -f <path-to-document>")
        return

    bm25_searcher = BM25Searcher()
    
    # If cold start populated FAISS but not BM25 index on disk
    if len(bm25_searcher._chunks) == 0:
        logger.info("Syncing BM25 cache from active FAISS database metadata...")
        from embeddings.embedding_generator import EmbeddedChunk
        chunks = []
        for idx in range(faiss_mgr._index.ntotal):
            chunk = EmbeddedChunk(
                text=faiss_mgr._chunk_texts[idx],
                chunk_index=faiss_mgr._chunk_indices[idx],
                embedding=[],
                metadata=faiss_mgr._metadata_map[idx],
            )
            chunks.append(chunk)
        bm25_searcher.build_index(chunks, persist=True)

    hybrid_searcher = HybridSearcher(faiss_mgr, bm25_searcher)
    rag_engine = RAGEngine(hybrid_searcher, generator)

    # Execute query
    t0 = sys.modules['time'].time()
    res = rag_engine.answer_query(question)
    latency = sys.modules['time'].time() - t0

    stream = res["stream"]
    retrieved_chunks = res["retrieved_chunks"]

    # Stream out tokens
    print("=" * 60)
    print("  GENERATED ANSWER (STREAMING)")
    print("=" * 60)
    for token in stream:
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n" + "=" * 60 + "\n")

    # Print citations
    print("[CITATIONS] CITED SOURCES:")
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata.get("filename", "unknown")
        print(f"  [{idx + 1}] Source: {source} | Chunk Index: {chunk.chunk_index}")
        snippet = chunk.text[:120].replace('\n', ' ')
        print(f"      Snippet: {snippet}...")
    
    print(f"\n[TIME] Retrieval & init time: {latency:.2f}s")


def run_eval_mode():
    """Execute strategy quantitative evaluations."""
    print("[EVAL] Running chunking strategy comparative benchmarks...")
    from evaluation.evaluate_chunking import run_evaluation
    run_evaluation()


def run_serve_mode():
    """Launch the FastAPI REST server using uvicorn."""
    import uvicorn
    settings = get_settings()
    print("\n[SERVER] Starting REST API Server & RAG Dashboard UI...")
    print("[LINK] Open http://127.0.0.1:8000 in your browser to access the dashboard!\n")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)


def main():
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.index:
            run_index_mode(args)
        elif args.query is not None:
            run_query_mode(args.query)
        elif args.eval:
            run_eval_mode()
        elif args.serve:
            run_serve_mode()
        else:
            # Default to basic ingestion pipeline if no mutual flags
            run_basic_pipeline(args)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Intelligent Document Q&A Engine — CLI Pipeline Runner.

Usage:
    python main.py --file <path> --chunker recursive|semantic|sliding_window [--embed]

Examples:
    python main.py --file docs/report.pdf --chunker recursive
    python main.py --file docs/notes.txt --chunker sliding_window --embed
    python main.py --file docs/paper.docx --chunker semantic --embed
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
        description="Intelligent Document Q&A Engine — Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the document file (PDF, DOCX, or TXT).",
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
        help="Generate OpenAI embeddings for chunks.",
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


def run_pipeline(args: argparse.Namespace):
    """Execute the document processing pipeline."""

    file_path = Path(args.file).resolve()
    logger.info(f"Starting pipeline for: {file_path.name}")

    # ── Step 1: Ingest ──────────────────────────────────────────────────
    print("\n📄 Step 1: Document Ingestion")
    document = ingest(str(file_path))

    print_summary("Document Info", {
        "Source": document.metadata.get("filename", "unknown"),
        "Type": document.metadata.get("file_type", "unknown"),
        "Characters": f"{document.char_count:,}",
        "Words": f"{document.word_count:,}",
    })

    # ── Step 2: Chunk ───────────────────────────────────────────────────
    print(f"✂️  Step 2: Chunking (strategy: {args.chunker})")

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
        print("  ⚠️  No chunks generated.\n")
        return

    # ── Step 3: Embed (optional) ────────────────────────────────────────
    if args.embed:
        print("🧠 Step 3: Embedding Generation")
        from embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embedded_chunks = generator.generate(chunks)

        if embedded_chunks:
            print_summary("Embedding Results", {
                "Chunks Embedded": len(embedded_chunks),
                "Dimensions": embedded_chunks[0].dimensions,
                "Model": generator.model,
            })

            # Save to disk
            save_path = generator.save(
                embedded_chunks,
                output_path=args.save_embeddings,
            )
            print(f"  💾 Embeddings saved to: {save_path}\n")
    else:
        print("  ℹ️  Embedding generation skipped (use --embed to enable).\n")

    print("✅ Pipeline complete!")


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

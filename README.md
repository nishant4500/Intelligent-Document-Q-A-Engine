# Intelligent Document Q&A Engine

A production-ready RAG (Retrieval-Augmented Generation) pipeline for ingesting documents, chunking text, and generating vector embeddings.

## Features

### Document Ingestion (US-102, US-103)
- **PDF** — Multi-page text extraction with metadata (PyPDF2)
- **DOCX** — Paragraph, table, and property extraction (python-docx)
- **TXT** — Auto-encoding detection, BOM stripping, whitespace normalization (chardet)

### Chunking Strategies (US-104, US-105, US-106)
| Strategy | Method | Speed | Best For |
|:---------|:-------|:------|:---------|
| **Recursive** | Character-based with separator hierarchy | Fast | General purpose (default) |
| **Semantic** | Embedding-similarity breakpoints | Slow | Topic-diverse documents |
| **Sliding Window** | Fixed-size overlapping windows | Fast | Dense information retrieval |

### Embedding Generation (US-107)
- OpenAI `text-embedding-3-small` (1536 dimensions)
- Batch processing with progress tracking
- Disk persistence for FAISS indexing

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the Pipeline
```bash
# Basic: Ingest + chunk a document
python main.py --file docs/report.pdf --chunker recursive

# With embeddings
python main.py --file docs/notes.txt --chunker sliding_window --embed

# Semantic chunking (requires API key)
python main.py --file docs/paper.docx --chunker semantic --embed

# Custom chunk sizes
python main.py --file docs/report.pdf --chunker recursive --chunk-size 1000 --chunk-overlap 100
```

---

## Project Structure

```
├── config/
│   └── settings.py              # Centralized Pydantic settings
├── ingestion/
│   ├── base_reader.py           # Abstract reader interface + Document model
│   ├── pdf_reader.py            # PDF ingestion (PyPDF2)
│   ├── docx_reader.py           # DOCX ingestion (python-docx)
│   └── txt_reader.py            # TXT ingestion (chardet)
├── chunking/
│   ├── base_chunker.py          # Abstract chunker interface + Chunk model
│   ├── recursive_chunker.py     # LangChain RecursiveCharacterTextSplitter
│   ├── semantic_chunker.py      # LangChain SemanticChunker (experimental)
│   └── sliding_window_chunker.py# Custom sliding-window implementation
├── embeddings/
│   └── embedding_generator.py   # OpenAI embedding generation + persistence
├── tests/
│   ├── test_ingestion.py        # Ingestion tests
│   ├── test_chunking.py         # Chunking tests
│   └── test_embeddings.py       # Embedding tests
├── main.py                      # CLI pipeline runner
├── requirements.txt             # Dependencies
└── .env.example                 # Environment variable template
```

---

## Configuration

All settings are centralized in `config/settings.py` and loaded from `.env`:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPENAI_API_KEY` | — | Required for embeddings and semantic chunking |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHUNK_SIZE` | `512` | Max characters per chunk (recursive) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (recursive) |
| `SLIDING_WINDOW_SIZE` | `512` | Window size (sliding window) |
| `SLIDING_WINDOW_STEP` | `256` | Step size (sliding window) |
| `SEMANTIC_BREAKPOINT_TYPE` | `percentile` | Breakpoint detection method |

---

## Testing

```bash
# Run all tests (except API-dependent ones)
pytest tests/ -v -m "not api"

# Run all tests including API integration
pytest tests/ -v

# Run specific test suites
pytest tests/test_ingestion.py -v
pytest tests/test_chunking.py -v
```

---

## API Usage

```python
from dotenv import load_dotenv
load_dotenv()

from ingestion import ingest
from chunking import get_chunker
from embeddings import EmbeddingGenerator

# 1. Ingest a document
doc = ingest("path/to/document.pdf")
print(f"Extracted {doc.word_count} words")

# 2. Chunk the document
chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(doc)
print(f"Generated {len(chunks)} chunks")

# 3. Generate embeddings
generator = EmbeddingGenerator()
embedded = generator.generate(chunks)
print(f"Created {len(embedded)} embeddings ({embedded[0].dimensions}D)")

# 4. Save for later use
generator.save(embedded)
```

---

## Authors

- **Nishant Dwivedi** — Document Ingestion & Chunking (US-102 to US-107)
- **Mehul Agarwal** — FAISS Indexing, Retrieval & API (US-108 to US-114)
- **Vidushi Negi** — Evaluation & Advanced Extensions (US-115 to US-121)

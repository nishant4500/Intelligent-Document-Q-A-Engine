"""
Quantitative Strategy Evaluation Script — US-118, US-120.

Compares Recursive, Sliding Window, and Semantic chunking strategies
on a sample document corpus, computing BLEU, ROUGE-L, Faithfulness,
and retrieval latencies.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from chunking import get_chunker
from config.settings import get_settings
from embeddings.embedding_generator import EmbeddingGenerator
from ingestion.base_reader import Document
from indexing.faiss_index import FAISSIndexManager
from indexing.hybrid_search import BM25Searcher, HybridSearcher
from qa.llm_client import LLMClient
from qa.rag_engine import RAGEngine
from evaluation.evaluator import RAGEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")

# ── Mock Corpus Data ────────────────────────────────────────────────────────

ARCHITECTURAL_CORPUS = """
# System Architecture Design Guidelines: Scalable Microservices

This document outlines the architectural specifications and design standards for our next-generation Enterprise SaaS platform. The system is designed to handle high-throughput transactions with sub-50ms latencies and 99.99% availability.

## 1. Microservices and API Gateway Ingress
The platform adopts a microservices architecture. All client requests are routed through a high-performance API Gateway built on top of Kong and NGINX. The API Gateway manages centralized cross-cutting concerns, specifically:
- Rate Limiting: Rate limits are enforced on a per-API-key basis, allowing up to 10,000 requests per hour for standard tier clients and 1,000,000 requests per hour for enterprise accounts.
- Authentication: OAuth 2.0 with JSON Web Tokens (JWT) is used. Tokens are signed using asymmetric RS256 algorithms and validated against a distributed JWKS endpoint cached in memory.
- Dynamic Routing: Requests are forwarded to internal services based on request path matching rules.

## 2. Distributed Database Partitioning and Replication
To sustain 20,000 concurrent writes per second, we implement a sharded database architecture.
- Write Path: Primary transaction records are written to a PostgreSQL cluster sharded across 16 database nodes. Sharding is executed via a consistent hashing algorithm using the account_uuid as the sharding key.
- Read Path: Read requests are handled by read-replicas, with each shard having three distinct replicas across three Availability Zones to prevent data loss.
- Caching Layer: We employ a multi-tier Redis cluster for caching. Session cache records expire after 15 minutes, whereas static configuration cache expires after 24 hours. Cache eviction employs the Least Recently Used (LRU) policy.

## 3. Asynchronous Event-Driven Messaging
Services communicate asynchronously via Apache Kafka.
- Message Brokering: Event topics are partitioned into 8 partitions to enable parallel processing by consumer groups.
- Message Delivery Guarantee: We enforce an At-Least-Once delivery semantics. Producers must set acks=all to ensure messages are replicated to all in-sync replicas before acknowledgement.
- Schema Registry: Confluent Schema Registry manages Apache Avro schemas, ensuring strict backward compatibility for all topic models.
- Dead Letter Queue: Messages failing processing three consecutive times are routed to a Dead Letter Queue (DLQ) for manual operator inspection.

## 4. Disaster Recovery and Backup Operations
Business continuity is governed by a strict Recovery Point Objective (RPO) and Recovery Time Objective (RTO).
- RPO Specification: The target RPO is set to 5 minutes. Transaction write-ahead logs (WAL) are streamed in real-time to AWS S3.
- RTO Specification: The target RTO is set to 2 hours. Service recovery is automated using Kubernetes multi-region failover.
- Backup Retention: Complete database snapshots are captured daily at 01:00 UTC and retained for exactly 30 days under strict GFS (Grandfather-Father-Son) rotation schedules.
"""

GOLD_DATASET = [
    {
        "query": "What are the rate limits enforced on standard tier clients at the API Gateway?",
        "reference": "At the API Gateway, rate limiting allows up to 10,000 requests per hour for standard tier clients (and 1,000,000 requests per hour for enterprise accounts).",
    },
    {
        "query": "Explain the sharding strategy and sharding key used to achieve high transaction throughput.",
        "reference": "The transactional PostgreSQL cluster is sharded across 16 database nodes. Sharding is executed using a consistent hashing algorithm with account_uuid as the sharding key.",
    },
    {
        "query": "What caching policy is used in Redis and how long do session cache records expire?",
        "reference": "Redis uses a Least Recently Used (LRU) cache eviction policy. Session cache records are configured to expire after 15 minutes, while static config cache expires after 24 hours.",
    },
    {
        "query": "What message delivery guarantee does the Kafka broker enforce and what producer setting is required?",
        "reference": "The Kafka broker enforces At-Least-Once delivery semantics. Producers must configure acks=all to ensure replication to all in-sync replicas before acknowledgement.",
    },
    {
        "query": "What are the specific RPO and RTO specifications for disaster recovery in the system?",
        "reference": "Disaster recovery specifications dictate a Recovery Point Objective (RPO) of 5 minutes and a Recovery Time Objective (RTO) of 2 hours.",
    }
]


# ── Evaluation Runner ───────────────────────────────────────────────────────

def run_evaluation() -> dict[str, Any]:
    """Execute evaluation comparing Recursive, Sliding Window, and Semantic chunkers."""
    logger.info("Starting strategy evaluation suite...")

    # Initialize evaluator
    evaluator = RAGEvaluator()
    generator = EmbeddingGenerator()
    llm = LLMClient(provider="mock")  # Use mock to ensure reproducibility and cost-free execution
    
    # Wrap corpus as ingestion document
    document = Document(
        text=ARCHITECTURAL_CORPUS,
        metadata={
            "filename": "system_design_specs.txt",
            "file_type": ".txt",
            "source": "architectural_specs.txt",
            "file_size_bytes": len(ARCHITECTURAL_CORPUS),
        }
    )

    strategies = ["recursive", "sliding_window", "semantic"]
    comparison_results = {}

    for strategy in strategies:
        logger.info(f"\nEvaluating strategy: {strategy}...")
        
        # 1. Initialize Chunker
        chunker_kwargs = {}
        if strategy == "recursive":
            chunker_kwargs = {"chunk_size": 350, "chunk_overlap": 50}
        elif strategy == "sliding_window":
            chunker_kwargs = {"window_size": 350, "step_size": 200}
        
        chunker = get_chunker(strategy, **chunker_kwargs)
        chunks = chunker.chunk(document)
        logger.info(f"Generated {len(chunks)} chunks using {strategy}.")

        # 2. Generate Embeddings
        embedded_chunks = generator.generate(chunks, show_progress=False)

        # 3. Create Temp FAISS and BM25 search indices
        # We save index in a temp folder to avoid overwriting production index
        tmp_dir = f"./output/eval_index_{strategy}"
        faiss_mgr = FAISSIndexManager(index_dir=tmp_dir, dimensions=generator.dimensions)
        faiss_mgr.clear()
        faiss_mgr.add_chunks(embedded_chunks, persist=True)

        bm25_searcher = BM25Searcher(index_path=f"{tmp_dir}/bm25_index.pkl")
        bm25_searcher.clear()
        bm25_searcher.build_index(embedded_chunks, persist=True)

        hybrid_searcher = HybridSearcher(faiss_mgr, bm25_searcher)
        rag_engine = RAGEngine(hybrid_searcher, generator, llm)

        # 4. Run Gold Queries
        query_metrics = []
        latencies = []

        for item in GOLD_DATASET:
            query = item["query"]
            ref_answer = item["reference"]

            t0 = time.perf_counter()
            
            # Run RAG
            res = rag_engine.answer_query(query, k=3)
            
            # Consume stream to get full answer
            full_answer_list = list(res["stream"])
            full_answer = "".join(full_answer_list)
            
            # Calculate Latency
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            # Context strings
            contexts = [c.text for c in res["retrieved_chunks"]]

            # Compute metrics
            metrics = evaluator.evaluate_response(
                candidate=full_answer,
                reference=ref_answer,
                contexts=contexts,
            )
            query_metrics.append(metrics)

            logger.info(f"Query: {query[:30]}... | BLEU: {metrics['bleu']:.2f} | ROUGE-L: {metrics['rouge_l']:.2f} | Faithfulness: {metrics['faithfulness']:.2f} | Latency: {latency_ms:.1f}ms")

        # 5. Aggregate metrics
        avg_bleu = sum(m["bleu"] for m in query_metrics) / len(query_metrics)
        avg_rouge = sum(m["rouge_l"] for m in query_metrics) / len(query_metrics)
        avg_faith = sum(m["faithfulness"] for m in query_metrics) / len(query_metrics)
        avg_latency = sum(latencies) / len(latencies)

        comparison_results[strategy] = {
            "chunks_count": len(chunks),
            "avg_bleu": avg_bleu,
            "avg_rouge_l": avg_rouge,
            "avg_faithfulness": avg_faith,
            "avg_latency_ms": avg_latency,
            "individual_queries": query_metrics,
        }

        # Clear temp databases
        faiss_mgr.clear()
        bm25_searcher.clear()
        
        # Remove temp directory
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass

    # 6. Output Table to CLI
    print("\n" + "=" * 80)
    print("  STRATEGY RETRIEVAL COMPARISON BENCHMARK")
    print("=" * 80)
    print(f" {'Strategy':<18} │ {'Chunks':<6} │ {'BLEU':<6} │ {'ROUGE-L':<8} │ {'Faithful':<8} │ {'Latency':<8}")
    print("─" * 80)
    for name, res in comparison_results.items():
        print(
            f" {name:<18} │ {res['chunks_count']:<6} │ {res['avg_bleu']:.3f}  │ {res['avg_rouge_l']:.3f}   │ {res['avg_faithfulness']:.3f}   │ {res['avg_latency_ms']:.1f}ms"
        )
    print("=" * 80 + "\n")

    return comparison_results


if __name__ == "__main__":
    run_evaluation()

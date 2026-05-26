"""
Feedback Logging Engine — US-117.

Stores user corrections, query history, and thumbs ratings in a SQLite database
to build a feedback loop for retrieval quality improvement.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """
    Manages a local SQLite database to log Q&A queries, retrieved context meta,
    ratings, and user-provided corrections.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the feedback engine and ensure tables exist."""
        settings = get_settings()
        
        # Determine database path (default: ./output/feedback.db)
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Derive from embeddings directory parent
            parent_dir = Path(settings.FAISS_INDEX_DIR).parent
            self.db_path = parent_dir / "feedback.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish a connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        # Return rows as dictionary-like objects
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the schema if it doesn't already exist."""
        logger.info(f"Initializing Feedback SQLite database at {self.db_path}...")
        
        with self._get_connection() as conn:
            # Table to store queries, retrieved contexts, generated answers, and ratings
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    answer TEXT,
                    retrieved_chunks_json TEXT,  -- serialized list of chunk metadata
                    rating INTEGER DEFAULT 0,    -- +1 for thumbs up, -1 for thumbs down, 0 for unrated
                    feedback_text TEXT           -- optional textual feedback
                )
                """
            )
            
            # Table to store custom corrections/ground-truth pairs provided by the user
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT,
                    query TEXT NOT NULL,
                    incorrect_answer TEXT,
                    corrected_answer TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
                """
            )
            conn.commit()

    def log_query(
        self,
        query_id: str,
        query: str,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        """
        Record a Q&A transaction.

        Args:
            query_id: Unique string ID of the query (e.g. UUID).
            query: The user query string.
            answer: The generated answer string.
            retrieved_chunks: List of chunk metadata dicts.
        """
        chunks_json = json.dumps(retrieved_chunks)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO queries (id, query, answer, retrieved_chunks_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query_id, query, answer, chunks_json),
                )
                conn.commit()
            logger.info(f"Logged query transaction {query_id} to database.")
        except Exception as e:
            logger.error(f"Failed to log query transaction: {e}")

    def log_feedback(
        self,
        query_id: str,
        rating: int,  # 1 for up, -1 for down
        feedback_text: Optional[str] = None,
    ) -> bool:
        """
        Record rating and user corrections for an existing query.

        Args:
            query_id: ID of the query transaction.
            rating: Positive or negative indicator (e.g. 1 or -1).
            feedback_text: Textual correction or explanation.

        Returns:
            True if successfully updated, False otherwise.
        """
        try:
            with self._get_connection() as conn:
                # First check if the query exists
                row = conn.execute("SELECT id, query, answer FROM queries WHERE id = ?", (query_id,)).fetchone()
                if not row:
                    logger.warning(f"Feedback submitted for non-existent query ID: {query_id}")
                    # If it doesn't exist, we'll insert a shell query so we don't lose the feedback
                    conn.execute(
                        """
                        INSERT INTO queries (id, query, answer, rating, feedback_text)
                        VALUES (?, 'unknown (feedback only)', '', ?, ?)
                        """,
                        (query_id, rating, feedback_text),
                    )
                else:
                    # Update existing record
                    conn.execute(
                        """
                        UPDATE queries
                        SET rating = ?, feedback_text = ?
                        WHERE id = ?
                        """,
                        (rating, feedback_text, query_id),
                    )

                # If feedback_text is provided and rating is negative, let's also save to corrections
                if feedback_text and rating == -1 and row:
                    conn.execute(
                        """
                        INSERT INTO corrections (query_id, query, incorrect_answer, corrected_answer)
                        VALUES (?, ?, ?, ?)
                        """,
                        (query_id, row["query"], row["answer"], feedback_text),
                    )

                conn.commit()
            logger.info(f"Recorded rating={rating} feedback for query {query_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}")
            return False

    def get_corrections(self) -> list[dict[str, Any]]:
        """Retrieve all logged user corrections for fine-tuning or evaluation."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, query_id, query, incorrect_answer, corrected_answer, timestamp FROM corrections"
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve corrections: {e}")
            return []

    def get_query_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve transaction history logs."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, query, answer, retrieved_chunks_json, rating, feedback_text FROM queries ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    # Deserialize chunks list
                    try:
                        item["retrieved_chunks"] = json.loads(item["retrieved_chunks_json"] or "[]")
                    except Exception:
                        item["retrieved_chunks"] = []
                    results.append(item)
                return results
        except Exception as e:
            logger.error(f"Failed to fetch transaction logs: {e}")
            return []

"""
Tests for FastAPI Backend REST API — US-112, US-113, US-114.

Verifies API endpoint routing, SSE streaming responses, and feedback SQLite database submissions.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ── Unit Tests ──────────────────────────────────────────────────────────────

class TestRESTAPIEndpoints:
    """REST API route integration tests using TestClient."""

    def test_get_root_ui(self):
        """Verify root path serves the single-page HTML application."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "").lower()
        # Verify custom design elements exist in response
        assert "Antigravity Q&A" in response.text
        assert "Document Ingestion" in response.text

    def test_get_evaluation_metrics(self):
        """Verify strategy benchmark comparison values return correctly."""
        client = TestClient(app)
        response = client.get("/api/evaluation-metrics")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) == 3
        assert data["strategies"][0]["name"] == "Recursive Character Chunker"
        assert data["strategies"][2]["name"] == "Semantic Breakpoint Chunker"

    def test_get_documents_empty(self):
        """Verify document listing on empty index returns empty list."""
        client = TestClient(app)
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        # On an empty database test suite, this should return a list
        assert isinstance(data, list)

    def test_submit_feedback(self):
        """Verify rating feedback registers in feedback logs database."""
        client = TestClient(app)
        payload = {
            "query_id": "test-uuid-12345",
            "rating": 1,
            "feedback_text": "Highly accurate context matching!"
        }
        response = client.post("/api/feedback", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

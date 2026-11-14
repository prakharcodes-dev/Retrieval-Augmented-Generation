"""
Unit tests for PerformanceTracker:
Query latency recording, percentile calculation, method breakdown,
ingestion throughput measurement, and trace logging.
"""

import pytest
from app.core.performance_tracker import PerformanceTracker


def test_performance_tracker_query_recording_and_summary():
    tracker = PerformanceTracker(max_history=10)
    tracker.reset()

    # Record 3 queries
    tracker.record_query(
        query="What is the Golden Temple?",
        retrieval_method="Hybrid RRF",
        retrieval_ms=30.0,
        generation_ms=500.0,
        chunks_retrieved=5,
        model_used="gemini-2.5-flash",
    )
    tracker.record_query(
        query="Who published the guide?",
        retrieval_method="Hybrid RRF",
        retrieval_ms=20.0,
        generation_ms=400.0,
        chunks_retrieved=5,
        model_used="gemini-2.5-flash",
    )
    tracker.record_query(
        query="List landmarks",
        retrieval_method="BM25 Search",
        retrieval_ms=10.0,
        generation_ms=300.0,
        chunks_retrieved=3,
        model_used="gemini-2.5-flash",
    )

    summary = tracker.get_summary()
    assert summary["total_queries"] == 3
    # Avg retrieval = (30 + 20 + 10) / 3 = 20.0 ms
    assert summary["avg_retrieval_ms"] == 20.0
    # Avg generation = (500 + 400 + 300) / 3 = 400.0 ms
    assert summary["avg_generation_ms"] == 400.0
    # Avg total = 420.0 ms
    assert summary["avg_total_ms"] == 420.0

    # Method breakdown
    assert "Hybrid RRF" in summary["by_method"]
    assert summary["by_method"]["Hybrid RRF"]["count"] == 2
    assert summary["by_method"]["Hybrid RRF"]["avg_retrieval_ms"] == 25.0

    assert "BM25 Search" in summary["by_method"]
    assert summary["by_method"]["BM25 Search"]["count"] == 1
    assert summary["by_method"]["BM25 Search"]["avg_retrieval_ms"] == 10.0

    # Recent traces
    traces = tracker.get_recent_query_records()
    assert len(traces) == 3
    assert traces[0]["query_preview"].startswith("List landmarks")


def test_performance_tracker_ingestion_throughput():
    tracker = PerformanceTracker(max_history=10)
    tracker.reset()

    rec = tracker.record_ingestion(
        filename="test_manual.pdf",
        file_size_bytes=1024 * 1024,
        total_pages=10,
        chunks_stored=50,
        duration_seconds=2.0,
    )

    assert rec.throughput_chunks_per_sec == 25.0
    summary = tracker.get_summary()
    assert summary["total_documents_ingested"] == 1
    assert summary["avg_ingestion_throughput_chunks_sec"] == 25.0


def test_performance_tracker_reset():
    tracker = PerformanceTracker(max_history=10)
    tracker.record_query("test", "hybrid", 10.0, 10.0, 2)
    assert tracker.get_summary()["total_queries"] == 1

    tracker.reset()
    assert tracker.get_summary()["total_queries"] == 0
    assert tracker.get_recent_query_records() == []

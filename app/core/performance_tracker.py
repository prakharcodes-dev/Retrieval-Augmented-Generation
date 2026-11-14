"""
Lightweight, thread-safe performance monitor for tracking retrieval, generation, and ingestion latencies.
Computes average latency, percentiles, method breakdowns, and ingestion throughput.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class QueryTimingRecord:
    """Record of a single RAG query's end-to-end latency metrics."""

    timestamp: str
    query_preview: str
    retrieval_method: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    chunks_retrieved: int
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionTimingRecord:
    """Record of a document ingestion pipeline run."""

    timestamp: str
    filename: str
    file_size_bytes: int
    total_pages: int
    chunks_stored: int
    duration_seconds: float
    throughput_chunks_per_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """Thread-safe performance statistics recorder."""

    _INSTANCE: Optional[PerformanceTracker] = None
    _LOCK = threading.Lock()

    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self.max_history = max_history
        self._query_records: deque[QueryTimingRecord] = deque(maxlen=max_history)
        self._ingestion_records: deque[IngestionTimingRecord] = deque(maxlen=max_history)

    @classmethod
    def get_instance(cls) -> PerformanceTracker:
        with cls._LOCK:
            if cls._INSTANCE is None:
                cls._INSTANCE = cls()
            return cls._INSTANCE

    def record_query(
        self,
        query: str,
        retrieval_method: str,
        retrieval_ms: float,
        generation_ms: float,
        chunks_retrieved: int,
        model_used: str = "default",
    ) -> QueryTimingRecord:
        """Records latency measurements for a completed question-answering workflow."""
        total_ms = retrieval_ms + generation_ms
        preview = query.strip()[:60] + "..." if len(query.strip()) > 60 else query.strip()
        record = QueryTimingRecord(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            query_preview=preview,
            retrieval_method=retrieval_method,
            retrieval_ms=max(0.0, retrieval_ms),
            generation_ms=max(0.0, generation_ms),
            total_ms=max(0.0, total_ms),
            chunks_retrieved=chunks_retrieved,
            model_used=model_used,
        )
        with self._lock:
            self._query_records.append(record)
        return record

    def record_ingestion(
        self,
        filename: str,
        file_size_bytes: int,
        total_pages: int,
        chunks_stored: int,
        duration_seconds: float,
    ) -> IngestionTimingRecord:
        """Records throughput and execution time for document ingestion."""
        duration = max(0.001, duration_seconds)
        throughput = chunks_stored / duration
        record = IngestionTimingRecord(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            filename=filename,
            file_size_bytes=file_size_bytes,
            total_pages=total_pages,
            chunks_stored=chunks_stored,
            duration_seconds=round(duration, 3),
            throughput_chunks_per_sec=round(throughput, 2),
        )
        with self._lock:
            self._ingestion_records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Calculates aggregate summary statistics across all recorded queries."""
        with self._lock:
            records = list(self._query_records)
            ingest_records = list(self._ingestion_records)

        avg_throughput = 0.0
        if ingest_records:
            avg_throughput = sum(r.throughput_chunks_per_sec for r in ingest_records) / len(ingest_records)

        if not records:
            return {
                "total_queries": 0,
                "avg_retrieval_ms": 0.0,
                "avg_generation_ms": 0.0,
                "avg_total_ms": 0.0,
                "p50_total_ms": 0.0,
                "p95_total_ms": 0.0,
                "by_method": {},
                "total_documents_ingested": len(ingest_records),
                "avg_ingestion_throughput_chunks_sec": round(avg_throughput, 2),
            }

        retrieval_times = [r.retrieval_ms for r in records]
        gen_times = [r.generation_ms for r in records if r.generation_ms > 0]
        total_times = sorted([r.total_ms for r in records])

        n = len(records)
        p50 = total_times[int(0.50 * (n - 1))]
        p95 = total_times[int(0.95 * (n - 1))]

        # Method breakdowns
        by_method: Dict[str, Dict[str, Any]] = {}
        for r in records:
            m = r.retrieval_method
            if m not in by_method:
                by_method[m] = {"count": 0, "latencies": []}
            by_method[m]["count"] += 1
            by_method[m]["latencies"].append(r.retrieval_ms)

        method_summary: Dict[str, Dict[str, float]] = {}
        for m, data in by_method.items():
            lats = data["latencies"]
            method_summary[m] = {
                "count": data["count"],
                "avg_retrieval_ms": round(sum(lats) / len(lats), 2),
                "min_ms": round(min(lats), 2),
                "max_ms": round(max(lats), 2),
            }

        avg_throughput = 0.0
        if ingest_records:
            avg_throughput = sum(r.throughput_chunks_per_sec for r in ingest_records) / len(ingest_records)

        return {
            "total_queries": n,
            "avg_retrieval_ms": round(sum(retrieval_times) / n, 2),
            "avg_generation_ms": round(sum(gen_times) / len(gen_times), 2) if gen_times else 0.0,
            "avg_total_ms": round(sum(total_times) / n, 2),
            "p50_total_ms": round(p50, 2),
            "p95_total_ms": round(p95, 2),
            "by_method": method_summary,
            "total_documents_ingested": len(ingest_records),
            "avg_ingestion_throughput_chunks_sec": round(avg_throughput, 2),
        }

    def get_recent_query_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._query_records)[-limit:]
        return [r.to_dict() for r in reversed(records)]

    def get_recent_ingestion_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._ingestion_records)[-limit:]
        return [r.to_dict() for r in reversed(records)]

    def reset(self) -> None:
        with self._lock:
            self._query_records.clear()
            self._ingestion_records.clear()


# Global tracker instance
performance_tracker = PerformanceTracker.get_instance()

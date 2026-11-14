"""
Benchmark and comparison system for Vector Search, BM25 Search, and Hybrid RRF Retrieval.
Executes identical test datasets across all three retrieval modalities under identical conditions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RAGEvaluator, EvaluationReport


class _DirectAdapter:
    """Adapts a vector store or sparse index search function into the standard retriever contract."""

    def __init__(self, search_fn, name: str):
        self.search_fn = search_fn
        self.name = name

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or 5
        return self.search_fn(query, k)


@dataclass
class ComparisonReport:
    """Holds comparative benchmark results across Vector, BM25, and Hybrid retrieval."""

    timestamp: str
    top_k: int
    total_queries: int
    method_reports: Dict[str, EvaluationReport]
    best_overall_method: str
    best_latency_method: str
    best_hit_rate_method: str
    key_findings: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Constructs a clean comparison table for side-by-side metric inspection."""
        rows = []
        for method_name, r in self.method_reports.items():
            rows.append({
                "Retrieval Method": method_name,
                f"Precision@{self.top_k}": round(r.mean_precision, 4),
                f"Recall@{self.top_k}": round(r.mean_recall, 4),
                "Hit Rate": round(r.hit_rate, 4),
                "MRR": round(r.mrr, 4),
                "Context Precision": round(r.mean_context_precision, 4),
                "Avg Latency (ms)": round(r.mean_retrieval_latency_ms, 2),
            })
        return pd.DataFrame(rows)

    def to_csv(self) -> str:
        return self.to_dataframe().to_csv(index=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "top_k": self.top_k,
            "total_queries": self.total_queries,
            "best_overall_method": self.best_overall_method,
            "best_latency_method": self.best_latency_method,
            "best_hit_rate_method": self.best_hit_rate_method,
            "key_findings": self.key_findings,
            "summary_table": self.to_dataframe().to_dict(orient="records"),
            "method_reports": {k: v.to_dict() for k, v in self.method_reports.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class RetrievalComparator:
    """Compares Vector Search, BM25 Search, and Hybrid RRF on the same benchmark dataset."""

    def __init__(self, top_k: int = 5):
        self.top_k = max(1, int(top_k))
        self.evaluator = RAGEvaluator(top_k=self.top_k)

    def compare_methods(
        self,
        dataset: EvaluationDataset,
        vector_store: Any,
        sparse_index: Any,
        hybrid_retriever: Any,
        top_k: Optional[int] = None,
    ) -> ComparisonReport:
        """
        Executes Vector (dense), BM25 (sparse), and Hybrid (RRF) search over the dataset.
        Measures accuracy and latency, determines the top-performing method, and generates a report.
        """
        k = max(1, int(top_k if top_k is not None else self.top_k))

        # 1. Define retrieval adapters for the 3 methods
        def vector_search(q: str, limit: int):
            return vector_store.query(query_text=q, top_k=limit)

        def bm25_search(q: str, limit: int):
            return sparse_index.search(q, top_k=limit)

        methods = {
            "Vector Search (Dense)": _DirectAdapter(vector_search, "Vector Search (Dense)"),
            "BM25 Search (Sparse)": _DirectAdapter(bm25_search, "BM25 Search (Sparse)"),
            "Hybrid Search (RRF)": hybrid_retriever,
        }

        # 2. Run evaluation on each method
        reports: Dict[str, EvaluationReport] = {}
        for method_name, adapter in methods.items():
            reports[method_name] = self.evaluator.evaluate_retrieval(
                dataset=dataset,
                retriever=adapter,
                top_k=k,
                method_name=method_name,
            )

        # 3. Determine best methods based on empirical results
        # Composite score: (MRR * 0.4) + (Hit Rate * 0.3) + (Recall * 0.2) + (Precision * 0.1)
        best_overall = max(
            reports.keys(),
            key=lambda m: (
                reports[m].mrr * 0.4
                + reports[m].hit_rate * 0.3
                + reports[m].mean_recall * 0.2
                + reports[m].mean_precision * 0.1
            ),
        )

        best_latency = min(reports.keys(), key=lambda m: reports[m].mean_retrieval_latency_ms)
        best_hit_rate = max(reports.keys(), key=lambda m: (reports[m].hit_rate, reports[m].mrr))

        findings: List[str] = [
            f"Best overall retrieval performance: '{best_overall}' based on composite MRR and recall.",
            f"Fastest retrieval method: '{best_latency}' with {reports[best_latency].mean_retrieval_latency_ms:.2f}ms average latency.",
            f"Highest coverage (Hit Rate): '{best_hit_rate}' with {reports[best_hit_rate].hit_rate * 100:.1f}% hit rate.",
        ]

        if "Hybrid Search (RRF)" in reports and "Vector Search (Dense)" in reports:
            diff = reports["Hybrid Search (RRF)"].hit_rate - reports["Vector Search (Dense)"].hit_rate
            if diff > 0:
                findings.append(f"Hybrid RRF fusion improved hit rate by {diff * 100:.1f}% over standalone dense vector search.")

        return ComparisonReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            top_k=k,
            total_queries=len(dataset),
            method_reports=reports,
            best_overall_method=best_overall,
            best_latency_method=best_latency,
            best_hit_rate_method=best_hit_rate,
            key_findings=findings,
        )

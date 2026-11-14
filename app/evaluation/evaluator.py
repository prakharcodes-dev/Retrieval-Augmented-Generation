"""
Execution engine for RAG pipeline evaluation, reporting, and metric aggregation.
Generates structured EvaluationReport objects with JSON and CSV export capabilities.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.evaluation.dataset import EvaluationDataset, EvaluationSample
from app.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate,
    calculate_reciprocal_rank,
    calculate_mrr,
    calculate_context_precision,
    calculate_faithfulness,
    calculate_answer_relevancy,
    is_chunk_relevant,
)


@dataclass
class EvaluationReport:
    """Holds summary metrics and granular per-query results of an evaluation run."""

    name: str
    timestamp: str
    total_queries: int
    top_k: int
    mean_precision: float
    mean_recall: float
    hit_rate: float
    mrr: float
    mean_context_precision: float
    mean_retrieval_latency_ms: float
    mean_generation_latency_ms: Optional[float] = None
    mean_faithfulness: Optional[float] = None
    mean_answer_relevancy: Optional[float] = None
    query_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts detailed query-level evaluation results to a pandas DataFrame."""
        rows = []
        for r in self.query_results:
            rows.append({
                "Sample ID": r.get("id"),
                "Category": r.get("category"),
                "Question": r.get("question"),
                "Precision@K": round(r.get("precision", 0.0), 4),
                "Recall@K": round(r.get("recall", 0.0), 4),
                "Hit": int(r.get("hit", 0)),
                "Reciprocal Rank": round(r.get("reciprocal_rank", 0.0), 4),
                "Context Precision": round(r.get("context_precision", 0.0), 4),
                "Retrieval Latency (ms)": round(r.get("retrieval_latency_ms", 0.0), 2),
                "Faithfulness": round(r["faithfulness"], 4) if r.get("faithfulness") is not None else None,
                "Relevancy": round(r["answer_relevancy"], 4) if r.get("answer_relevancy") is not None else None,
                "Retrieved Chunks": len(r.get("retrieved_chunks", [])),
                "Generated Answer": (r.get("generated_answer") or "")[:100] + "..." if r.get("generated_answer") else "",
            })
        return pd.DataFrame(rows)

    def to_csv(self) -> str:
        """Exports the detailed results to CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)


class RAGEvaluator:
    """Evaluates RAG retrieval and answer generation using ground truth test datasets."""

    def __init__(self, top_k: int = 5):
        self.top_k = max(1, int(top_k))

    def evaluate_retrieval(
        self,
        dataset: EvaluationDataset,
        retriever: Any,
        top_k: Optional[int] = None,
        method_name: str = "Retrieval Evaluation",
    ) -> EvaluationReport:
        """
        Runs retrieval evaluation across all samples in the dataset.
        Computes Precision@K, Recall@K, Hit Rate, MRR, Context Precision, and latency.
        """
        k = max(1, int(top_k if top_k is not None else self.top_k))
        results: List[Dict[str, Any]] = []

        reciprocal_ranks: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        hits: List[float] = []
        context_precisions: List[float] = []
        latencies_ms: List[float] = []

        for sample in dataset:
            t0 = time.perf_counter()
            try:
                retrieved = retriever.retrieve(sample.question, top_k=k)
            except Exception as e:
                retrieved = []
            retrieval_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(retrieval_ms)

            # Evaluate metrics
            p = calculate_precision_at_k(
                retrieved,
                expected_snippets=sample.expected_context_snippets,
                relevant_docs=sample.relevant_docs,
                relevant_chunk_ids=sample.relevant_chunk_ids,
                k=k,
            )
            r = calculate_recall_at_k(
                retrieved,
                expected_snippets=sample.expected_context_snippets,
                relevant_docs=sample.relevant_docs,
                relevant_chunk_ids=sample.relevant_chunk_ids,
                k=k,
            )
            hit = calculate_hit_rate(
                retrieved,
                expected_snippets=sample.expected_context_snippets,
                relevant_docs=sample.relevant_docs,
                relevant_chunk_ids=sample.relevant_chunk_ids,
                k=k,
            )
            rr = calculate_reciprocal_rank(
                retrieved,
                expected_snippets=sample.expected_context_snippets,
                relevant_docs=sample.relevant_docs,
                relevant_chunk_ids=sample.relevant_chunk_ids,
                k=k,
            )
            cp = calculate_context_precision(
                retrieved,
                expected_snippets=sample.expected_context_snippets,
                relevant_docs=sample.relevant_docs,
                relevant_chunk_ids=sample.relevant_chunk_ids,
                k=k,
            )

            precisions.append(p)
            recalls.append(r)
            hits.append(hit)
            reciprocal_ranks.append(rr)
            context_precisions.append(cp)

            results.append({
                "id": sample.id,
                "category": sample.category,
                "question": sample.question,
                "expected_answer": sample.expected_answer,
                "precision": p,
                "recall": r,
                "hit": hit,
                "reciprocal_rank": rr,
                "context_precision": cp,
                "retrieval_latency_ms": retrieval_ms,
                "retrieved_chunks": retrieved,
            })

        n = len(dataset) or 1
        return EvaluationReport(
            name=method_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_queries=len(dataset),
            top_k=k,
            mean_precision=sum(precisions) / n,
            mean_recall=sum(recalls) / n,
            hit_rate=sum(hits) / n,
            mrr=calculate_mrr(reciprocal_ranks),
            mean_context_precision=sum(context_precisions) / n,
            mean_retrieval_latency_ms=sum(latencies_ms) / n,
            query_results=results,
        )

    def evaluate_pipeline(
        self,
        dataset: EvaluationDataset,
        retriever: Any,
        generator: Optional[Any] = None,
        top_k: Optional[int] = None,
        evaluate_answers: bool = True,
        method_name: str = "End-to-End RAG Evaluation",
    ) -> EvaluationReport:
        """
        Runs complete RAG evaluation including retrieval and optional answer generation.
        Calculates retrieval metrics plus Faithfulness and Answer Relevancy.
        """
        report = self.evaluate_retrieval(dataset, retriever, top_k=top_k, method_name=method_name)
        if not evaluate_answers or generator is None:
            return report

        faithfulness_scores: List[float] = []
        relevancy_scores: List[float] = []
        gen_latencies_ms: List[float] = []

        for idx, sample in enumerate(dataset):
            query_record = report.query_results[idx]
            retrieved = query_record.get("retrieved_chunks", [])

            t0 = time.perf_counter()
            try:
                gen_result = generator.generate(sample.question, retrieved)
                gen_ms = (time.perf_counter() - t0) * 1000.0
                ans_text = getattr(gen_result, "answer", str(gen_result))
            except Exception as e:
                gen_ms = (time.perf_counter() - t0) * 1000.0
                ans_text = f"Error during generation: {e}"

            gen_latencies_ms.append(gen_ms)
            f_score = calculate_faithfulness(ans_text, retrieved)
            rel_score = calculate_answer_relevancy(ans_text, sample.question, sample.expected_answer)

            faithfulness_scores.append(f_score)
            relevancy_scores.append(rel_score)

            query_record["generated_answer"] = ans_text
            query_record["generation_latency_ms"] = gen_ms
            query_record["faithfulness"] = f_score
            query_record["answer_relevancy"] = rel_score

        n = len(dataset) or 1
        report.mean_generation_latency_ms = sum(gen_latencies_ms) / n if gen_latencies_ms else None
        report.mean_faithfulness = sum(faithfulness_scores) / n if faithfulness_scores else None
        report.mean_answer_relevancy = sum(relevancy_scores) / n if relevancy_scores else None

        return report

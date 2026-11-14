"""RAG Evaluation, Benchmarking, and Retrieval Comparison System."""

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
from app.evaluation.evaluator import RAGEvaluator, EvaluationReport
from app.evaluation.comparator import RetrievalComparator, ComparisonReport

__all__ = [
    "EvaluationDataset",
    "EvaluationSample",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_hit_rate",
    "calculate_reciprocal_rank",
    "calculate_mrr",
    "calculate_context_precision",
    "calculate_faithfulness",
    "calculate_answer_relevancy",
    "is_chunk_relevant",
    "RAGEvaluator",
    "EvaluationReport",
    "RetrievalComparator",
    "ComparisonReport",
]

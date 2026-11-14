"""
Unit tests for 3-way retrieval method comparison:
Vector Search (Dense), BM25 Search (Sparse), and Hybrid Search (RRF).
Verifies metric computation, report generation, CSV/JSON export, and best method selection.
"""

import json
import pytest
import pandas as pd
from app.evaluation.dataset import EvaluationDataset, EvaluationSample
from app.evaluation.comparator import RetrievalComparator, ComparisonReport


class _MockVectorStore:
    def __init__(self, dense_results):
        self.dense_results = dense_results

    def query(self, query_text: str, top_k: int, where=None):
        return self.dense_results[:top_k]


class _MockSparseIndex:
    def __init__(self, sparse_results):
        self.sparse_results = sparse_results

    def search(self, query: str, top_k: int, where=None):
        return self.sparse_results[:top_k]


class _MockHybridRetriever:
    def __init__(self, hybrid_results):
        self.hybrid_results = hybrid_results

    def retrieve(self, query: str, top_k: int = 5):
        return self.hybrid_results[:top_k]


def test_retrieval_comparator_3way():
    sample1 = EvaluationSample(
        id="c1",
        question="Where is the Golden Temple?",
        expected_answer="Amritsar, Punjab.",
        expected_context_snippets=["Golden Temple destination"],
        category="factual",
    )
    dataset = EvaluationDataset(samples=[sample1])

    dense_chunk = {"chunk_id": "d1", "text": "Noise content about hotels."}
    sparse_chunk = {"chunk_id": "s1", "text": "Mentions Golden Temple destination guide."}
    hybrid_chunk = {"chunk_id": "h1", "text": "Detailed Golden Temple destination guide."}

    v_store = _MockVectorStore([dense_chunk])
    s_index = _MockSparseIndex([sparse_chunk])
    h_retriever = _MockHybridRetriever([hybrid_chunk])

    comparator = RetrievalComparator(top_k=1)
    report = comparator.compare_methods(
        dataset=dataset,
        vector_store=v_store,
        sparse_index=s_index,
        hybrid_retriever=h_retriever,
        top_k=1,
    )

    assert isinstance(report, ComparisonReport)
    assert len(report.method_reports) == 3
    assert "Vector Search (Dense)" in report.method_reports
    assert "BM25 Search (Sparse)" in report.method_reports
    assert "Hybrid Search (RRF)" in report.method_reports

    # BM25 and Hybrid should have hit rate 1.0, Vector 0.0
    assert report.method_reports["BM25 Search (Sparse)"].hit_rate == 1.0
    assert report.method_reports["Hybrid Search (RRF)"].hit_rate == 1.0
    assert report.method_reports["Vector Search (Dense)"].hit_rate == 0.0

    # DataFrame formatting check
    df = report.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "Precision@1" in df.columns
    assert "MRR" in df.columns

    # CSV and JSON export verification
    csv_str = report.to_csv()
    assert "Retrieval Method" in csv_str
    assert "Hybrid Search (RRF)" in csv_str

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["top_k"] == 1
    assert "best_overall_method" in parsed
    assert len(parsed["key_findings"]) >= 2

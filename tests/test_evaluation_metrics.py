"""
Unit tests for RAG evaluation metric calculations:
Precision@K, Recall@K, Hit Rate, MRR, Context Precision, Faithfulness, and Answer Relevancy.
"""

import pytest
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


def test_is_chunk_relevant_by_snippet_substring():
    chunk = {
        "chunk_id": "c1",
        "text": "The Ministry of Tourism published the official heritage guide in 2024.",
        "metadata": {"source": "guide.pdf"},
        "source": "guide.pdf",
    }
    assert is_chunk_relevant(chunk, expected_snippets=["Ministry of Tourism"])
    assert not is_chunk_relevant(chunk, expected_snippets=["Department of Transportation"])


def test_is_chunk_relevant_by_doc_name():
    chunk = {
        "chunk_id": "c1",
        "text": "General tourism content.",
        "metadata": {"source": "sample_travel_guide.pdf"},
        "source": "sample_travel_guide.pdf",
    }
    assert is_chunk_relevant(chunk, relevant_docs=["sample_travel_guide.pdf"])
    assert not is_chunk_relevant(chunk, relevant_docs=["medical_manual.pdf"])


def test_is_chunk_relevant_by_chunk_id():
    chunk = {
        "chunk_id": "target_chunk_123",
        "text": "Some text",
        "metadata": {},
    }
    assert is_chunk_relevant(chunk, relevant_chunk_ids=["target_chunk_123", "other"])
    assert not is_chunk_relevant(chunk, relevant_chunk_ids=["wrong_id"])


def test_precision_at_k():
    chunks = [
        {"chunk_id": "c1", "text": "Relevant snippet about Tirumala Palace."},
        {"chunk_id": "c2", "text": "Completely irrelevant content about gardening."},
        {"chunk_id": "c3", "text": "Relevant note on Tirumala Palace architecture."},
        {"chunk_id": "c4", "text": "Irrelevant sports news."},
    ]
    snippets = ["Tirumala Palace"]

    p2 = calculate_precision_at_k(chunks, expected_snippets=snippets, k=2)
    assert p2 == 0.5  # 1 out of 2

    p4 = calculate_precision_at_k(chunks, expected_snippets=snippets, k=4)
    assert p4 == 0.5  # 2 out of 4

    p_empty = calculate_precision_at_k([], expected_snippets=snippets, k=3)
    assert p_empty == 0.0


def test_recall_at_k():
    chunks = [
        {"chunk_id": "c1", "text": "Information about Golden Temple."},
        {"chunk_id": "c2", "text": "Details on Taj Mahal monument."},
    ]
    snippets = ["Golden Temple", "Taj Mahal", "Tirumala Palace"]

    # 2 out of 3 target snippets retrieved
    r = calculate_recall_at_k(chunks, expected_snippets=snippets, k=2)
    assert round(r, 2) == 0.67


def test_hit_rate():
    chunks_hit = [
        {"chunk_id": "c1", "text": "Irrelevant noise"},
        {"chunk_id": "c2", "text": "Match for Ministry of Tourism"},
    ]
    chunks_miss = [
        {"chunk_id": "c1", "text": "Irrelevant noise"},
        {"chunk_id": "c2", "text": "More unrelated content"},
    ]

    assert calculate_hit_rate(chunks_hit, expected_snippets=["Ministry of Tourism"], k=2) == 1.0
    assert calculate_hit_rate(chunks_miss, expected_snippets=["Ministry of Tourism"], k=2) == 0.0


def test_reciprocal_rank_and_mrr():
    chunks = [
        {"chunk_id": "c1", "text": "Irrelevant intro"},
        {"chunk_id": "c2", "text": "Relevant info on Golden Temple"},
        {"chunk_id": "c3", "text": "Another mention of Golden Temple"},
    ]
    # First hit is at rank 2 -> RR = 1/2 = 0.5
    rr = calculate_reciprocal_rank(chunks, expected_snippets=["Golden Temple"], k=3)
    assert rr == 0.5

    # Test MRR across multiple queries
    rrs = [1.0, 0.5, 0.25, 0.0]
    assert calculate_mrr(rrs) == 0.4375


def test_context_precision():
    # Relevant chunk at rank 1, irrelevant at rank 2
    chunks_high = [
        {"chunk_id": "c1", "text": "Match target snippet"},
        {"chunk_id": "c2", "text": "Irrelevant info"},
    ]
    cp_high = calculate_context_precision(chunks_high, expected_snippets=["target snippet"], k=2)
    assert cp_high == 1.0  # (1/1 * 1) / 1 = 1.0

    # Irrelevant at rank 1, relevant at rank 2
    chunks_low = [
        {"chunk_id": "c1", "text": "Irrelevant info"},
        {"chunk_id": "c2", "text": "Match target snippet"},
    ]
    cp_low = calculate_context_precision(chunks_low, expected_snippets=["target snippet"], k=2)
    assert cp_low == 0.5  # (1/2 * 1) / 1 = 0.5


def test_faithfulness_calculation():
    context = [
        {"chunk_id": "c1", "text": "The Taj Mahal is an ivory-white marble mausoleum located in Agra."}
    ]

    # Faithful answer supported by context
    ans_faithful = "The Taj Mahal is an ivory-white marble mausoleum in Agra."
    assert calculate_faithfulness(ans_faithful, context) >= 0.8

    # Unfaithful answer hallucinating unrelated facts
    ans_hallucinated = "The Eiffel Tower was constructed in Paris during the nineteenth century."
    assert calculate_faithfulness(ans_hallucinated, context) <= 0.3

    # Refusal on missing context is 100% faithful
    ans_refusal = "I couldn't find sufficient evidence in the provided documents to answer that question."
    assert calculate_faithfulness(ans_refusal, []) == 1.0


def test_answer_relevancy_calculation():
    question = "Who published the official travel guide?"
    expected = "The Ministry of Tourism, Government of India published the guide."

    matching_answer = "The Ministry of Tourism, Government of India published the official travel guide."
    assert calculate_answer_relevancy(matching_answer, question, expected) >= 0.7

    empty_answer = ""
    assert calculate_answer_relevancy(empty_answer, question, expected) == 0.0

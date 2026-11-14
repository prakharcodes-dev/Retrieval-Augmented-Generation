"""
Unit tests for evaluation dataset loading, schema validation, serialization, and filtering.
"""

import json
import pytest
from pathlib import Path
from app.evaluation.dataset import EvaluationDataset, EvaluationSample


def test_evaluation_sample_validations():
    # Valid sample
    sample = EvaluationSample(
        id="sample_1",
        question="What is RAG?",
        expected_answer="Retrieval-Augmented Generation.",
        category="definition",
    )
    assert sample.id == "sample_1"
    d = sample.to_dict()
    assert d["question"] == "What is RAG?"

    # Missing ID raises ValueError
    with pytest.raises(ValueError, match="must have an 'id'"):
        EvaluationSample.from_dict({"question": "Test?"})

    # Missing question raises ValueError
    with pytest.raises(ValueError, match="non-empty 'question'"):
        EvaluationSample.from_dict({"id": "s1", "question": ""})


def test_dataset_filtering_and_summary():
    s1 = EvaluationSample(id="1", question="Q1", expected_answer="A1", category="factual")
    s2 = EvaluationSample(id="2", question="Q2", expected_answer="A2", category="numerical")
    s3 = EvaluationSample(id="3", question="Q3", expected_answer="A3", category="factual")

    dataset = EvaluationDataset(samples=[s1, s2, s3], name="Test DS")
    assert len(dataset) == 3

    factual_ds = dataset.filter_by_category("factual")
    assert len(factual_ds) == 2
    assert all(s.category == "factual" for s in factual_ds)

    summary = dataset.summary()
    assert summary["total_samples"] == 3
    assert summary["categories"]["factual"] == 2
    assert summary["categories"]["numerical"] == 1


def test_dataset_save_and_load(tmp_path):
    s = EvaluationSample(
        id="eval_test",
        question="Sample question?",
        expected_answer="Sample answer.",
        relevant_docs=["doc1.pdf"],
        expected_context_snippets=["important snippet"],
        category="overview",
    )
    ds = EvaluationDataset(samples=[s], name="Exported DS")
    target_file = tmp_path / "saved_eval.json"
    ds.save_to_file(target_file)

    loaded = EvaluationDataset.load_from_file(target_file)
    assert len(loaded) == 1
    assert loaded[0].id == "eval_test"
    assert loaded[0].expected_context_snippets == ["important snippet"]


def test_default_eval_dataset_integrity():
    default_path = Path("data/evaluation/default_eval_dataset.json")
    assert default_path.exists(), "Default benchmark dataset file must exist"

    dataset = EvaluationDataset.load_from_file(default_path)
    assert len(dataset) >= 20, f"Expected at least 20 test cases, found {len(dataset)}"

    categories = dataset.get_categories()
    assert "factual" in categories
    assert "overview" in categories

    for s in dataset:
        assert s.id.startswith("eval_")
        assert len(s.question.strip()) > 5
        assert len(s.expected_answer.strip()) > 0

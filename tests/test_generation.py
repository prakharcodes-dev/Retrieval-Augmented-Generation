import pytest
from app.generation.answer_generator import RAGGenerator, CitedAnswer


@pytest.fixture
def mock_generator():
    return RAGGenerator(llm_provider="mock")


def test_format_context(mock_generator):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The company policy requires 14 days notice for paid time off.",
            "metadata": {"source": "hr_policy.pdf", "page_number": 4}
        },
        {
            "chunk_id": "c2",
            "text": "Expense reports must be submitted before the 5th of each month.",
            "metadata": {"source": "finance_guide.pdf", "page_number": 10}
        }
    ]

    formatted = mock_generator.format_context(chunks)

    assert "[Excerpt 1 | Document: hr_policy.pdf, Page: 4]" in formatted
    assert "14 days notice" in formatted
    assert "[Excerpt 2 | Document: finance_guide.pdf, Page: 10]" in formatted
    assert "5th of each month" in formatted


def test_generate_empty_retrieval(mock_generator):
    answer = mock_generator.generate(query="What is the refund policy?", retrieved_chunks=[])

    assert isinstance(answer, CitedAnswer)
    assert answer.refusal is True
    assert answer.answer == RAGGenerator.REFUSAL_MESSAGE
    assert answer.citations == []


def test_generate_unsupported_question_refusal(mock_generator):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Acme Corporation offers 15 days of Paid Time Off (PTO) per calendar year.",
            "metadata": {"source": "employee_handbook.pdf", "page_number": 2}
        }
    ]

    # Question about CEO salary — not mentioned in the context excerpt
    answer = mock_generator.generate(
        query="What is the company CEO salary?",
        retrieved_chunks=chunks
    )

    assert isinstance(answer, CitedAnswer)
    assert answer.refusal is True
    assert answer.answer == RAGGenerator.REFUSAL_MESSAGE


def test_generate_with_context(mock_generator):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Standard business hours for all full-time Acme Corporation employees are 9:00 AM to 5:00 PM EST.",
            "metadata": {"source": "employee_handbook.pdf", "page_number": 1}
        }
    ]

    answer = mock_generator.generate(
        query="What are the standard business hours?",
        retrieved_chunks=chunks
    )

    assert isinstance(answer, CitedAnswer)
    assert answer.refusal is False
    assert "Sources:" in answer.answer
    assert "[1] employee_handbook.pdf — Page 1" in answer.answer

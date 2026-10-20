import pytest
import app.generation.answer_generator as answer_generator_module
from app.core.exceptions import GenerationError, TransientGenerationError
import sys
from types import ModuleType, SimpleNamespace
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


def test_gemini_auth_key_with_aq_prefix_is_not_rejected(mock_generator, monkeypatch):
    genai_types = ModuleType("google.genai")
    genai_types.types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "google.genai", genai_types)

    mock_generator.llm_provider = "gemini"
    mock_generator.api_key = "AQ.test-auth-key"
    mock_generator._gemini_types = genai_types.types
    mock_generator._gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text="Gemini answer")
        )
    )

    assert mock_generator._call_llm("test prompt") == "Gemini answer"


@pytest.mark.parametrize("status", [503, 429])
def test_gemini_transient_errors_retry_three_times_then_raise(mock_generator, monkeypatch, status):
    calls = []

    class ProviderError(Exception):
        code = status

    def fail(**kwargs):
        calls.append(kwargs)
        raise ProviderError("temporary")

    mock_generator.llm_provider = "gemini"
    mock_generator.groq_api_key = ""
    mock_generator.api_key = "configured"
    mock_generator._gemini_types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    mock_generator._gemini_client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    monkeypatch.setattr(answer_generator_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(answer_generator_module.random, "uniform", lambda low, high: high)

    with pytest.raises(TransientGenerationError, match="temporarily unavailable") as error:
        mock_generator._call_llm("keep this question")
    assert "Add GROQ_API_KEY" in str(error.value)
    assert len(calls) == 3


def test_gemini_falls_back_to_groq_after_primary_transient_failures(mock_generator, monkeypatch):
    calls = []

    class Unavailable(Exception):
        code = 503

    def gemini_generate(**kwargs):
        calls.append(("gemini", kwargs))
        raise Unavailable("busy")

    def groq_create(**kwargs):
        calls.append(("groq", kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="grounded fallback answer"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=groq_create)))
    groq_module = ModuleType("groq")
    groq_module.Groq = lambda **kwargs: fake_client
    monkeypatch.setitem(sys.modules, "groq", groq_module)

    mock_generator.llm_provider = "gemini"
    mock_generator.model_name = "primary-model"
    mock_generator.groq_fallback_model = "qwen/qwen3.8-27b"
    mock_generator.groq_api_key = "gsk-test-fallback-key"
    mock_generator.api_key = "configured"
    mock_generator._gemini_types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    mock_generator._gemini_client = SimpleNamespace(models=SimpleNamespace(generate_content=gemini_generate))
    monkeypatch.setattr(answer_generator_module.time, "sleep", lambda seconds: None)

    assert mock_generator._call_llm("same RAG context") == "grounded fallback answer"
    assert [provider for provider, _ in calls] == ["gemini", "gemini", "gemini", "groq"]
    assert calls[-1][1]["model"] == "qwen/qwen3.8-27b"
    assert calls[-1][1]["messages"][0]["content"] == "same RAG context"
    assert mock_generator._last_model_used == "Groq qwen/qwen3.8-27b"


def test_groq_rate_limit_is_classified_as_transient():
    class TooManyRequests(Exception):
        status_code = 429

    assert RAGGenerator._is_transient_groq_error(TooManyRequests("rate limited"))


def test_gemini_permanent_model_error_is_not_retried(mock_generator, monkeypatch):
    calls = []

    class NotFound(Exception):
        code = 404

    def fail(**kwargs):
        calls.append(kwargs)
        raise NotFound()

    mock_generator.llm_provider = "gemini"
    mock_generator.api_key = "configured"
    mock_generator._gemini_types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    mock_generator._gemini_client = SimpleNamespace(models=SimpleNamespace(generate_content=fail))
    with pytest.raises(GenerationError, match="unavailable"):
        mock_generator._call_llm("question")
    assert len(calls) == 1


def test_gemini_invalid_key_is_actionable_and_not_retried(mock_generator):
    calls = []

    class Unauthorized(Exception):
        code = 401

    def reject(**kwargs):
        calls.append(kwargs)
        raise Unauthorized("provider response may contain sensitive details")

    mock_generator.llm_provider = "gemini"
    mock_generator.api_key = "configured"
    mock_generator._gemini_types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    mock_generator._gemini_client = SimpleNamespace(models=SimpleNamespace(generate_content=reject))
    with pytest.raises(GenerationError, match="rejected GEMINI_API_KEY") as error:
        mock_generator._call_llm("question")
    assert "sensitive details" not in str(error.value)
    assert len(calls) == 1

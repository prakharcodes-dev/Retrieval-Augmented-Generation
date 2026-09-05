import pytest
from app.generation.citation_formatter import CitationFormatter


def test_extract_citations_unique():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {"source": "refund_policy.pdf", "page_number": 4}
        },
        {
            "chunk_id": "c2",
            "metadata": {"source": "refund_policy.pdf", "page_number": 4}  # Duplicate source & page
        },
        {
            "chunk_id": "c3",
            "metadata": {"source": "terms.txt", "page_number": None}
        }
    ]

    citations = CitationFormatter.extract_citations(chunks)

    assert len(citations) == 2
    assert citations[0] == {"source": "refund_policy.pdf", "page_number": 4, "chunk_id": "c1"}
    assert citations[1] == {"source": "terms.txt", "page_number": None, "chunk_id": "c3"}


def test_format_citations_block_pdf_and_txt():
    citations = [
        {"source": "refund_policy.pdf", "page_number": 4},
        {"source": "employee_handbook.txt", "page_number": None}
    ]

    block = CitationFormatter.format_citations_block(citations)

    assert "[1] refund_policy.pdf — Page 4" in block
    assert "[2] employee_handbook.txt" in block
    assert "Page None" not in block


def test_format_citations_block_empty():
    assert CitationFormatter.format_citations_block([]) == ""

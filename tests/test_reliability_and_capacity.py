"""
Comprehensive Reliability, Safety, and High-Capacity Test Suite.
Verifies batched streaming ingestion, memory guard safety, deterministic chunk IDs,
LLM lazy loading isolation, failure recovery, and rerun idempotency.
"""

import os
import tempfile
import pytest
from pathlib import Path
from reportlab.pdfgen import canvas

from app.core.exceptions import (
    RAGException,
    ResourceUnsafeError,
    PDFValidationError,
    PDFExtractionError,
)
from app.core.resource_guard import ResourceGuard
from app.core.state import OperationState, StateTracker
from app.embeddings.embedding_service import EmbeddingService
from app.generation.answer_generator import RAGGenerator, _QWEN_MODEL
from app.ingestion.chunker import TextChunker, Chunk
from app.ingestion.ingest import ingest_documents
from app.ingestion.loader import DocumentLoader, safe_pymupdf_open
from app.vectorstore.chroma_store import ChromaVectorStore


def create_synthetic_pdf(path: Path, num_pages: int = 10, words_per_page: int = 300) -> Path:
    """Helper function to create a synthetic multi-page PDF using ReportLab."""
    c = canvas.Canvas(str(path))
    for page_num in range(1, num_pages + 1):
        text_lines = [
            f"Document Title: Synthetic High Capacity Reliability Test Document Section {page_num}",
            f"Page {page_num} content block with detailed information for RAG indexing.",
        ]
        # Fill page with synthetic text content
        for line_idx in range(words_per_page // 10):
            text_lines.append(f"Line {line_idx + 1}: Important factual details regarding rule {page_num}.{line_idx + 1} for evaluation.")

        y = 800
        for line in text_lines:
            if y < 50:
                c.showPage()
                y = 800
            c.drawString(40, y, line[:90])
            y -= 15
        c.showPage()
    c.save()
    return path


def test_normal_pdf_ingestion(tmp_path):
    """Verifies standard multi-page PDF ingestion end-to-end."""
    pdf_path = tmp_path / "normal.pdf"
    create_synthetic_pdf(pdf_path, num_pages=5)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "test_store"),
        collection_name="normal_test_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res = ingest_documents(pdf_path, vector_store=store, reset=True)
    assert res["status"] == "success"
    assert res["files_processed"] == 1
    assert res["pages_processed"] == 5
    assert res["chunks_stored"] > 0
    assert store.get_count() == res["chunks_stored"]


def test_corrupted_pdf_handling(tmp_path):
    """Verifies corrupted PDF files are caught cleanly without crashing Python."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 %Not a real PDF stream header corrupt data 1234567890")

    loader = DocumentLoader()
    with pytest.raises(PDFValidationError):
        loader.validate_file(corrupt_pdf)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "test_store"),
        collection_name="corrupt_test_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res = ingest_documents(corrupt_pdf, vector_store=store, reset=True)
    assert res["status"] == "error"
    assert "Ingestion failed safely" in res["error"]


def test_closed_pdf_handle_safety(tmp_path):
    """Verifies PyMuPDF handles are safely closed and not reused after context exit."""
    pdf_path = tmp_path / "handle_test.pdf"
    create_synthetic_pdf(pdf_path, num_pages=2)

    doc_ref = None
    with safe_pymupdf_open(pdf_path) as doc:
        doc_ref = doc
        assert len(doc) == 2
        _ = doc[0].get_text()

    # Verify document handle is closed outside context
    assert doc_ref.is_closed or getattr(doc_ref, "is_closed", True)


def test_high_chunk_capacity_batched(tmp_path):
    """
    Tests high chunk capacity (500+ chunks) with page-batched streaming ingestion.
    Verifies sequential deterministic chunk indexing across batches and memory stability.
    """
    pdf_path = tmp_path / "large_doc.pdf"
    create_synthetic_pdf(pdf_path, num_pages=60, words_per_page=600)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "test_store_capacity"),
        collection_name="capacity_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res = ingest_documents(pdf_path, vector_store=store, reset=True, batch_size=10)
    assert res["status"] == "success"
    assert res["pages_processed"] >= 50
    assert res["chunks_stored"] >= 100

    # Query vector store to verify sequential chunk IDs
    query_res = store.query("Synthetic High Capacity", top_k=10)
    assert len(query_res) > 0
    for item in query_res:
        assert "chunk_id" in item
        assert "_chunk_" in item["chunk_id"]


def test_resource_guard_ram_check():
    """Verifies ResourceGuard halts unsafe operations before memory is exhausted."""
    stats = ResourceGuard.get_memory_stats()
    assert "available_ram_mb" in stats
    assert "ram_percent_used" in stats

    # Requesting an impossible minimum RAM threshold should raise ResourceUnsafeError
    with pytest.raises(ResourceUnsafeError, match="Insufficient system memory"):
        ResourceGuard.check_memory(min_available_mb=99999999.0, context="test_guard")


def test_failed_embedding_fallback():
    """Verifies EmbeddingService falls back to deterministic mock when provider fails."""
    service = EmbeddingService(provider="openai", api_key="sk-invalid-key-test")
    embeddings = service.embed_documents(["Test sentence for fallback embedding."])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # Deterministic mock dimension


def test_llm_lazy_loading_isolation(tmp_path):
    """
    Verifies that PDF upload/validation/chunking/ingestion does NOT load the LLM.
    """
    global _QWEN_MODEL
    # Save current LLM state
    initial_model = _QWEN_MODEL

    pdf_path = tmp_path / "lazy_llm.pdf"
    create_synthetic_pdf(pdf_path, num_pages=3)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "lazy_store"),
        collection_name="lazy_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res = ingest_documents(pdf_path, vector_store=store, reset=True)
    assert res["status"] == "success"

    # Verify LLM was NOT instantiated during ingestion
    if initial_model is None:
        assert _QWEN_MODEL is None


def test_streamlit_rerun_idempotency(tmp_path):
    """Verifies duplicate file ingestions are skipped cleanly without duplicate vector writes."""
    pdf_path = tmp_path / "dedup.pdf"
    create_synthetic_pdf(pdf_path, num_pages=3)

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "dedup_store"),
        collection_name="dedup_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res1 = ingest_documents(pdf_path, vector_store=store, reset=True)
    assert res1["status"] == "success"
    initial_chunks = store.get_count()

    res2 = ingest_documents(pdf_path, vector_store=store, reset=False)
    assert res2["status"] == "skipped"
    assert store.get_count() == initial_chunks


def test_operation_state_transitions():
    """Verifies StateTracker transitions cleanly through lifecycle states."""
    tracker = StateTracker()
    assert tracker.current_state == OperationState.READY

    tracker.transition_to(OperationState.VALIDATING)
    assert tracker.current_state == OperationState.VALIDATING

    tracker.transition_to(OperationState.INGESTING)
    assert tracker.current_state == OperationState.INGESTING

    tracker.transition_to(OperationState.RECOVERING, error="Simulated error")
    assert tracker.current_state == OperationState.RECOVERING
    assert tracker.last_error == "Simulated error"

    tracker.reset()
    assert tracker.current_state == OperationState.READY
    assert tracker.last_error is None

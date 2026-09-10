import pytest
from pathlib import Path
from app.ingestion.loader import DocumentLoader, DocumentPage
from app.ingestion.cleaner import TextCleaner


def test_document_loader_invalid_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("non_existent_file.pdf")


def test_document_loader_unsupported_type(tmp_path):
    loader = DocumentLoader()
    doc_path = tmp_path / "sample.xyz"
    doc_path.write_text("Hello world", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_file(doc_path)


def test_document_loader_txt_file(tmp_path):
    loader = DocumentLoader()
    txt_path = tmp_path / "company_rules.txt"
    txt_path.write_text("Rule 1: Standard hours are 9 AM to 5 PM EST.", encoding="utf-8")

    pages = loader.load_file(txt_path)

    assert len(pages) == 1
    page = pages[0]
    assert "Standard hours" in page.text
    assert page.metadata["file_type"] == "txt"
    assert page.metadata["filename"] == "company_rules.txt"
    assert page.metadata["page_number"] is None
    assert "document_id" in page.metadata
    assert "ingestion_timestamp" in page.metadata


def test_text_cleaner_fix_unspaced_pdf_text():
    cleaner = TextCleaner()
    raw_unspaced = "Inprecovidyears,thesharestradedincreasedmarginally.Butinpostcovidyearsitincreasedtremendously."

    cleaned = cleaner.clean(raw_unspaced)

    assert "In pre" in cleaned or "Inprecovidyears" not in cleaned
    assert "the shares traded" in cleaned or "thesharestraded" not in cleaned
    assert "post covid" in cleaned or "postcovid" not in cleaned


def test_text_cleaner_whitespace_normalization():
    cleaner = TextCleaner()
    raw_text = "   Paragraph 1 header.   \n\n\n\n\n   Paragraph 2 details.   "

    cleaned = cleaner.clean(raw_text)

    assert "Paragraph 1 header." in cleaned
    assert "Paragraph 2 details." in cleaned
    assert "\n\n\n" not in cleaned


def test_document_loader_file_hash(tmp_path):
    loader = DocumentLoader()
    doc_path = tmp_path / "test_hash.txt"
    doc_path.write_text("Unique file content for hash testing", encoding="utf-8")

    pages = loader.load_file(doc_path)

    assert len(pages) == 1
    assert "file_hash" in pages[0].metadata
    assert len(pages[0].metadata["file_hash"]) == 64


def test_ingest_documents_deduplication(tmp_path):
    from app.ingestion.ingest import ingest_documents
    from app.vectorstore.chroma_store import ChromaVectorStore
    from app.embeddings.embedding_service import EmbeddingService

    doc_path = tmp_path / "dedup_test.txt"
    doc_path.write_text("Deduplication test file content line 1.", encoding="utf-8")

    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "test_store"),
        collection_name="dedup_coll",
        embedding_service=EmbeddingService(provider="mock")
    )

    res1 = ingest_documents(doc_path, vector_store=store, reset=True)
    assert res1["status"] == "success"
    assert res1["chunks_stored"] == 1

    res2 = ingest_documents(doc_path, vector_store=store, reset=False)
    assert res2["status"] == "skipped"
    assert res2["chunks_stored"] == 0


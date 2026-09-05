import pytest
from app.ingestion.chunker import TextChunker, Chunk, validate_chunks
from app.ingestion.loader import DocumentPage


def test_chunker_default_token_size():
    chunker = TextChunker(chunk_size=700, chunk_overlap=100)
    assert chunker.chunk_size == 700
    assert chunker.chunk_overlap == 100


def test_chunker_basic_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "The quick brown fox jumps over the lazy dog. " * 30
    metadata = {
        "document_id": "doc123",
        "filename": "sample.pdf",
        "file_type": "pdf",
        "source": "sample.pdf",
        "title": "Sample Document",
        "page_number": 1
    }

    chunks = chunker.split_text(sample_text, metadata)

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.metadata["document_id"] == "doc123"
        assert chunk.metadata["filename"] == "sample.pdf"
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["chunk_index"] == idx
        assert "doc123" in chunk.chunk_id
        assert chunk.chunk_id.endswith(f"_chunk_{idx:03d}")
        assert chunk.token_count > 0


def test_validate_chunks_success():
    chunks = [
        Chunk(
            text="Valid chunk text excerpt.",
            metadata={"document_id": "doc1", "chunk_index": 0, "source": "doc1.pdf"},
            chunk_id="doc1_chunk_000"
        ),
        Chunk(
            text="Second valid chunk text excerpt.",
            metadata={"document_id": "doc1", "chunk_index": 1, "source": "doc1.pdf"},
            chunk_id="doc1_chunk_001"
        )
    ]
    assert validate_chunks(chunks) is True


def test_validate_chunks_duplicate_id_raises():
    chunks = [
        Chunk(text="Chunk 1", metadata={"chunk_index": 0}, chunk_id="same_id"),
        Chunk(text="Chunk 2", metadata={"chunk_index": 1}, chunk_id="same_id")
    ]
    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        validate_chunks(chunks)


def test_chunk_documents():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    pages = [
        DocumentPage(
            text="Page 1 text about company policies and core working hours.",
            metadata={"document_id": "d1", "source": "handbook.pdf", "page_number": 1}
        ),
        DocumentPage(
            text="Page 2 text about paid time off and leave request policies.",
            metadata={"document_id": "d1", "source": "handbook.pdf", "page_number": 2}
        )
    ]

    chunks = chunker.chunk_documents(pages)

    assert len(chunks) >= 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[-1].metadata["page_number"] == 2

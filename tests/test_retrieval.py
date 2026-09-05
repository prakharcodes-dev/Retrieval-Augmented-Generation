import pytest
import shutil
from pathlib import Path
from app.ingestion.chunker import Chunk
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.vector_retriever import VectorRetriever


@pytest.fixture
def temp_vector_store(tmp_path):
    """Fixture providing an isolated ephemeral ChromaVectorStore for testing."""
    persist_dir = str(tmp_path / "test_vectorstore")
    mock_embedder = EmbeddingService(provider="mock")
    store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="test_collection",
        embedding_service=mock_embedder
    )
    yield store
    try:
        shutil.rmtree(persist_dir)
    except Exception:
        pass


def test_vector_store_add_and_count(temp_vector_store):
    chunks = [
        Chunk(
            text="Python is an interpreted high-level programming language.",
            metadata={"document_id": "d1", "source": "python.pdf", "page_number": 1, "chunk_index": 0},
            chunk_id="d1_chunk_000"
        ),
        Chunk(
            text="ChromaDB is an open-source embedding database for AI apps.",
            metadata={"document_id": "d2", "source": "chroma.pdf", "page_number": 1, "chunk_index": 0},
            chunk_id="d2_chunk_000"
        )
    ]

    added = temp_vector_store.add_chunks(chunks)
    assert added == 2
    assert temp_vector_store.get_count() == 2


def test_vector_store_query_top_k(temp_vector_store):
    chunks = [
        Chunk(
            text="Vector databases store high-dimensional embeddings for fast nearest-neighbor search.",
            metadata={"document_id": "rg", "source": "rag_guide.pdf", "page_number": 5, "chunk_index": 0},
            chunk_id="rg_chunk_000"
        ),
        Chunk(
            text="Large language models generate coherent text based on contextual prompts.",
            metadata={"document_id": "rg", "source": "rag_guide.pdf", "page_number": 12, "chunk_index": 1},
            chunk_id="rg_chunk_001"
        )
    ]
    temp_vector_store.add_chunks(chunks)

    results = temp_vector_store.query("Tell me about vector database search", top_k=1)
    assert len(results) == 1
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert results[0]["metadata"]["source"] == "rag_guide.pdf"
    assert results[0]["page_number"] == 5


def test_retriever_empty_store(temp_vector_store):
    retriever = VectorRetriever(vector_store=temp_vector_store)
    results = retriever.retrieve("Any question", top_k=3)
    assert results == []


def test_retriever_configurable_top_k(temp_vector_store):
    chunks = [
        Chunk(
            text=f"Chunk paragraph content number {i}",
            metadata={"document_id": "doc1", "source": "doc.pdf", "page_number": i, "chunk_index": i - 1},
            chunk_id=f"doc1_chunk_{i:03d}"
        )
        for i in range(1, 10)
    ]
    temp_vector_store.add_chunks(chunks)

    retriever = VectorRetriever(vector_store=temp_vector_store)

    res_2 = retriever.retrieve("paragraph content", top_k=2)
    assert len(res_2) == 2

    res_5 = retriever.retrieve("paragraph content", top_k=5)
    assert len(res_5) == 5

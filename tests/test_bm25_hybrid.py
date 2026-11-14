import pytest

from app.core.exceptions import RetrievalError
from app.ingestion.chunker import Chunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vector_retriever import VectorRetriever
from app.application.document_service import DocumentService
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore


def test_bm25_persistence_retrieval_metadata_and_delete(tmp_path):
    path = tmp_path / "bm25.sqlite3"
    index = BM25Index(path)
    index.upsert_chunks([
        Chunk("The rare quasar protocol controls orbital telemetry.",
              {"document_id": "doc-a", "source": "manual.pdf", "page_number": 7}, "a-1"),
        Chunk("The kitchen recipe uses flour and water.",
              {"document_id": "doc-b", "source": "recipe.txt", "page_number": None}, "b-1"),
    ])
    assert index.search("quasar telemetry", 3)[0]["chunk_id"] == "a-1"
    reopened = BM25Index(path)
    assert reopened.search("recipe flour", 3)[0]["metadata"]["source"] == "recipe.txt"
    reopened.delete_document("doc-a")
    assert [r["chunk_id"] for r in reopened.search("quasar telemetry", 3)] == []


def _chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"document_id": "doc", "chunk_index": int(chunk_id[-1]), "source": "source.pdf", "page_number": 2},
        "document_id": "doc",
        "source": "source.pdf",
        "page_number": 2,
    }


class _FakeSparse:
    def __init__(self, results):
        self.results = results

    def sync_from_store(self, store):
        pass

    def search(self, query, top_k, where=None):
        return self.results[:top_k]


class _FakeStore:
    persist_directory = "."

    def __init__(self, dense):
        self.dense = dense

    def get_count(self):
        return 100

    def query(self, query_text, top_k, where=None):
        return self.dense[:top_k]


def test_rrf_merges_and_deduplicates_dense_and_sparse_candidates():
    dense = [_chunk("c1", "semantic result"), _chunk("c2", "both methods")]
    sparse = [_chunk("c2", "both methods"), _chunk("c3", "exact term")]
    retriever = VectorRetriever(
        _FakeStore(dense), sparse_index=_FakeSparse(sparse),
        dense_candidate_k=10, sparse_candidate_k=10, rrf_k=10,
    )
    results = retriever.retrieve("meaning", top_k=3)
    assert [item["chunk_id"] for item in results] == ["c2", "c1", "c3"]
    assert len({item["chunk_id"] for item in results}) == 3
    assert results[0]["metadata"]["page_number"] == 2
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_dense_or_sparse_empty_is_safe():
    dense = [_chunk("c1", "dense only")]
    assert len(VectorRetriever(_FakeStore(dense), _FakeSparse([])).retrieve("x", 2)) == 1
    assert VectorRetriever(_FakeStore([]), _FakeSparse([])).retrieve("x", 2) == []


def test_sparse_failure_does_not_silently_fall_back_to_dense():
    class BrokenSparse(_FakeSparse):
        def search(self, query, top_k, where=None):
            raise OSError("sqlite unavailable")

    retriever = VectorRetriever(_FakeStore([_chunk("c1", "dense")]), BrokenSparse([]))
    with pytest.raises(RetrievalError, match="No dense-only answer"):
        retriever.retrieve("x")


def test_document_service_deletes_from_chroma_and_bm25(tmp_path):
    persist = tmp_path / "store"
    store = ChromaVectorStore(
        persist_directory=str(persist),
        collection_name="deletion-test",
        embedding_service=EmbeddingService(provider="mock"),
    )
    store.add_chunks([
        Chunk("quasar telemetry", {"document_id": "doc-x", "source": "x.pdf", "chunk_index": 0}, "x-0")
    ])
    sparse = BM25Index(persist / "bm25.sqlite3")
    sparse.upsert_chunks([Chunk("quasar telemetry", {"document_id": "doc-x", "source": "x.pdf"}, "x-0")])
    sparse.set_status("doc-x", "hash", "x.pdf", "completed")

    assert DocumentService(store, sparse).delete_document("doc-x")
    assert store.get_count() == sparse.count() == 0
    assert sparse.get_status("doc-x") is None

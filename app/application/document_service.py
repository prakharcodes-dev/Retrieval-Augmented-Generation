"""Document lifecycle operations that keep Chroma and the BM25 index aligned."""

from pathlib import Path
from typing import Any, Dict, List

from app.config.settings import settings
from app.core.exceptions import VectorStoreError
from app.retrieval.bm25_index import BM25Index
from app.vectorstore.base import BaseVectorStore


class DocumentService:
    def __init__(self, vector_store: BaseVectorStore, sparse_index: BM25Index | None = None):
        self.vector_store = vector_store
        path = Path(getattr(vector_store, "persist_directory", settings.CHROMA_PERSIST_DIR)) / "bm25.sqlite3"
        self.sparse_index = sparse_index or BM25Index(path)
        try:
            self.sparse_index.sync_from_store(self.vector_store)
        except Exception as exc:
            raise VectorStoreError("Cannot manage documents because the BM25 index could not synchronize with Chroma.") from exc

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.sparse_index.list_documents()

    def delete_document(self, document_id: str) -> bool:
        """Delete one document from both stores; never clears the collection."""
        if not document_id:
            raise ValueError("A document ID is required.")
        self.vector_store.delete_document(document_id=document_id)
        try:
            self.sparse_index.forget_document(document_id)
        except Exception as exc:
            # Chroma is authoritative. Reconcile the sparse index after a
            # partial dual-store operation and report if repair also fails.
            try:
                self.sparse_index.sync_from_store(self.vector_store)
                self.sparse_index.remove_document_status(document_id)
            except Exception as repair_exc:
                raise VectorStoreError(
                    "Document was deleted from Chroma but BM25 synchronization failed; retrieval is blocked until repaired."
                ) from repair_exc
            raise VectorStoreError("Document was deleted, but BM25 metadata cleanup failed.") from exc
        return True

"""Dense and BM25 retrieval combined with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import settings
from app.core.exceptions import RetrievalError
from app.retrieval.bm25_index import BM25Index
from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class BaseRetriever:
    """Small retrieval contract shared by the application entry points."""

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """Retrieve independent dense and sparse candidates, then rank with RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        sparse_index: Optional[BM25Index] = None,
        dense_candidate_k: Optional[int] = None,
        sparse_candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ):
        self.vector_store = vector_store
        database_path = Path(getattr(vector_store, "persist_directory", settings.CHROMA_PERSIST_DIR)) / "bm25.sqlite3"
        self.sparse_index = sparse_index or BM25Index(database_path)
        self.dense_candidate_k = max(1, int(dense_candidate_k or settings.DENSE_CANDIDATE_K))
        self.sparse_candidate_k = max(1, int(sparse_candidate_k or settings.SPARSE_CANDIDATE_K))
        self.rrf_k = max(1, int(rrf_k or settings.RRF_K))
        if str(settings.RETRIEVAL_FUSION).lower() != "rrf":
            raise RetrievalError(
                f"Unsupported retrieval fusion '{settings.RETRIEVAL_FUSION}'. Configure retrieval.fusion: rrf."
            )
        try:
            # One-time app/service initialization reconciliation supports old
            # Chroma indexes and repairs a prior cross-index interrupted write.
            self.sparse_index.sync_from_store(self.vector_store)
        except Exception as exc:
            logger.exception("Could not synchronize BM25 index with Chroma.")
            raise RetrievalError(
                "Sparse BM25 index is unavailable or out of sync with Chroma. Retrieval was stopped; "
                "repair/rebuild the sparse index before retrying."
            ) from exc

    @staticmethod
    def _identity(item: Dict[str, Any]) -> Optional[str]:
        metadata = item.get("metadata") or {}
        chunk_id = item.get("chunk_id") or metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        document_id = item.get("document_id") or metadata.get("document_id")
        chunk_index = metadata.get("chunk_index")
        if document_id is not None and chunk_index is not None:
            return f"{document_id}:chunk:{chunk_index}"
        return None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        k = max(1, int(top_k if top_k is not None else settings.TOP_K))
        if self.vector_store.get_count() == 0:
            return []

        try:
            dense = self.vector_store.query(
                query_text=query,
                top_k=min(self.vector_store.get_count(), max(k, self.dense_candidate_k)),
                where=where,
            )
            sparse = self.sparse_index.search(query, top_k=self.sparse_candidate_k, where=where)
        except Exception as exc:
            logger.exception("Hybrid retrieval failed.")
            raise RetrievalError(
                "Hybrid retrieval failed because dense or BM25 search could not complete. "
                "No dense-only answer was generated."
            ) from exc

        fused: Dict[str, Dict[str, Any]] = {}
        for results in (dense, sparse):
            for rank, item in enumerate(results, start=1):
                identity = self._identity(item)
                if identity is None:
                    logger.warning("Skipping retrieval result without a stable chunk identity.")
                    continue
                result = fused.setdefault(identity, dict(item))
                result["hybrid_score"] = result.get("hybrid_score", 0.0) + 1.0 / (self.rrf_k + rank)
                result.setdefault("chunk_id", identity)
                result["metadata"] = result.get("metadata") or item.get("metadata") or {}
                if not result.get("text") and item.get("text"):
                    result["text"] = item["text"]
                result.setdefault("source", item.get("source", result["metadata"].get("source", "")))
                result.setdefault("page_number", item.get("page_number", result["metadata"].get("page_number")))

        ranked = sorted(fused.values(), key=lambda item: item["hybrid_score"], reverse=True)
        return ranked[:k]


# Backwards compatibility alias
DenseRetriever = VectorRetriever

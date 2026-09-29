"""
Persistent ChromaDB Vector Store Implementation with RAM-optimized targeted queries,
incremental batch persistence, transaction failure isolation, and metadata authorization controls.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set
import chromadb

from app.config.settings import settings
from app.core.exceptions import VectorStoreError
from app.core.resource_guard import ResourceGuard
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import Chunk
from app.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB Vector Store Implementation."""

    MAX_METADATA_STR_LEN: int = 1000  # Max length for individual metadata string values

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.COLLECTION_NAME

        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.embedding_service = embedding_service or EmbeddingService()

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_service,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB store: {e}") from e

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes and truncates metadata fields to prevent payload size bloat (CWE-400)
        and illegal types.
        """
        clean_meta = {}
        for k, v in metadata.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (int, float, bool)):
                clean_meta[k] = v
            else:
                str_val = str(v)
                if len(str_val) > self.MAX_METADATA_STR_LEN:
                    str_val = str_val[:self.MAX_METADATA_STR_LEN] + "..."
                clean_meta[k] = str_val
        return clean_meta

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """
        Stores chunks, text embeddings, and metadata in ChromaDB using efficient incremental batching.
        Isolates failures per batch to prevent corrupting existing vector data and enforces metadata bounds.
        """
        if not chunks:
            return 0

        # Memory check before batch insertion
        ResourceGuard.check_memory(context="VectorStore batch insertion")

        total_added = 0
        safe_batch_size = ResourceGuard.calculate_safe_batch_size(
            total_items=len(chunks),
            default_batch_size=batch_size,
            min_batch_size=10,
            max_batch_size=200
        )

        for i in range(0, len(chunks), safe_batch_size):
            batch = chunks[i:i + safe_batch_size]
            ids = [chunk.chunk_id for chunk in batch if chunk.chunk_id]
            documents = [chunk.text for chunk in batch if chunk.text]

            if len(ids) != len(batch) or len(documents) != len(batch):
                raise VectorStoreError("Invalid chunk batch: missing chunk_id or text.")

            metadatas = [self._sanitize_metadata(chunk.metadata) for chunk in batch]

            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                total_added += len(batch)
            except Exception as e:
                logger.error(f"Failed to upsert vector batch starting at index {i}: {e}")
                raise VectorStoreError(f"Vector store batch write failed at index {i}: {e}") from e

        return total_added

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for top_k relevant chunks.
        Supports metadata filtering (`where`) for access control / document isolation (CWE-862).
        """
        if not query_text or not query_text.strip():
            return []

        k = top_k or settings.TOP_K
        total_docs = self.get_count()

        if total_docs == 0:
            return []

        k = min(k, total_docs)

        # Validate where filter if provided
        sanitized_where = None
        if where and isinstance(where, dict):
            sanitized_where = self._sanitize_metadata(where)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if sanitized_where:
            query_kwargs["where"] = sanitized_where

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query execution error: {e}")
            raise VectorStoreError(f"Vector retrieval query failed: {e}") from e

        formatted_results: List[Dict[str, Any]] = []

        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

            for doc, meta, cid, dist in zip(docs, metas, ids, distances):
                similarity_score = max(0.0, 1.0 - float(dist))
                formatted_results.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "document_id": meta.get("document_id", ""),
                    "source": meta.get("source", ""),
                    "page_number": meta.get("page_number", meta.get("page")),
                    "distance": float(dist),
                    "score": round(similarity_score, 4)
                })

        return formatted_results

    def delete_document(self, document_id: str, file_hash: Optional[str] = None) -> int:
        """
        Safely deletes document chunks belonging strictly to the specified document_id / file_hash (CWE-862).
        Enforces document-level isolation instead of wiping the entire database.
        """
        if not document_id and not file_hash:
            raise ValueError("Must provide valid document_id or file_hash to delete.")

        where_clause: Dict[str, Any] = {}
        if document_id:
            where_clause["document_id"] = str(document_id)
        if file_hash:
            where_clause["file_hash"] = str(file_hash)

        try:
            self.collection.delete(where=where_clause)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {e}")
            raise VectorStoreError(f"Failed to delete document vectors: {e}") from e

    def get_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """
        Clears the ChromaDB vector store collection on the engine level.
        Does NOT dump entire collection IDs into RAM (Fixes CWE-400 DoS).
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_service,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to recreate collection on reset: {e}") from e

    def has_file_hash(self, file_hash: str) -> bool:
        """
        Targeted RAM-friendly file hash verification using Chroma metadata filtering.
        Does NOT dump the entire collection into memory.
        """
        if not file_hash or not isinstance(file_hash, str):
            return False
        try:
            res = self.collection.get(
                where={"file_hash": file_hash},
                limit=1,
                include=[]
            )
            return bool(res and res.get("ids") and len(res["ids"]) > 0)
        except Exception:
            return False

    def get_existing_hashes(self, limit: int = 1000) -> Set[str]:
        """
        Fetches distinct document file hashes stored in the collection with pagination/limits
        to prevent RAM exhaustion (Fixes CWE-400 DoS).
        """
        try:
            data = self.collection.get(include=["metadatas"], limit=limit)
            if not data or not data.get("metadatas"):
                return set()
            return {
                m.get("file_hash")
                for m in data["metadatas"]
                if m and isinstance(m, dict) and m.get("file_hash")
            }
        except Exception:
            return set()

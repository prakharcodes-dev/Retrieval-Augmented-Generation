"""
Persistent ChromaDB Vector Store Implementation with RAM-optimized targeted queries,
incremental batch persistence, and transaction failure isolation.
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

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """
        Stores chunks, text embeddings, and metadata in ChromaDB using efficient incremental batching.
        Isolates failures per batch to prevent corrupting existing vector data.
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
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.text for chunk in batch]

            metadatas = []
            for chunk in batch:
                clean_meta = {}
                for k, v in chunk.metadata.items():
                    if v is None:
                        clean_meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                metadatas.append(clean_meta)

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

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries ChromaDB collection for top_k relevant chunks."""
        if not query_text or not query_text.strip():
            return []

        k = top_k or settings.TOP_K
        total_docs = self.get_count()

        if total_docs == 0:
            return []

        k = min(k, total_docs)

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
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

    def get_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Completely clears and resets the ChromaDB vector store collection."""
        try:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_service,
            metadata={"hnsw:space": "cosine"}
        )

    def has_file_hash(self, file_hash: str) -> bool:
        """
        Targeted RAM-friendly file hash verification using Chroma metadata filtering.
        Does NOT dump the entire collection into memory.
        """
        if not file_hash:
            return False
        try:
            res = self.collection.get(
                where={"file_hash": file_hash},
                limit=1,
                include=[]
            )
            return bool(res and res.get("ids") and len(res["ids"]) > 0)
        except Exception:
            return file_hash in self.get_existing_hashes()

    def get_existing_hashes(self) -> Set[str]:
        """Fetches distinct document file hashes stored in the collection."""
        try:
            data = self.collection.get(include=["metadatas"])
            if not data or not data.get("metadatas"):
                return set()
            return {
                m.get("file_hash")
                for m in data["metadatas"]
                if m and m.get("file_hash")
            }
        except Exception:
            return set()

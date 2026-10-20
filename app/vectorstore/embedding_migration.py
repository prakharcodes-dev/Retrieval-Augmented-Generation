"""Explicit, non-destructive Chroma embedding migration with a retained backup."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Optional

from app.core.exceptions import VectorStoreError
from app.ingestion.chunker import TextChunker


def migrate_collection_embeddings(
    persist_directory: str,
    collection_name: str,
    embedding_service: Any,
    batch_size: int = 16,
    token_budget_per_minute: int = 18_000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Re-embed a collection into a temporary collection, then swap names.

    The original collection is retained as ``<name>_pre_migration``. This
    function never deletes the original data and aborts before swapping if any
    batch or dimension check fails.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_directory)
    old = client.get_collection(name=collection_name)
    original_count = old.count()
    backup_name = f"{collection_name}_pre_migration"
    staging_name = f"{collection_name}_migration_staging"
    existing = {item.name for item in client.list_collections()}
    if backup_name in existing:
        raise VectorStoreError(
            f"Migration stopped because backup collection '{backup_name}' already exists. "
            "Inspect the collections before trying another migration; nothing was overwritten."
        )

    if staging_name in existing:
        # A prior attempt may have stopped part-way through a provider outage.
        # Reuse it and upsert the source records by ID; do not discard partial work.
        staging = client.get_collection(
            name=staging_name,
            embedding_function=embedding_service,
        )
        if staging.count() and getattr(embedding_service, "embedding_dimension", None):
            existing_vectors = staging.get(limit=1, include=["embeddings"]).get("embeddings")
            if existing_vectors is not None and len(existing_vectors):
                if len(existing_vectors[0]) != embedding_service.embedding_dimension:
                    raise VectorStoreError(
                        "The existing migration staging collection has a different embedding dimension. "
                        "It was left untouched; inspect it before retrying."
                    )
    else:
        staging = client.create_collection(
            name=staging_name,
            embedding_function=embedding_service,
            metadata=old.metadata or {"hnsw:space": "cosine"},
        )
    offset = 0
    token_counter = TextChunker()
    recent_usage = deque()
    try:
        while offset < original_count:
            batch = old.get(
                limit=min(batch_size, original_count - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids", [])
            if not ids:
                break

            documents = batch.get("documents") or []
            batch_tokens = sum(token_counter.count_tokens(text or "") for text in documents)
            if batch_tokens > token_budget_per_minute:
                raise VectorStoreError(
                    f"One migration batch contains {batch_tokens} input tokens, above the safe "
                    f"per-minute budget of {token_budget_per_minute}. Reduce batch_size before retrying."
                )

            # Reserve headroom below Gemini's configured TPM limit. Count
            # accepted batches over a rolling 60-second window and pause before
            # sending a batch that would exceed this migration's conservative budget.
            while True:
                now = time.monotonic()
                while recent_usage and now - recent_usage[0][0] >= 60:
                    recent_usage.popleft()
                used_tokens = sum(item[1] for item in recent_usage)
                if used_tokens + batch_tokens <= token_budget_per_minute:
                    break
                wait_seconds = max(0.1, 60 - (now - recent_usage[0][0]) + 0.1)
                time.sleep(wait_seconds)

            staging.upsert(
                ids=ids,
                documents=documents,
                metadatas=batch.get("metadatas"),
            )
            offset += len(ids)
            recent_usage.append((time.monotonic(), batch_tokens))
            if progress_callback:
                progress_callback(offset, original_count)

        if staging.count() != original_count:
            raise VectorStoreError(
                f"Migration copied {staging.count()} of {original_count} records; original remains intact."
            )
        if original_count:
            vectors = staging.get(limit=1, include=["embeddings"]).get("embeddings")
            dimension = getattr(embedding_service, "embedding_dimension", None)
            if vectors is not None and len(vectors) and dimension and len(vectors[0]) != dimension:
                raise VectorStoreError("New embedding dimension validation failed; original remains intact.")

        old.modify(name=backup_name)
        try:
            staging.modify(name=collection_name)
        except Exception:
            old.modify(name=collection_name)
            raise
        return backup_name
    except Exception:
        # Keep staging for diagnosis if it contains data; never delete user data.
        raise

"""Explicitly re-embed the configured Chroma collection and retain a backup."""
from app.config.settings import settings
from app.core.exceptions import EmbeddingError
from app.core.exceptions import VectorStoreError
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.embedding_migration import migrate_collection_embeddings


if __name__ == "__main__":
    try:
        service = EmbeddingService()

        def report_progress(done: int, total: int) -> None:
            print(f"Re-embedded {done}/{total} chunks.")

        backup = migrate_collection_embeddings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
            embedding_service=service,
            progress_callback=report_progress,
        )
        print(f"Migration complete. Original index retained as collection: {backup}")
    except (EmbeddingError, VectorStoreError) as exc:
        raise SystemExit(
            f"Migration stopped: {exc} The original collection was retained. "
            "Fix the issue above, then rerun this command to resume staging."
        ) from None

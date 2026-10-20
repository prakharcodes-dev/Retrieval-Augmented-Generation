import pytest

from app.core.exceptions import EmbeddingError
from app.embeddings.embedding_service import EmbeddingService
from app.config.settings import settings
from app.vectorstore.embedding_migration import migrate_collection_embeddings


def test_missing_embedding_credentials_has_actionable_error(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        EmbeddingService(provider="auto")


def test_embedding_providers_have_explicit_dimensions(monkeypatch):
    service = EmbeddingService(provider="gemini", api_key="test-gemini-key")
    assert service.embedding_dimension == 768

    mock = EmbeddingService(provider="mock")
    assert mock.embedding_dimension == 384


def test_embedding_migration_retains_original_collection(tmp_path):
    import chromadb

    db_path = str(tmp_path / "migration_db")
    client = chromadb.PersistentClient(path=db_path)
    original = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})
    original.upsert(
        ids=["chunk-1", "chunk-2"],
        documents=["first source text", "second source text"],
        metadatas=[{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 2}],
        embeddings=[[0.1] * 384, [0.2] * 384],
    )

    backup_name = migrate_collection_embeddings(
        persist_directory=db_path,
        collection_name="docs",
        embedding_service=EmbeddingService(provider="mock"),
        batch_size=1,
    )

    assert backup_name == "docs_pre_migration"
    assert client.get_collection(name=backup_name).count() == 2
    migrated = client.get_collection(name="docs")
    assert migrated.count() == 2
    copied = migrated.get(include=["documents", "metadatas"])
    assert set(copied["documents"]) == {"first source text", "second source text"}
    assert {meta["source"] for meta in copied["metadatas"]} == {"a.pdf", "b.pdf"}

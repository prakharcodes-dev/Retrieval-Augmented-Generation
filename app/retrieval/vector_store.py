"""
Backwards-compatibility module re-exporting ChromaVectorStore and CustomEmbeddingFunction.
"""
from app.embeddings.embedding_service import EmbeddingService as CustomEmbeddingFunction
from app.vectorstore.chroma_store import ChromaVectorStore as VectorStore

__all__ = ["VectorStore", "CustomEmbeddingFunction"]

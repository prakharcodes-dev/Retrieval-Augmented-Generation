"""
Backwards-compatibility module re-exporting VectorRetriever and BaseRetriever.
"""
from app.retrieval.vector_retriever import BaseRetriever, VectorRetriever, DenseRetriever

__all__ = ["BaseRetriever", "VectorRetriever", "DenseRetriever"]

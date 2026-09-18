"""
Core application module containing exception hierarchy, state tracking, and resource guard.
"""

from app.core.exceptions import (
    RAGException,
    ResourceUnsafeError,
    PDFValidationError,
    PDFExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    GenerationError,
)
from app.core.state import OperationState, StateTracker
from app.core.resource_guard import ResourceGuard

__all__ = [
    "RAGException",
    "ResourceUnsafeError",
    "PDFValidationError",
    "PDFExtractionError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "GenerationError",
    "OperationState",
    "StateTracker",
    "ResourceGuard",
]

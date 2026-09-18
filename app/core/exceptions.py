"""
Isolated exception hierarchy for the RAG Application.
Ensures distinct subsystem failure modes without killing the application process.
"""

class RAGException(Exception):
    """Base exception class for all RAG application errors."""
    pass


class ResourceUnsafeError(RAGException):
    """Raised when system memory (RAM/VRAM) or workload exceeds safe operational limits."""
    pass


class PDFValidationError(ValueError, RAGException):
    """Raised when PDF file is invalid, encrypted, corrupt, or unreadable."""
    pass


class PDFExtractionError(RAGException):
    """Raised when text extraction from a PDF fails."""
    pass


class ChunkingError(ValueError, RAGException):
    """Raised when document chunking fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when vector embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector database operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when vector retrieval fails."""
    pass


class GenerationError(RAGException):
    """Raised when LLM answer generation fails."""
    pass

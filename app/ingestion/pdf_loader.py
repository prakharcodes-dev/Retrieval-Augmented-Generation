"""
Backwards-compatibility module forwarding DocumentLoader and DocumentPage.
"""
from app.ingestion.loader import DocumentLoader, PDFLoader, DocumentPage

__all__ = ["DocumentLoader", "PDFLoader", "DocumentPage"]

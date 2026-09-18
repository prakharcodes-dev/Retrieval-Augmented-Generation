"""
Document Loader module with safe PDF handle lifecycle management and streaming page batching.
Supports PyMuPDF (fitz) and PyPDF fallback with deterministic open/close context cleanup.
"""

import hashlib
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple

from app.config.settings import settings
from app.core.exceptions import PDFValidationError, PDFExtractionError
from app.core.resource_guard import ResourceGuard


@dataclass
class DocumentPage:
    """Represents text extracted from a single document page/section with metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@contextlib.contextmanager
def safe_pymupdf_open(file_path: Path):
    """
    Context manager that guarantees PyMuPDF document handle is safely opened and closed.
    Prevents 'document closed' errors and leaked file descriptors.
    """
    import pymupdf
    doc = None
    try:
        doc = pymupdf.open(str(file_path))
        yield doc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


@contextlib.contextmanager
def safe_pypdf_open(file_path: Path):
    """
    Context manager that guarantees PyPDF file handle is safely opened and closed.
    """
    from pypdf import PdfReader
    f = None
    try:
        f = open(str(file_path), "rb")
        reader = PdfReader(f)
        yield reader
    finally:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


class DocumentLoader:
    """
    Loads PDF and TXT documents page-by-page with streaming page batching,
    safe PDF lifecycle management, and resource checks.
    """

    def compute_file_hash(self, path: Path) -> str:
        """Computes SHA-256 hash of a file for exact deduplication."""
        hasher = hashlib.sha256()
        with open(str(path), "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_file(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Validates file existence, size, readability, and PDF structure
        without reading all text content into memory.
        Returns metadata dict containing total page count and document ID.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a valid file: {path}")

        file_bytes = path.stat().st_size
        ResourceGuard.check_pdf_workload(file_bytes, filename=path.name)

        ext = path.suffix.lower()
        if ext not in (".pdf", ".txt", ".md", ".text"):
            raise ValueError(f"Unsupported file type '{ext}'. Supported formats: .pdf, .txt, .md")

        file_hash = self.compute_file_hash(path)
        doc_id = f"doc_{file_hash[:12]}"
        total_pages = 1

        if ext == ".pdf":
            total_pages = self._validate_pdf(path)

        return {
            "path": path,
            "filename": path.name,
            "file_type": ext.lstrip("."),
            "file_hash": file_hash,
            "document_id": doc_id,
            "file_size_bytes": file_bytes,
            "total_pages": total_pages,
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
        }

    def _validate_pdf(self, path: Path) -> int:
        """Performs lightweight PDF structural validation."""
        doc_name = path.name

        # Try PyMuPDF first
        try:
            with safe_pymupdf_open(path) as doc:
                if getattr(doc, "is_encrypted", False):
                    raise PDFValidationError(f"PDF file '{doc_name}' is password-protected and cannot be read.")
                total_pages = len(doc)
                if total_pages == 0:
                    raise PDFValidationError(f"PDF file '{doc_name}' contains no pages.")
                return total_pages
        except PDFValidationError:
            raise
        except Exception:
            pass

        # PyPDF fallback
        try:
            with safe_pypdf_open(path) as reader:
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        raise PDFValidationError(f"PDF file '{doc_name}' is password-protected and cannot be read.")
                total_pages = len(reader.pages)
                if total_pages == 0:
                    raise PDFValidationError(f"PDF file '{doc_name}' contains no pages.")
                return total_pages
        except PDFValidationError:
            raise
        except Exception as e:
            if "encrypted" in str(e).lower() or "password" in str(e).lower():
                raise PDFValidationError(f"PDF file '{doc_name}' is password-protected.") from e
            raise PDFValidationError(f"Failed to validate PDF file {doc_name}: {str(e)}") from e

    def stream_pdf_batches(
        self,
        file_path: str | Path,
        batch_size: int = 20
    ) -> Generator[List[DocumentPage], None, None]:
        """
        Yields batches of DocumentPage objects for a PDF file using streaming handle management.
        Guarantees that PyMuPDF/PyPDF handles are closed after each batch slice.
        """
        path = Path(file_path)
        val_info = self.validate_file(path)
        total_pages = val_info["total_pages"]
        doc_id = val_info["document_id"]
        doc_name = val_info["filename"]
        file_hash = val_info["file_hash"]
        title = val_info["title"]
        timestamp = datetime.now(timezone.utc).isoformat()

        # Page batch loop
        for start_idx in range(0, total_pages, batch_size):
            end_idx = min(start_idx + batch_size, total_pages)
            batch_pages: List[DocumentPage] = []

            # Check memory before reading batch
            ResourceGuard.check_memory(context=f"PDF extraction pages {start_idx + 1}-{end_idx}")

            # Attempt batch extraction using PyMuPDF first
            extracted_by_mupdf = False
            try:
                with safe_pymupdf_open(path) as doc:
                    for i in range(start_idx, end_idx):
                        try:
                            page = doc[i]
                            page_text = page.get_text() or ""
                        except Exception as p_err:
                            page_text = ""
                        
                        metadata = {
                            "document_id": doc_id,
                            "filename": doc_name,
                            "file_type": "pdf",
                            "file_hash": file_hash,
                            "source": doc_name,
                            "title": title,
                            "page_number": i + 1,
                            "page": i + 1,
                            "ingestion_timestamp": timestamp,
                            "file_path": str(path.resolve())
                        }
                        batch_pages.append(DocumentPage(text=page_text, metadata=metadata))
                extracted_by_mupdf = True
            except Exception:
                batch_pages.clear()

            # PyPDF fallback for batch
            if not extracted_by_mupdf:
                try:
                    with safe_pypdf_open(path) as reader:
                        for i in range(start_idx, end_idx):
                            try:
                                page = reader.pages[i]
                                page_text = page.extract_text() or ""
                            except Exception:
                                page_text = ""
                            metadata = {
                                "document_id": doc_id,
                                "filename": doc_name,
                                "file_type": "pdf",
                                "file_hash": file_hash,
                                "source": doc_name,
                                "title": title,
                                "page_number": i + 1,
                                "page": i + 1,
                                "ingestion_timestamp": timestamp,
                                "file_path": str(path.resolve())
                            }
                            batch_pages.append(DocumentPage(text=page_text, metadata=metadata))
                except Exception as e:
                    raise PDFExtractionError(f"Failed to extract text from {doc_name} pages {start_idx + 1}-{end_idx}: {e}") from e

            yield batch_pages

    def load_file(self, file_path: str | Path) -> List[DocumentPage]:
        """
        Loads a PDF or TXT document and returns DocumentPage objects.
        Uses memory-safe batching internally.
        """
        path = Path(file_path)
        val_info = self.validate_file(path)
        ext = path.suffix.lower()

        if ext in (".txt", ".md", ".text"):
            return self._load_txt(path, val_info)

        pages: List[DocumentPage] = []
        for batch in self.stream_pdf_batches(path, batch_size=50):
            pages.extend(batch)
        return pages

    def _load_txt(self, path: Path, val_info: Dict[str, Any]) -> List[DocumentPage]:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise PDFExtractionError(f"Failed to read text file {path.name}: {str(e)}") from e

        metadata = {
            "document_id": val_info["document_id"],
            "filename": val_info["filename"],
            "file_type": val_info["file_type"],
            "file_hash": val_info["file_hash"],
            "source": val_info["filename"],
            "title": val_info["title"],
            "page_number": None,
            "page": None,
            "ingestion_timestamp": timestamp,
            "file_path": str(path.resolve())
        }
        return [DocumentPage(text=content, metadata=metadata)]

    def load_directory(self, dir_path: str | Path) -> List[DocumentPage]:
        """Loads all supported documents in a directory."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        all_pages: List[DocumentPage] = []
        files = [p for p in path.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md", ".text")]

        for f in files:
            try:
                pages = self.load_file(f)
                all_pages.extend(pages)
            except Exception as e:
                print(f"Warning: Skipping file {f} due to error: {e}")

        return all_pages


class PDFLoader(DocumentLoader):
    """Backwards compatibility alias."""
    pass

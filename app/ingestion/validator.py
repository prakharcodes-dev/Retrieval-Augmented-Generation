"""
Production-grade document upload validation, filename sanitization, and corruption detection.
Prevents malformed, empty, encrypted, oversized, and unreadable files from entering the pipeline.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from app.config.settings import settings
from app.core.exceptions import PDFValidationError
from app.core.resource_guard import ResourceGuard


_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.\- ]+")
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}


@dataclass
class ValidationResult:
    """Detailed outcome of a document upload validation check."""

    is_valid: bool
    sanitized_filename: str
    file_extension: str
    file_size_bytes: int
    file_hash: str
    total_pages: int = 1
    is_duplicate: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


class DocumentValidator:
    """Validates files prior to temporary staging and vector ingestion."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes raw uploaded filename to prevent directory traversal,
        reserved OS names, and injection of control/special characters.
        """
        if not filename:
            return "unnamed_document.pdf"

        # 1. Strip directory traversal tokens and paths
        clean_name = Path(filename).name.strip()
        # 2. Remove null bytes and control chars
        clean_name = clean_name.replace("\x00", "").strip()
        # 3. Clean dangerous characters while preserving extension
        clean_name = _SAFE_FILENAME_RE.sub("_", clean_name)
        # 4. Collapse consecutive underscores and dots
        clean_name = re.sub(r"_{2,}", "_", clean_name)
        clean_name = re.sub(r"\.{2,}", ".", clean_name)

        stem = Path(clean_name).stem.upper()
        ext = Path(clean_name).suffix.lower()

        if stem in _RESERVED_NAMES or not stem:
            clean_name = f"doc_{hashlib.md5(filename.encode()).hexdigest()[:8]}{ext or '.pdf'}"

        return clean_name

    @staticmethod
    def compute_bytes_hash(file_bytes: bytes) -> str:
        """Computes SHA-256 hash of byte buffer."""
        return hashlib.sha256(file_bytes).hexdigest()

    @classmethod
    def validate_upload(
        cls,
        file_bytes: bytes,
        filename: str,
        vector_store: Optional[Any] = None,
        max_size_mb: Optional[int] = None,
    ) -> ValidationResult:
        """
        Executes comprehensive validation on an uploaded document buffer.
        Checks: extension, empty size, size limit, PDF header magic bytes,
        structural integrity, encryption, extractable text, and duplicate hash.
        """
        sanitized_name = cls.sanitize_filename(filename)
        ext = Path(sanitized_name).suffix.lower()
        size_bytes = len(file_bytes) if file_bytes else 0
        limit_mb = max_size_mb or getattr(settings, "MAX_FILE_SIZE_MB", 25)
        max_bytes = limit_mb * 1024 * 1024

        # 1. Extension validation
        if ext not in _SUPPORTED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_extension=ext,
                file_size_bytes=size_bytes,
                file_hash="",
                error_message=(
                    f"Unsupported file format '{ext or 'unknown'}'. "
                    f"Supported formats: PDF (.pdf), Text (.txt, .md)."
                ),
            )

        # 2. Empty file validation
        if size_bytes == 0:
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_extension=ext,
                file_size_bytes=0,
                file_hash="",
                error_message="Uploaded file is empty (0 bytes). Please upload a valid document with content.",
            )

        # 3. Size limit validation
        if size_bytes > max_bytes:
            size_mb = size_bytes / (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_extension=ext,
                file_size_bytes=size_bytes,
                file_hash="",
                error_message=(
                    f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb} MB. "
                    "Please upload a smaller document."
                ),
            )

        # Check memory guard workload
        try:
            ResourceGuard.check_pdf_workload(size_bytes, filename=sanitized_name)
        except Exception as rg_err:
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_extension=ext,
                file_size_bytes=size_bytes,
                file_hash="",
                error_message=f"System resource safety check rejected document: {rg_err}",
            )

        file_hash = cls.compute_bytes_hash(file_bytes)

        # 4. Duplicate document check
        is_duplicate = False
        if vector_store is not None:
            try:
                if hasattr(vector_store, "has_file_hash") and vector_store.has_file_hash(file_hash):
                    is_duplicate = True
            except Exception:
                pass

        # 5. Format-specific deep content validation
        total_pages = 1
        warnings: List[str] = []

        if ext == ".pdf":
            # 5a. Magic bytes check
            # Standard PDF files must start with '%PDF-' within the first 1024 bytes
            header_sample = file_bytes[:1024]
            if b"%PDF-" not in header_sample:
                return ValidationResult(
                    is_valid=False,
                    sanitized_filename=sanitized_name,
                    file_extension=ext,
                    file_size_bytes=size_bytes,
                    file_hash=file_hash,
                    error_message=(
                        "Invalid PDF structure: File does not start with standard '%PDF-' header. "
                        "The document may be corrupted or renamed from another format."
                    ),
                )

            # 5b. Structural validation via PyMuPDF / PyPDF
            pdf_valid, total_pages, err_msg, warn_list = cls._deep_validate_pdf_bytes(
                file_bytes, sanitized_name
            )
            warnings.extend(warn_list)
            if not pdf_valid:
                return ValidationResult(
                    is_valid=False,
                    sanitized_filename=sanitized_name,
                    file_extension=ext,
                    file_size_bytes=size_bytes,
                    file_hash=file_hash,
                    total_pages=0,
                    error_message=err_msg,
                    warnings=warnings,
                )

        elif ext in (".txt", ".md", ".text"):
            # Text encoding & empty text check
            try:
                text_content = file_bytes.decode("utf-8", errors="replace").strip()
                if not text_content:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_filename=sanitized_name,
                        file_extension=ext,
                        file_size_bytes=size_bytes,
                        file_hash=file_hash,
                        error_message="Text document contains only whitespace and no readable characters.",
                    )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    sanitized_filename=sanitized_name,
                    file_extension=ext,
                    file_size_bytes=size_bytes,
                    file_hash=file_hash,
                    error_message=f"Failed to read text document: {e}",
                )

        return ValidationResult(
            is_valid=True,
            sanitized_filename=sanitized_name,
            file_extension=ext,
            file_size_bytes=size_bytes,
            file_hash=file_hash,
            total_pages=total_pages,
            is_duplicate=is_duplicate,
            warnings=warnings,
        )

    @classmethod
    def _deep_validate_pdf_bytes(
        cls, file_bytes: bytes, filename: str
    ) -> Tuple[bool, int, Optional[str], List[str]]:
        """
        Validates internal PDF structures, encryption status, and text extractability.
        Returns (is_valid, total_pages, error_message, warnings).
        """
        warnings: List[str] = []

        # Try PyMuPDF first
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            try:
                if getattr(doc, "is_encrypted", False):
                    return False, 0, f"PDF file '{filename}' is password-protected and cannot be indexed.", []

                pages = len(doc)
                if pages == 0:
                    return False, 0, f"PDF file '{filename}' contains 0 pages.", []

                # Sample pages for extractable text
                extracted_chars = 0
                sample_limit = min(pages, 10)
                for i in range(sample_limit):
                    page_text = doc[i].get_text() or ""
                    extracted_chars += len(page_text.strip())

                if extracted_chars == 0 and pages > 0:
                    warnings.append(
                        "Document appears to be a scanned image or contains no extractable digital text. "
                        "Text indexing may produce 0 chunks unless optical character recognition (OCR) is applied."
                    )

                return True, pages, None, warnings
            finally:
                doc.close()

        except Exception as mupdf_err:
            mupdf_msg = str(mupdf_err).lower()
            if "password" in mupdf_msg or "encrypted" in mupdf_msg:
                return False, 0, f"PDF file '{filename}' is password-protected.", []

        # PyPDF Fallback
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                return False, 0, f"PDF file '{filename}' is password-protected.", []

            pages = len(reader.pages)
            if pages == 0:
                return False, 0, f"PDF file '{filename}' contains 0 pages.", []

            return True, pages, None, warnings
        except Exception as pypdf_err:
            pypdf_msg = str(pypdf_err).lower()
            if "password" in pypdf_msg or "encrypted" in pypdf_msg:
                return False, 0, f"PDF file '{filename}' is password-protected.", []
            return False, 0, f"PDF file '{filename}' is corrupted or malformed and cannot be parsed ({pypdf_err}).", []

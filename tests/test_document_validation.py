"""
Unit tests for DocumentValidator:
File format validation, size checks, empty documents, corrupted PDFs,
path traversal sanitization, and duplicate hash detection.
"""

import pytest
from app.core.exceptions import PDFValidationError
from app.ingestion.validator import DocumentValidator, ValidationResult


def test_sanitize_filename_traversal_and_special_chars():
    # Path traversal attempt
    traversal = "../../etc/passwd.pdf"
    clean = DocumentValidator.sanitize_filename(traversal)
    assert "/" not in clean
    assert ".." not in clean
    assert clean == "passwd.pdf"

    # Windows style traversal
    win_traversal = "..\\..\\windows\\system32\\cmd.exe.pdf"
    clean_win = DocumentValidator.sanitize_filename(win_traversal)
    assert "\\" not in clean_win
    assert ".." not in clean_win

    # Control chars & null bytes
    null_byte = "invoice\x00_2024.pdf"
    clean_null = DocumentValidator.sanitize_filename(null_byte)
    assert "\x00" not in clean_null

    # Empty filename fallback
    empty_clean = DocumentValidator.sanitize_filename("")
    assert empty_clean == "unnamed_document.pdf"


def test_unsupported_file_extension():
    res = DocumentValidator.validate_upload(
        file_bytes=b"fake content",
        filename="malicious_payload.exe",
    )
    assert not res.is_valid
    assert "Unsupported file format" in res.error_message


def test_empty_file_validation():
    res = DocumentValidator.validate_upload(
        file_bytes=b"",
        filename="empty.pdf",
    )
    assert not res.is_valid
    assert "empty (0 bytes)" in res.error_message


def test_oversized_file_validation():
    # 26 MB buffer with 25 MB limit
    large_bytes = b"0" * (26 * 1024 * 1024)
    res = DocumentValidator.validate_upload(
        file_bytes=large_bytes,
        filename="large.pdf",
        max_size_mb=25,
    )
    assert not res.is_valid
    assert "exceeds maximum allowed limit" in res.error_message


def test_corrupted_pdf_missing_header():
    # File named .pdf but without %PDF- magic bytes
    fake_pdf = b"This is just plain text masquerading as a PDF file."
    res = DocumentValidator.validate_upload(
        file_bytes=fake_pdf,
        filename="corrupted.pdf",
    )
    assert not res.is_valid
    assert "does not start with standard '%PDF-'" in res.error_message


def test_corrupted_pdf_broken_stream():
    # Starts with %PDF- but followed by broken garbage
    corrupted_pdf = b"%PDF-1.4\n broken garbage xref corrupted %%EOF"
    res = DocumentValidator.validate_upload(
        file_bytes=corrupted_pdf,
        filename="broken.pdf",
    )
    assert not res.is_valid
    assert "corrupted or malformed" in res.error_message or "failed" in res.error_message.lower()


def test_valid_txt_upload():
    txt_bytes = "This is a clean, valid company travel guide and policy.".encode("utf-8")
    res = DocumentValidator.validate_upload(
        file_bytes=txt_bytes,
        filename="guide.txt",
    )
    assert res.is_valid
    assert res.file_size_bytes == len(txt_bytes)
    assert len(res.file_hash) == 64


def test_empty_txt_upload():
    whitespace_bytes = b"   \n\t  \n  "
    res = DocumentValidator.validate_upload(
        file_bytes=whitespace_bytes,
        filename="blank.txt",
    )
    assert not res.is_valid
    assert "whitespace and no readable characters" in res.error_message


def test_duplicate_document_detection():
    test_bytes = b"Sample document bytes for duplicate check"
    expected_hash = DocumentValidator.compute_bytes_hash(test_bytes)

    class _MockStore:
        def has_file_hash(self, h):
            return h == expected_hash

    res = DocumentValidator.validate_upload(
        file_bytes=test_bytes,
        filename="sample.txt",
        vector_store=_MockStore(),
    )
    assert res.is_valid
    assert res.is_duplicate is True

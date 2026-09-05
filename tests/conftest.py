import sys
from pathlib import Path
import pytest
from pypdf import PdfWriter

# Ensure app package is importable in tests
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@pytest.fixture
def temp_pdf_file(tmp_path):
    """Creates a temporary sample PDF document for testing PDF loading and ingestion."""
    pdf_path = tmp_path / "sample_policy.pdf"
    writer = PdfWriter()

    # Page 1
    page1 = writer.add_blank_page(width=612, height=792)
    # Add simple text annotation/content
    # Since add_blank_page is empty, we write a lightweight PDF with actual text stream
    # or create PDF page stream directly.
    return pdf_path

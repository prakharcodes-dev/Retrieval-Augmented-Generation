import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


@dataclass
class DocumentPage:
    """Represents text extracted from a single document page/section with metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads PDF and TXT documents, extracting text page-by-page with robust error tolerance."""

    def load_file(self, file_path: str | Path) -> List[DocumentPage]:
        """Loads a single PDF or TXT file and returns DocumentPage objects with attached metadata."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a valid file: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".txt", ".md", ".text"):
            return self._load_txt(path)
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Supported formats: .pdf, .txt, .md")

    def compute_file_hash(self, path: Path) -> str:
        with open(str(path), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _generate_doc_id(self, path: Path) -> str:
        file_hash = self.compute_file_hash(path)
        return f"doc_{file_hash[:12]}"

    def _load_pdf(self, path: Path) -> List[DocumentPage]:
        pages: List[DocumentPage] = []
        file_hash = self.compute_file_hash(path)
        doc_id = f"doc_{file_hash[:12]}"
        doc_name = path.name
        timestamp = datetime.now(timezone.utc).isoformat()
        title = path.stem.replace("_", " ").replace("-", " ").title()

        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            for i, page in enumerate(doc):
                page_text = page.get_text() or ""
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
                pages.append(DocumentPage(text=page_text, metadata=metadata))
            doc.close()
            return pages
        except Exception:
            pass

        try:
            with open(str(path), "rb") as f:
                reader = PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        raise ValueError(f"PDF file '{doc_name}' is password-protected and cannot be read.")

                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as page_err:
                        print(f"Warning: Failed to extract text on page {i + 1} of {doc_name}: {page_err}")
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
                    pages.append(DocumentPage(text=page_text, metadata=metadata))
        except Exception as e:
            if "encrypted" in str(e).lower() or "password" in str(e).lower():
                raise ValueError(f"PDF file '{doc_name}' is password-protected.") from e
            raise RuntimeError(f"Failed to read PDF file {doc_name}: {str(e)}") from e

        return pages


    def _load_txt(self, path: Path) -> List[DocumentPage]:
        file_hash = self.compute_file_hash(path)
        doc_id = f"doc_{file_hash[:12]}"
        doc_name = path.name
        timestamp = datetime.now(timezone.utc).isoformat()
        title = path.stem.replace("_", " ").replace("-", " ").title()

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Failed to read text file {doc_name}: {str(e)}") from e

        metadata = {
            "document_id": doc_id,
            "filename": doc_name,
            "file_type": "txt",
            "file_hash": file_hash,
            "source": doc_name,
            "title": title,
            "page_number": None,
            "page": None,
            "ingestion_timestamp": timestamp,
            "file_path": str(path.resolve())
        }

        return [DocumentPage(text=content, metadata=metadata)]


    def load_directory(self, dir_path: str | Path) -> List[DocumentPage]:
        """Loads all supported PDF and TXT files in a directory recursively."""
        path = Path(dir_path)

        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        all_pages: List[DocumentPage] = []
        files = [p for p in path.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md", ".text")]

        if not files:
            print(f"Warning: No supported documents (.pdf, .txt) found in directory {path}")
            return []

        for file_path in files:
            try:
                pages = self.load_file(file_path)
                all_pages.extend(pages)
            except Exception as e:
                print(f"Warning: Skipping file {file_path} due to error: {e}")

        return all_pages


# Backwards compatibility alias for PDFLoader
class PDFLoader(DocumentLoader):
    pass

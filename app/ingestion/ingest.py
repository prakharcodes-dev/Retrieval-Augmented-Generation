from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from app.config.settings import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker

if TYPE_CHECKING:
    from app.vectorstore.chroma_store import ChromaVectorStore


def ingest_documents(
    path: str | Path,
    vector_store: Optional["ChromaVectorStore"] = None,
    reset: bool = False
) -> Dict[str, Any]:
    """
    High-level Ingestion Pipeline:
    Upload/Path -> Document Loading (.pdf, .txt) -> Text Extraction -> Text Cleaning -> Token Chunking -> ChromaDB Vector Store
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Ingestion path does not exist: {target}")

    loader = DocumentLoader()
    cleaner = TextCleaner()

    if target.is_file():
        pages = loader.load_file(target)
        file_count = 1
    elif target.is_dir():
        pages = loader.load_directory(target)
        files = [p for p in target.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md", ".text")]
        file_count = len(files)
    else:
        raise ValueError(f"Invalid path type for ingestion: {target}")

    if not pages:
        return {
            "status": "warning",
            "message": "No supported document content was found in the specified path.",
            "files_processed": file_count,
            "pages_processed": 0,
            "chunks_stored": 0,
            "total_chunks_in_db": 0
        }

    # Clean text for each page prior to chunking
    for page in pages:
        page.text = cleaner.clean(page.text)

    # Filter out empty pages after cleaning
    valid_pages = [p for p in pages if p.text and p.text.strip()]

    if not valid_pages:
        return {
            "status": "warning",
            "message": "No extractable text was found in the document. The document might be a scanned image, image-only PDF, or empty file.",
            "files_processed": file_count,
            "pages_processed": len(pages),
            "chunks_stored": 0,
            "total_chunks_in_db": vector_store.get_count() if vector_store else 0
        }

    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    chunks = chunker.chunk_documents(valid_pages)

    from app.vectorstore.chroma_store import ChromaVectorStore
    store = vector_store or ChromaVectorStore()
    if reset:
        store.reset()

    added_count = store.add_chunks(chunks)

    return {
        "status": "success",
        "files_processed": file_count,
        "pages_processed": len(valid_pages),
        "chunks_stored": added_count,
        "total_chunks_in_db": store.get_count()
    }

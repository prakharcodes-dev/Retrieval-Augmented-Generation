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
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Ingestion path does not exist: {target}")

    from app.vectorstore.chroma_store import ChromaVectorStore
    store = vector_store or ChromaVectorStore()
    if reset:
        store.reset()

    existing_hashes = store.get_existing_hashes() if not reset else set()

    loader = DocumentLoader()
    cleaner = TextCleaner()

    if target.is_file():
        file_hash = loader.compute_file_hash(target)
        if file_hash in existing_hashes:
            return {
                "status": "skipped",
                "message": f"File '{target.name}' already processed. Reusing existing vector store.",
                "files_processed": 0,
                "pages_processed": 0,
                "chunks_stored": 0,
                "total_chunks_in_db": store.get_count()
            }
        pages = loader.load_file(target)
        file_count = 1
    elif target.is_dir():
        files = [p for p in target.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md", ".text")]
        new_files = [f for f in files if loader.compute_file_hash(f) not in existing_hashes]
        if not new_files:
            return {
                "status": "skipped",
                "message": "All documents in directory already processed. Reusing existing vector store.",
                "files_processed": 0,
                "pages_processed": 0,
                "chunks_stored": 0,
                "total_chunks_in_db": store.get_count()
            }
        pages = []
        for f in new_files:
            pages.extend(loader.load_file(f))
        file_count = len(new_files)
    else:
        raise ValueError(f"Invalid path type for ingestion: {target}")

    if not pages:
        return {
            "status": "warning",
            "message": "No supported document content was found in the specified path.",
            "files_processed": file_count,
            "pages_processed": 0,
            "chunks_stored": 0,
            "total_chunks_in_db": store.get_count()
        }

    for page in pages:
        page.text = cleaner.clean(page.text)

    valid_pages = [p for p in pages if p.text and p.text.strip()]

    if not valid_pages:
        return {
            "status": "warning",
            "message": "No extractable text was found in the document.",
            "files_processed": file_count,
            "pages_processed": len(pages),
            "chunks_stored": 0,
            "total_chunks_in_db": store.get_count()
        }

    chunker = TextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    chunks = chunker.chunk_documents(valid_pages)
    added_count = store.add_chunks(chunks)

    return {
        "status": "success",
        "files_processed": file_count,
        "pages_processed": len(valid_pages),
        "chunks_stored": added_count,
        "total_chunks_in_db": store.get_count()
    }


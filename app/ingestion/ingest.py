"""
Production-ready page-batched streaming document ingestion pipeline with memory management,
incremental vector persistence, and graceful failure recovery.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from app.config.settings import settings
from app.core.exceptions import (
    RAGException,
    ResourceUnsafeError,
    PDFValidationError,
    PDFExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
)
from app.core.resource_guard import ResourceGuard
from app.core.state import OperationState, StateTracker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import TextChunker, Chunk
from app.ingestion.loader import DocumentLoader, DocumentPage

if TYPE_CHECKING:
    from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def ingest_documents(
    path: str | Path,
    vector_store: Optional["ChromaVectorStore"] = None,
    reset: bool = False,
    batch_size: int = 20
) -> Dict[str, Any]:
    """
    Ingests documents into vector store using streaming page-batch processing.
    Ensures safe memory utilization, incremental persistence, and robust failure recovery.
    """
    tracker = StateTracker(OperationState.VALIDATING)
    target = Path(path)

    if not target.exists():
        tracker.transition_to(OperationState.RECOVERING, error=f"Ingestion path does not exist: {target}")
        ResourceGuard.collect_garbage()
        tracker.transition_to(OperationState.READY)
        raise FileNotFoundError(f"Ingestion path does not exist: {target}")

    from app.vectorstore.chroma_store import ChromaVectorStore
    try:
        store = vector_store or ChromaVectorStore()
        if reset:
            store.reset()
    except Exception as e:
        tracker.transition_to(OperationState.RECOVERING, error=f"Vector store initialization failed: {e}")
        ResourceGuard.collect_garbage()
        tracker.transition_to(OperationState.READY)
        return {
            "status": "error",
            "error": f"Vector store initialization failed: {e}",
            "files_processed": 0,
            "pages_processed": 0,
            "chunks_stored": 0,
            "total_chunks_in_db": 0
        }

    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # File discovery & validation phase
    files_to_process: List[Path] = []
    if target.is_file():
        files_to_process.append(target)
    elif target.is_dir():
        files_to_process = [
            p for p in target.glob("**/*")
            if p.suffix.lower() in (".pdf", ".txt", ".md", ".text")
        ]
    else:
        raise ValueError(f"Invalid path type for ingestion: {target}")

    if not files_to_process:
        return {
            "status": "warning",
            "message": f"No supported documents (.pdf, .txt) found in '{target}'.",
            "files_processed": 0,
            "pages_processed": 0,
            "chunks_stored": 0,
            "total_chunks_in_db": store.get_count()
        }

    # Deduplication check using RAM-friendly targeted hash lookup
    new_files: List[Path] = []
    for f in files_to_process:
        try:
            val_info = loader.validate_file(f)
            if not reset and store.has_file_hash(val_info["file_hash"]):
                logger.info(f"File '{f.name}' already ingested. Skipping.")
                continue
            new_files.append(f)
        except Exception as e:
            if target.is_file():
                logger.error(f"Validation failed for file {f.name}: {e}")
                tracker.transition_to(OperationState.RECOVERING, error=str(e))
                ResourceGuard.collect_garbage()
                tracker.transition_to(OperationState.READY)
                return {
                    "status": "error",
                    "error": f"Ingestion failed safely: {str(e)}",
                    "files_processed": 0,
                    "pages_processed": 0,
                    "chunks_stored": 0,
                    "total_chunks_in_db": store.get_count()
                }
            else:
                logger.warning(f"Validation failed for file {f.name}: {e}")

    if not new_files:
        return {
            "status": "skipped",
            "message": "All documents in specified path already exist in vector store.",
            "files_processed": 0,
            "pages_processed": 0,
            "chunks_stored": 0,
            "total_chunks_in_db": store.get_count()
        }

    # Ingestion processing loop
    tracker.transition_to(OperationState.INGESTING)
    total_files_processed = 0
    total_pages_processed = 0
    total_chunks_stored = 0
    chunk_counters: Dict[str, int] = {}

    try:
        for file_path in new_files:
            val_info = loader.validate_file(file_path)
            file_ext = val_info["file_type"]

            if file_ext == "pdf":
                # Calculate safe batch size based on available RAM
                safe_batch = ResourceGuard.calculate_safe_batch_size(
                    total_items=val_info["total_pages"],
                    default_batch_size=batch_size
                )
                for page_batch in loader.stream_pdf_batches(file_path, batch_size=safe_batch):
                    if not page_batch:
                        continue
                    
                    # Clean page text
                    for page in page_batch:
                        page.text = cleaner.clean(page.text)

                    valid_pages = [p for p in page_batch if p.text and p.text.strip()]
                    if not valid_pages:
                        continue

                    total_pages_processed += len(valid_pages)

                    # Chunk valid pages incrementally
                    batch_chunks = chunker.chunk_documents(
                        valid_pages, start_indices=chunk_counters
                    )

                    if batch_chunks:
                        # Add chunks to vector store incrementally
                        added = store.add_chunks(batch_chunks)
                        total_chunks_stored += added

                    # Release batch memory
                    page_batch.clear()
                    batch_chunks.clear()
                    ResourceGuard.collect_garbage()

            else:
                # Text/Markdown files
                pages = loader.load_file(file_path)
                for page in pages:
                    page.text = cleaner.clean(page.text)
                valid_pages = [p for p in pages if p.text and p.text.strip()]
                if valid_pages:
                    total_pages_processed += len(valid_pages)
                    batch_chunks = chunker.chunk_documents(
                        valid_pages, start_indices=chunk_counters
                    )
                    if batch_chunks:
                        added = store.add_chunks(batch_chunks)
                        total_chunks_stored += added

            total_files_processed += 1

        tracker.transition_to(OperationState.READY)
        return {
            "status": "success",
            "message": f"Successfully ingested {total_files_processed} document(s).",
            "files_processed": total_files_processed,
            "pages_processed": total_pages_processed,
            "chunks_stored": total_chunks_stored,
            "total_chunks_in_db": store.get_count()
        }

    except Exception as e:
        logger.error(f"Ingestion pipeline failure: {e}")
        tracker.transition_to(OperationState.RECOVERING, error=str(e))
        ResourceGuard.collect_garbage()
        tracker.transition_to(OperationState.READY)

        return {
            "status": "error",
            "error": f"Ingestion failed safely: {str(e)}",
            "files_processed": total_files_processed,
            "pages_processed": total_pages_processed,
            "chunks_stored": total_chunks_stored,
            "total_chunks_in_db": store.get_count()
        }

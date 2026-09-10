import argparse
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.ingestion.ingest import ingest_documents
from app.vectorstore.chroma_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--path", "-p", required=True, help="Path to PDF/TXT file or directory")
    parser.add_argument("--reset", "-r", action="store_true", help="Reset vector store before ingesting")
    args = parser.parse_args()

    start_time = time.perf_counter()
    store = ChromaVectorStore()
    result = ingest_documents(path=args.path, vector_store=store, reset=args.reset)
    elapsed = time.perf_counter() - start_time

    print(f"\n--- Ingestion Results ---")
    print(f"Status:             {result.get('status', '').upper()}")
    print(f"Message:            {result.get('message', 'Completed successfully')}")
    print(f"Files Processed:    {result.get('files_processed', 0)}")
    print(f"Pages Processed:    {result.get('pages_processed', 0)}")
    print(f"Chunks Stored:      {result.get('chunks_stored', 0)}")
    print(f"Total DB Chunks:    {result.get('total_chunks_in_db', 0)}")
    print(f"Time Taken:         {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()

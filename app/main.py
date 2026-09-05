import argparse
import os
import sys
from pathlib import Path

# Add root directory to sys.path if running as script
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import settings
from app.ingestion import ingest_documents
from app.retrieval import ChromaVectorStore, VectorRetriever
from app.generation import RAGGenerator, CitationFormatter


def handle_ingest(args):
    """Handles document ingestion (PDF & TXT)."""
    path = args.path
    reset = args.reset
    print(f"\n--- Ingesting Documents from: {path} ---")
    try:
        result = ingest_documents(path=path, reset=reset)
        print(f"Status:              {result.get('status').upper()}")
        print(f"Files Processed:     {result.get('files_processed')}")
        print(f"Pages/Docs Loaded:   {result.get('pages_processed')}")
        print(f"Chunks Added:        {result.get('chunks_stored')}")
        print(f"Total Chunks in DB:  {result.get('total_chunks_in_db')}\n")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)


def handle_ask(args):
    """Handles query retrieval and grounded answer generation."""
    query = args.question
    top_k = args.top_k or settings.TOP_K

    vector_store = ChromaVectorStore()
    if vector_store.get_count() == 0:
        print("Warning: Vector database is empty. Please run 'ingest' command first.\n")

    retriever = VectorRetriever(vector_store=vector_store)
    retrieved_chunks = retriever.retrieve(query=query, top_k=top_k)

    generator = RAGGenerator()
    result = generator.generate(query=query, retrieved_chunks=retrieved_chunks)

    # Clean user output (Answer + Sources block)
    print("\n--- Answer ---")
    print(result.answer)
    print()

    # Developer verbose output if requested
    if args.verbose and result.retrieved_chunks:
        print("--- Developer Diagnostics (Retrieved Context Chunks) ---")
        for idx, chunk in enumerate(result.retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            score = chunk.get("score", 0.0)
            page = meta.get("page_number", meta.get("page", "N/A"))
            print(f"[{idx}] Chunk ID: {chunk.get('chunk_id')} | Source: {meta.get('source')} | Page: {page} | Score: {score}")
            print(f"    Snippet: {chunk.get('text', '')[:120]}...\n")


def handle_interactive(args):
    """Interactive CLI chat session for domain Q&A."""
    top_k = args.top_k or settings.TOP_K
    vector_store = ChromaVectorStore()
    retriever = VectorRetriever(vector_store=vector_store)
    generator = RAGGenerator()

    print("\n==================================================")
    print("  Ask My Docs — Interactive RAG Pipeline CLI ")
    print("==================================================")
    print(f"Vector Store Chunk Count: {vector_store.get_count()}")
    print("Type 'exit' or 'quit' to end session.\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            retrieved_chunks = retriever.retrieve(query=query, top_k=top_k)
            result = generator.generate(query=query, retrieved_chunks=retrieved_chunks)

            print("\n--- Answer ---")
            print(result.answer)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break


def handle_stats(args):
    """Prints vector database statistics."""
    store = ChromaVectorStore()
    count = store.get_count()
    print(f"\n--- Vector Database Statistics ---")
    print(f"Persist Directory: {store.persist_directory}")
    print(f"Collection Name:   {store.collection_name}")
    print(f"Total Chunks:      {count}\n")


def handle_reset(args):
    """Resets and clears the vector database collection."""
    store = ChromaVectorStore()
    store.reset()
    print("\nVector store collection has been reset and cleared.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ask My Docs — Phase 1 RAG Fundamentals CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF/TXT document(s)")
    ingest_parser.add_argument(
        "--path", "-p", required=True, help="Path to PDF/TXT file or directory"
    )
    ingest_parser.add_argument(
        "--reset", "-r", action="store_true", help="Reset vectorstore before ingesting"
    )
    ingest_parser.set_defaults(func=handle_ingest)

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested docs")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument("--top-k", "-k", type=int, help="Number of chunks to retrieve")
    ask_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print developer diagnostics (retrieved chunks)"
    )
    ask_parser.set_defaults(func=handle_ask)

    # Interactive subcommand
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive CLI Q&A session")
    interactive_parser.add_argument("--top-k", "-k", type=int, help="Number of chunks to retrieve")
    interactive_parser.set_defaults(func=handle_interactive)

    # Stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Print vector store info")
    stats_parser.set_defaults(func=handle_stats)

    # Reset subcommand
    reset_parser = subparsers.add_parser("reset", help="Reset and clear vector store")
    reset_parser.set_defaults(func=handle_reset)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config import settings
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.vector_retriever import VectorRetriever
from app.generation.answer_generator import RAGGenerator


def handle_ask(args, vector_store):
    query = args.question
    top_k = args.top_k or settings.TOP_K

    if vector_store.get_count() == 0:
        print("\nWarning: Vector database is empty. Please run 'python ingest.py --path <pdf>' first.\n")
        return

    retriever = VectorRetriever(vector_store=vector_store)
    retrieved_chunks = retriever.retrieve(query=query, top_k=top_k)

    generator = RAGGenerator()
    result = generator.generate(query=query, retrieved_chunks=retrieved_chunks)

    print("\n--- Answer ---")
    print(result.answer)
    print()

    if getattr(args, "verbose", False) and result.retrieved_chunks:
        print("--- Developer Diagnostics (Retrieved Context Chunks) ---")
        for idx, chunk in enumerate(result.retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            score = chunk.get("score", 0.0)
            page = meta.get("page_number", meta.get("page", "N/A"))
            print(f"[{idx}] Chunk ID: {chunk.get('chunk_id')} | Source: {meta.get('source')} | Page: {page} | Score: {score}")
            print(f"    Snippet: {chunk.get('text', '')[:120]}...\n")


def handle_interactive(args, vector_store):
    top_k = getattr(args, "top_k", None) or settings.TOP_K
    retriever = VectorRetriever(vector_store=vector_store)
    generator = RAGGenerator()

    print("\n==================================================")
    print("  Ask My Docs — RAG Interactive Assistant ")
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


def handle_stats(args, vector_store):
    count = vector_store.get_count()
    print(f"\n--- Vector Database Statistics ---")
    print(f"Persist Directory: {vector_store.persist_directory}")
    print(f"Collection Name:   {vector_store.collection_name}")
    print(f"Total Chunks:      {count}\n")


def main():
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(description="Ask My Docs RAG Application")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested docs")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument("--top-k", "-k", type=int, help="Number of chunks to retrieve")
    ask_parser.add_argument("--verbose", "-v", action="store_true", help="Print developer diagnostics")

    interactive_parser = subparsers.add_parser("interactive", help="Start interactive CLI Q&A session")
    interactive_parser.add_argument("--top-k", "-k", type=int, help="Number of chunks to retrieve")

    stats_parser = subparsers.add_parser("stats", help="Print vector store info")

    args = parser.parse_args()

    vector_store = ChromaVectorStore()
    startup_time = time.perf_counter() - t0

    if os.getenv("DEBUG_RAG", "false").lower() == "true":
        print(f"[DEBUG] Application startup completed in {startup_time:.4f}s")

    if args.command == "ask":
        handle_ask(args, vector_store)
    elif args.command == "interactive":
        handle_interactive(args, vector_store)
    elif args.command == "stats":
        handle_stats(args, vector_store)
    else:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            args.question = sys.argv[1]
            args.top_k = None
            args.verbose = False
            handle_ask(args, vector_store)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

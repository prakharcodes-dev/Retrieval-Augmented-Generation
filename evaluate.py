"""
CLI command-line interface for running RAG evaluations and retrieval method comparisons.
Usage:
    python evaluate.py
    python evaluate.py --compare
    python evaluate.py --top-k 5 --generate
    python evaluate.py --output report.json --csv report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure root dir is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.config.settings import settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.comparator import RetrievalComparator
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.bm25_index import BM25Index


def main():
    parser = argparse.ArgumentParser(description="RAG Document Assistant — Benchmark & Evaluation Tool")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/evaluation/default_eval_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=settings.TOP_K,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare Vector Search vs BM25 Search vs Hybrid RRF Search",
    )
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Also evaluate answer generation quality (Faithfulness, Answer Relevancy)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save evaluation report as JSON",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to save query evaluation report as CSV",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Evaluation dataset not found at '{dataset_path}'.")
        sys.exit(1)

    print("==================================================")
    print("  RAG Document Assistant — Evaluation Suite       ")
    print("==================================================")
    print(f"Loading dataset: {dataset_path}")
    dataset = EvaluationDataset.load_from_file(dataset_path)
    summary = dataset.summary()
    print(f"Dataset: {summary['name']} (v{summary['version']})")
    print(f"Total evaluation samples: {summary['total_samples']}")
    print(f"Categories: {summary['categories']}\n")

    print("Initializing vector store and retriever...")
    vector_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
    )
    sparse_index = BM25Index(Path(settings.CHROMA_PERSIST_DIR) / "bm25.sqlite3")
    retriever = VectorRetriever(vector_store=vector_store, sparse_index=sparse_index)

    doc_count = vector_store.get_count()
    print(f"Vector store chunk count: {doc_count}")
    if doc_count == 0:
        print("Warning: Vector store is currently empty. Run 'python ingest.py --path <doc>' first to index documents.\n")

    if args.compare:
        print(f"\n--- Running 3-Way Retrieval Method Comparison (Top-K={args.top_k}) ---")
        comparator = RetrievalComparator(top_k=args.top_k)
        report = comparator.compare_methods(
            dataset=dataset,
            vector_store=vector_store,
            sparse_index=sparse_index,
            hybrid_retriever=retriever,
            top_k=args.top_k,
        )

        print("\n" + "=" * 80)
        print("RETRIEVAL METHOD COMPARISON SUMMARY")
        print("=" * 80)
        df = report.to_dataframe()
        print(df.to_string(index=False))
        print("-" * 80)
        print("Key Findings:")
        for f in report.key_findings:
            print(f"  • {f}")
        print("=" * 80 + "\n")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report.to_json(), encoding="utf-8")
            print(f"JSON report saved to: {out_p}")

        if args.csv:
            csv_p = Path(args.csv)
            csv_p.parent.mkdir(parents=True, exist_ok=True)
            csv_p.write_text(report.to_csv(), encoding="utf-8")
            print(f"CSV report saved to: {csv_p}")

    else:
        print(f"\n--- Running RAG Evaluation (Top-K={args.top_k}, Generate={args.generate}) ---")
        evaluator = RAGEvaluator(top_k=args.top_k)
        generator = None
        if args.generate:
            from app.generation.answer_generator import RAGGenerator
            generator = RAGGenerator()

        report = evaluator.evaluate_pipeline(
            dataset=dataset,
            retriever=retriever,
            generator=generator,
            top_k=args.top_k,
            evaluate_answers=args.generate,
        )

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Queries Evaluated:    {report.total_queries}")
        print(f"Precision@{report.top_k}:             {report.mean_precision:.4f}")
        print(f"Recall@{report.top_k}:                {report.mean_recall:.4f}")
        print(f"Hit Rate@{report.top_k}:              {report.hit_rate:.4f}")
        print(f"MRR (Mean Reciprocal Rank): {report.mrr:.4f}")
        print(f"Context Precision:          {report.mean_context_precision:.4f}")
        print(f"Avg Retrieval Latency:      {report.mean_retrieval_latency_ms:.2f} ms")
        if report.mean_faithfulness is not None:
            print(f"Faithfulness Score:         {report.mean_faithfulness:.4f}")
        if report.mean_answer_relevancy is not None:
            print(f"Answer Relevancy Score:     {report.mean_answer_relevancy:.4f}")
        if report.mean_generation_latency_ms is not None:
            print(f"Avg Generation Latency:     {report.mean_generation_latency_ms:.2f} ms")
        print("=" * 60 + "\n")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report.to_json(), encoding="utf-8")
            print(f"JSON report saved to: {out_p}")

        if args.csv:
            csv_p = Path(args.csv)
            csv_p.parent.mkdir(parents=True, exist_ok=True)
            csv_p.write_text(report.to_csv(), encoding="utf-8")
            print(f"CSV report saved to: {csv_p}")


if __name__ == "__main__":
    main()

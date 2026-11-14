"""
Streamlit Web UI for RAG Document Assistant.
Features state tracking, rerun safety, resource monitoring, batched ingestion,
document validation, RAG evaluation, retrieval comparison, and performance tracking.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

from app.config.settings import settings
from app.core.exceptions import RAGException, ResourceUnsafeError
from app.core.performance_tracker import performance_tracker
from app.core.resource_guard import ResourceGuard
from app.core.state import OperationState, StateTracker
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.comparator import RetrievalComparator
from app.generation.answer_generator import RAGGenerator
from app.ingestion.ingest import ingest_documents
from app.ingestion.validator import DocumentValidator
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from reddis_metadata import save_metadata as save_reddis_metadata

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_vector_store():
    embedding_service = EmbeddingService()
    return ChromaVectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        embedding_service=embedding_service
    )


@st.cache_resource
def get_sparse_index():
    return BM25Index(Path(settings.CHROMA_PERSIST_DIR) / "bm25.sqlite3")


@st.cache_resource
def get_generator():
    return RAGGenerator()


@st.cache_resource
def get_retriever():
    return VectorRetriever(vector_store=get_vector_store(), sparse_index=get_sparse_index())


def main():
    st.title("📚 RAG Document Assistant")
    st.caption("Production-Ready Retrieval-Augmented Generation System with Benchmark Evaluation & Performance Tracking")

    # Session state initialization for rerun idempotency and state tracking
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = set()
    if "app_state" not in st.session_state:
        st.session_state.app_state = OperationState.READY
    if "eval_report" not in st.session_state:
        st.session_state.eval_report = None
    if "comparison_report" not in st.session_state:
        st.session_state.comparison_report = None

    try:
        vector_store = get_vector_store()
        sparse_index = get_sparse_index()
        retriever = get_retriever()
    except Exception as exc:
        st.error(f"Cannot initialize document search: {exc}")
        st.info("Check the configured embedding credentials and vector index. If this is a BM25 synchronization error, restart after resolving the local SQLite/index issue; retrieval will not continue with dense-only results.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration & Status")

        # Memory & Vector Store metrics
        mem_stats = ResourceGuard.get_memory_stats()
        chunk_count = vector_store.get_count()
        doc_count = len(vector_store.get_existing_hashes())

        col1, col2 = st.columns(2)
        col1.metric("Documents", doc_count)
        col2.metric("Total Chunks", chunk_count)

        col3, col4 = st.columns(2)
        col3.metric("Free RAM (MB)", f"{mem_stats['available_ram_mb']:.0f}")
        col4.metric("State", st.session_state.app_state.value)

        st.divider()

        st.subheader("📤 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Document (.pdf, .txt, .md)",
            type=["pdf", "txt", "md", "text"]
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            # FEATURE 3: Deep Document Validation
            val_res = DocumentValidator.validate_upload(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                vector_store=vector_store,
            )

            if not val_res.is_valid:
                st.error(f"❌ {val_res.error_message}")
            elif val_res.is_duplicate or val_res.file_hash in st.session_state.processed_hashes:
                st.info(f"ℹ️ Document '{val_res.sanitized_filename}' is already indexed (hash: {val_res.file_hash[:8]}...).")
            else:
                st.success(f"✓ Valid: **{val_res.sanitized_filename}** ({val_res.file_size_mb:.2f} MB, {val_res.total_pages} page(s))")
                for w in val_res.warnings:
                    st.warning(f"⚠️ {w}")

                if st.button("Process Document", type="primary"):
                    st.session_state.app_state = OperationState.INGESTING
                    with st.spinner("Validating and processing document in memory-safe batches..."):
                        tmp_path = None
                        try:
                            # Safe temporary file staging
                            tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            tmp_path = tmp_dir / f"{val_res.file_hash[:8]}_{val_res.sanitized_filename}"
                            tmp_path.write_bytes(file_bytes)

                            # Ingest via streaming page batches
                            result = ingest_documents(tmp_path, vector_store=vector_store, reset=False)

                            if result["status"] == "success":
                                st.session_state.processed_hashes.add(val_res.file_hash)
                                save_reddis_metadata(f"rag:document:{val_res.file_hash}", {
                                    "project": "rag",
                                    "documentHash": val_res.file_hash,
                                    "status": "indexed",
                                    "chunksStored": result.get("chunks_stored", 0),
                                    "savedAt": datetime.now(timezone.utc).isoformat()
                                })
                                st.session_state.app_state = OperationState.READY
                                duration_msg = f" in {result.get('duration_seconds', 0):.2f}s" if "duration_seconds" in result else ""
                                st.success(f"Successfully ingested '{val_res.sanitized_filename}' ({result['chunks_stored']} chunks stored{duration_msg}).")
                                st.rerun()
                            elif result["status"] == "skipped":
                                st.session_state.processed_hashes.add(val_res.file_hash)
                                st.session_state.app_state = OperationState.READY
                                st.info(f"Document '{val_res.sanitized_filename}' already exists in vector store.")
                            else:
                                st.session_state.app_state = OperationState.RECOVERING
                                st.error(f"Ingestion failed: {result.get('error', 'Unknown error')}")
                                ResourceGuard.collect_garbage()
                                st.session_state.app_state = OperationState.READY

                        except Exception as e:
                            st.session_state.app_state = OperationState.RECOVERING
                            st.error(f"Document processing error: {str(e)}")
                            ResourceGuard.collect_garbage()
                            st.session_state.app_state = OperationState.READY
                        finally:
                            if tmp_path is not None and tmp_path.exists():
                                try:
                                    tmp_path.unlink()
                                except Exception:
                                    pass

        st.divider()

        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=settings.TOP_K)
        debug_mode = st.toggle("Developer Debug Mode", value=False)

        st.divider()
        st.caption("Clears all indexed documents for every user of this app.")
        if st.button("🧨 Clear Vector Database", key="clear_vector_database", type="primary"):
            try:
                # Clear the sparse side first. If Chroma then fails,
                # restore BM25 from Chroma before showing the error.
                sparse_index.reset()
                vector_store.reset()
                st.session_state.processed_hashes.clear()
                st.session_state.messages.clear()
                st.success("The active vector database and BM25 index were cleared.")
                st.rerun()
            except Exception as exc:
                try:
                    sparse_index.sync_from_store(vector_store)
                except Exception as sync_exc:
                    st.error(
                        "Database clearing failed and index reconciliation also failed "
                        f"({type(sync_exc).__name__}). Search may remain unavailable until repaired."
                    )
                else:
                    st.error(f"Could not clear the database ({type(exc).__name__}); existing data was reconciled where possible.")

        st.divider()

    # Organize features into tabs
    tab_chat, tab_eval, tab_compare, tab_perf = st.tabs([
        "💬 Chat Assistant",
        "🔬 RAG Evaluation",
        "⚖️ Retrieval Comparison",
        "⚡ Performance Monitor"
    ])

    # -------------------------------------------------------------
    # TAB 1: Chat Assistant (Preserves existing chatbot interface)
    # -------------------------------------------------------------
    with tab_chat:
        # Starter questions
        if not st.session_state.messages and doc_count > 0:
            st.markdown("##### 💡 Starter Questions")
            q_cols = st.columns(3)
            sample_questions = [
                "What is the summary of the uploaded document?",
                "What are the key points covered in this file?",
                "What details or guidelines are mentioned?"
            ]
            for idx, sq in enumerate(sample_questions):
                if q_cols[idx % 3].button(sq, key=f"sq_{idx}"):
                    st.session_state.pending_prompt = sq
                    st.rerun()

        # Render Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timing" in message and message["timing"]:
                    t_info = message["timing"]
                    st.caption(
                        f"⏱️ Retrieval: {t_info.get('retrieval_ms', 0):.1f}ms ({t_info.get('method', 'Hybrid')}) | "
                        f"Generation: {t_info.get('generation_ms', 0) / 1000.0:.2f}s | "
                        f"Total: {t_info.get('total_ms', 0) / 1000.0:.2f}s"
                    )
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources & Citations"):
                        for src in message["sources"]:
                            st.write(f"- **{src.get('source', 'Unknown')}** (Page {src.get('page_number', 'N/A')})")
                if debug_mode and "chunks" in message and message["chunks"]:
                    with st.expander("🛠️ Debug: Retrieved Chunks"):
                        st.json(message["chunks"])

        retry_prompt = getattr(st.session_state, "retry_prompt", None)
        if retry_prompt and st.button("↻ Retry the last question", key="retry_last_question"):
            st.session_state.pending_prompt = retry_prompt
            st.session_state.retry_prompt = None
            st.rerun()

        prompt = st.chat_input("Ask a question about your documents...")
        if getattr(st.session_state, "pending_prompt", None):
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        if prompt:
            # Bound per-session rendering and memory as chats get longer.
            st.session_state.messages = st.session_state.messages[-18:]
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if doc_count == 0:
                    resp_text = "No documents found in vector store. Please upload a document using the sidebar to begin."
                    st.warning(resp_text)
                    st.session_state.messages.append({"role": "assistant", "content": resp_text})
                else:
                    st.session_state.app_state = OperationState.RETRIEVING
                    with st.spinner("Searching documents & generating answer..."):
                        try:
                            # FEATURE 4: Timing retrieval
                            t0_retrieval = time.perf_counter()
                            retrieved = retriever.retrieve(prompt, top_k=top_k)
                            t_retrieval_ms = (time.perf_counter() - t0_retrieval) * 1000.0

                            st.session_state.app_state = OperationState.GENERATING
                            generator = get_generator()

                            # FEATURE 4: Timing generation
                            t0_gen = time.perf_counter()
                            result = generator.generate(prompt, retrieved)
                            t_gen_ms = (time.perf_counter() - t0_gen) * 1000.0
                            t_total_ms = t_retrieval_ms + t_gen_ms

                            st.markdown(result.answer)
                            if result.model_used and result.model_used != generator.model_name:
                                st.caption(f"Fallback model used: {result.model_used}")

                            st.caption(
                                f"⏱️ Retrieval: {t_retrieval_ms:.1f}ms (Hybrid RRF) | "
                                f"Generation: {t_gen_ms / 1000.0:.2f}s | "
                                f"Total: {t_total_ms / 1000.0:.2f}s"
                            )

                            if result.citations:
                                with st.expander("📚 View Sources & Citations"):
                                    for c in result.citations:
                                        st.write(f"- **{c.get('source', 'Unknown')}** (Page {c.get('page_number', 'N/A')})")

                            if debug_mode and result.retrieved_chunks:
                                with st.expander("🛠️ Debug: Retrieved Chunks"):
                                    st.json(result.retrieved_chunks)

                            # Record in performance tracker
                            performance_tracker.record_query(
                                query=prompt,
                                retrieval_method="Hybrid RRF",
                                retrieval_ms=t_retrieval_ms,
                                generation_ms=t_gen_ms,
                                chunks_retrieved=len(retrieved),
                                model_used=result.model_used or generator.model_name,
                            )

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result.answer,
                                "sources": result.citations,
                                "chunks": result.retrieved_chunks,
                                "timing": {
                                    "retrieval_ms": t_retrieval_ms,
                                    "generation_ms": t_gen_ms,
                                    "total_ms": t_total_ms,
                                    "method": "Hybrid RRF",
                                }
                            })
                            st.session_state.retry_prompt = None
                            st.session_state.app_state = OperationState.READY

                        except Exception as e:
                            st.session_state.app_state = OperationState.RECOVERING
                            from app.core.exceptions import TransientGenerationError
                            err_msg = f"An error occurred while generating response: {str(e)}"
                            if isinstance(e, TransientGenerationError):
                                st.session_state.retry_prompt = prompt
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
                            ResourceGuard.collect_garbage()
                            st.session_state.app_state = OperationState.READY

    # -------------------------------------------------------------
    # TAB 2: RAG Evaluation System (FEATURE 1)
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader("🔬 RAG Evaluation & Testing Suite")
        st.markdown(
            "Evaluate retrieval quality (**Precision@K**, **Recall@K**, **Hit Rate**, **MRR**, **Context Precision**) "
            "and answer quality (**Faithfulness**, **Answer Relevancy**) against ground truth test cases."
        )

        dataset_path = Path("data/evaluation/default_eval_dataset.json")
        eval_dataset = None
        if dataset_path.exists():
            try:
                eval_dataset = EvaluationDataset.load_from_file(dataset_path)
            except Exception as e:
                st.error(f"Error loading evaluation dataset: {e}")

        if eval_dataset:
            summary = eval_dataset.summary()
            c_info1, c_info2, c_info3 = st.columns(3)
            c_info1.metric("Benchmark Name", summary["name"])
            c_info2.metric("Total Test Cases", summary["total_samples"])
            c_info3.metric("Version", summary["version"])

            categories = ["All"] + eval_dataset.get_categories()
            c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 1, 1])
            selected_cat = c_ctrl1.selectbox("Filter Category", categories, index=0)
            eval_k = c_ctrl2.slider("Evaluation Top-K", 1, 10, 5, key="eval_k_slider")
            enable_gen = c_ctrl3.checkbox("Evaluate Answers", value=False, help="Runs answer generator to compute Faithfulness and Answer Relevancy.")

            target_ds = eval_dataset if selected_cat == "All" else eval_dataset.filter_by_category(selected_cat)

            if st.button("🚀 Run RAG Evaluation", type="primary", key="btn_run_eval"):
                with st.spinner(f"Evaluating {len(target_ds)} test cases across retrieval pipeline..."):
                    evaluator = RAGEvaluator(top_k=eval_k)
                    generator = get_generator() if enable_gen else None
                    report = evaluator.evaluate_pipeline(
                        dataset=target_ds,
                        retriever=retriever,
                        generator=generator,
                        top_k=eval_k,
                        evaluate_answers=enable_gen,
                    )
                    st.session_state.eval_report = report

            report = st.session_state.eval_report
            if report:
                st.markdown("### 📊 Evaluation Metrics")
                m_cols = st.columns(5)
                m_cols[0].metric(f"Precision@{report.top_k}", f"{report.mean_precision * 100:.1f}%")
                m_cols[1].metric(f"Recall@{report.top_k}", f"{report.mean_recall * 100:.1f}%")
                m_cols[2].metric("Hit Rate", f"{report.hit_rate * 100:.1f}%")
                m_cols[3].metric("MRR", f"{report.mrr:.4f}")
                m_cols[4].metric("Context Precision", f"{report.mean_context_precision:.4f}")

                if report.mean_faithfulness is not None or report.mean_answer_relevancy is not None:
                    g_cols = st.columns(3)
                    if report.mean_faithfulness is not None:
                        g_cols[0].metric("Faithfulness Score", f"{report.mean_faithfulness * 100:.1f}%")
                    if report.mean_answer_relevancy is not None:
                        g_cols[1].metric("Answer Relevancy", f"{report.mean_answer_relevancy * 100:.1f}%")
                    g_cols[2].metric("Avg Retrieval Latency", f"{report.mean_retrieval_latency_ms:.1f} ms")
                else:
                    st.caption(f"⏱️ Average Retrieval Latency: **{report.mean_retrieval_latency_ms:.2f} ms**")

                st.markdown("#### 📋 Detailed Query Results")
                df_results = report.to_dataframe()
                st.dataframe(df_results, use_container_width=True)

                d_col1, d_col2 = st.columns(2)
                d_col1.download_button(
                    label="📥 Download Report (JSON)",
                    data=report.to_json(),
                    file_name=f"rag_evaluation_report_{report.top_k}k.json",
                    mime="application/json",
                )
                d_col2.download_button(
                    label="📥 Download Results (CSV)",
                    data=report.to_csv(),
                    file_name=f"rag_evaluation_results_{report.top_k}k.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No evaluation dataset found in `data/evaluation/default_eval_dataset.json`.")

    # -------------------------------------------------------------
    # TAB 3: Retrieval Method Comparison (FEATURE 2)
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("⚖️ Retrieval Method Comparison")
        st.markdown(
            "Empirically benchmark and compare **Vector Search (Dense)**, **BM25 Search (Sparse)**, "
            "and **Hybrid Search (Dense + Sparse RRF)** across identical queries under identical conditions."
        )

        cmp_k = st.slider("Comparison Top-K", 1, 10, 5, key="cmp_k_slider")

        if st.button("⚖️ Compare All 3 Retrieval Methods", type="primary", key="btn_run_compare"):
            if not eval_dataset:
                st.error("Evaluation dataset is required for comparison.")
            else:
                with st.spinner("Benchmarking Vector, BM25, and Hybrid RRF across evaluation test suite..."):
                    comparator = RetrievalComparator(top_k=cmp_k)
                    comparison_report = comparator.compare_methods(
                        dataset=eval_dataset,
                        vector_store=vector_store,
                        sparse_index=sparse_index,
                        hybrid_retriever=retriever,
                        top_k=cmp_k,
                    )
                    st.session_state.comparison_report = comparison_report

        cmp_rep = st.session_state.comparison_report
        if cmp_rep:
            st.success(f"🏆 Best Overall Retrieval Method: **{cmp_rep.best_overall_method}**")

            with st.expander("📌 Key Benchmark Findings", expanded=True):
                for f in cmp_rep.key_findings:
                    st.markdown(f"- {f}")

            st.markdown("#### 📊 Metric Comparison Table")
            cmp_df = cmp_rep.to_dataframe()
            st.dataframe(cmp_df, use_container_width=True)

            # Comparative Charts
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("##### 🎯 Accuracy Metrics Comparison")
                acc_df = cmp_df.set_index("Retrieval Method")[[f"Precision@{cmp_rep.top_k}", f"Recall@{cmp_rep.top_k}", "Hit Rate", "MRR"]]
                st.bar_chart(acc_df)

            with c_chart2:
                st.markdown("##### ⚡ Latency Comparison (ms)")
                lat_df = cmp_df.set_index("Retrieval Method")[["Avg Latency (ms)"]]
                st.bar_chart(lat_df)

            cd_col1, cd_col2 = st.columns(2)
            cd_col1.download_button(
                label="📥 Download Comparison (CSV)",
                data=cmp_rep.to_csv(),
                file_name="retrieval_comparison_report.csv",
                mime="text/csv",
            )
            cd_col2.download_button(
                label="📥 Download Comparison (JSON)",
                data=cmp_rep.to_json(),
                file_name="retrieval_comparison_report.json",
                mime="application/json",
            )

    # -------------------------------------------------------------
    # TAB 4: Performance Monitor (FEATURE 4)
    # -------------------------------------------------------------
    with tab_perf:
        st.subheader("⚡ Retrieval & Generation Performance Monitor")
        st.markdown("Real-time latency metrics, percentiles, method breakdowns, and ingestion throughput tracking.")

        summary = performance_tracker.get_summary()

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Total Queries Tracked", summary["total_queries"])
        p_col2.metric("Avg Retrieval Latency", f"{summary['avg_retrieval_ms']:.1f} ms")
        p_col3.metric("Avg Generation Latency", f"{summary['avg_generation_ms']:.1f} ms")
        p_col4.metric("Avg Total QA Time", f"{summary['avg_total_ms']:.1f} ms")

        p_sub1, p_sub2 = st.columns(2)
        p_sub1.metric("P50 Total Latency", f"{summary['p50_total_ms']:.1f} ms")
        p_sub2.metric("P95 Total Latency", f"{summary['p95_total_ms']:.1f} ms")

        if summary["by_method"]:
            st.markdown("#### 🔍 Performance Breakdown by Retrieval Method")
            method_rows = []
            for method_name, m_data in summary["by_method"].items():
                method_rows.append({
                    "Method": method_name,
                    "Query Count": m_data["count"],
                    "Avg Latency (ms)": m_data["avg_retrieval_ms"],
                    "Min Latency (ms)": m_data["min_ms"],
                    "Max Latency (ms)": m_data["max_ms"],
                })
            st.dataframe(pd.DataFrame(method_rows), use_container_width=True)

        st.markdown("#### 📜 Recent Query Traces")
        recent_queries = performance_tracker.get_recent_query_records(limit=25)
        if recent_queries:
            df_queries = pd.DataFrame(recent_queries)
            st.dataframe(df_queries, use_container_width=True)
        else:
            st.info("No queries tracked yet in this session. Ask questions in the Chat Assistant tab to populate live performance traces.")

        recent_ingests = performance_tracker.get_recent_ingestion_records(limit=10)
        if recent_ingests:
            st.markdown("#### 📦 Recent Ingestion Traces")
            st.dataframe(pd.DataFrame(recent_ingests), use_container_width=True)

        if st.button("🗑️ Reset Performance History", key="btn_reset_perf"):
            performance_tracker.reset()
            st.success("Performance logs cleared.")
            st.rerun()


if __name__ == "__main__":
    main()

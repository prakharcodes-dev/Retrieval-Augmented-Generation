"""
Streamlit Web UI for RAG Document Assistant.
Features state tracking, rerun safety, resource monitoring, and batched ingestion.
"""

import os
import sys
import tempfile
import hashlib
from pathlib import Path
import streamlit as st

from app.config.settings import settings
from app.core.exceptions import RAGException, ResourceUnsafeError
from app.core.resource_guard import ResourceGuard
from app.core.state import OperationState, StateTracker
from app.embeddings.embedding_service import EmbeddingService
from app.generation.answer_generator import RAGGenerator
from app.ingestion.ingest import ingest_documents
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore

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
def get_generator():
    return RAGGenerator()


def main():
    st.title("📚 RAG Document Assistant")
    st.caption("Production-Ready Retrieval-Augmented Generation System")

    # Session state initialization for rerun idempotency and state tracking
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = set()
    if "app_state" not in st.session_state:
        st.session_state.app_state = OperationState.READY

    vector_store = get_vector_store()
    retriever = VectorRetriever(vector_store=vector_store)

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
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in st.session_state.processed_hashes or vector_store.has_file_hash(file_hash):
                st.info(f"Document '{uploaded_file.name}' is already indexed.")
            else:
                if st.button("Process Document", type="primary"):
                    st.session_state.app_state = OperationState.INGESTING
                    with st.spinner("Validating and processing document in memory-safe batches..."):
                        tmp_path = None
                        try:
                            # Resource safety pre-check
                            ResourceGuard.check_pdf_workload(len(file_bytes), filename=uploaded_file.name)

                            # Save to temporary file cleanly
                            suffix = Path(uploaded_file.name).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(file_bytes)
                                tmp_path = Path(tmp.name)

                            # Ingest via streaming page batches
                            result = ingest_documents(tmp_path, vector_store=vector_store, reset=False)

                            if result["status"] == "success":
                                st.session_state.processed_hashes.add(file_hash)
                                st.session_state.app_state = OperationState.READY
                                st.success(f"Successfully ingested '{uploaded_file.name}' ({result['chunks_stored']} chunks stored).")
                                st.rerun()
                            elif result["status"] == "skipped":
                                st.session_state.processed_hashes.add(file_hash)
                                st.session_state.app_state = OperationState.READY
                                st.info(f"Document '{uploaded_file.name}' already exists in vector store.")
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
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

        st.divider()

        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=20, value=settings.TOP_K)
        debug_mode = st.toggle("Developer Debug Mode", value=False)

        st.divider()

        if st.button("🗑️ Clear Vector Database", type="secondary"):
            with st.spinner("Resetting vector store..."):
                vector_store.reset()
                st.session_state.processed_hashes.clear()
                st.session_state.app_state = OperationState.READY
                ResourceGuard.collect_garbage()
                st.success("Vector database cleared successfully!")
                st.rerun()

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
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources & Citations"):
                    for src in message["sources"]:
                        st.write(f"- **{src.get('source', 'Unknown')}** (Page {src.get('page_number', 'N/A')})")
            if debug_mode and "chunks" in message and message["chunks"]:
                with st.expander("🛠️ Debug: Retrieved Chunks"):
                    st.json(message["chunks"])

    prompt = st.chat_input("Ask a question about your documents...")
    if getattr(st.session_state, "pending_prompt", None):
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
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
                        retrieved = retriever.retrieve(prompt, top_k=top_k)

                        st.session_state.app_state = OperationState.GENERATING
                        generator = get_generator()
                        result = generator.generate(prompt, retrieved)

                        st.markdown(result.answer)

                        if result.citations:
                            with st.expander("📚 View Sources & Citations"):
                                for c in result.citations:
                                    st.write(f"- **{c.get('source', 'Unknown')}** (Page {c.get('page_number', 'N/A')})")

                        if debug_mode and result.retrieved_chunks:
                            with st.expander("🛠️ Debug: Retrieved Chunks"):
                                st.json(result.retrieved_chunks)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.citations,
                            "chunks": result.retrieved_chunks
                        })

                        st.session_state.app_state = OperationState.READY

                    except Exception as e:
                        st.session_state.app_state = OperationState.RECOVERING
                        err_msg = f"An error occurred while generating response: {str(e)}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        ResourceGuard.collect_garbage()
                        st.session_state.app_state = OperationState.READY


if __name__ == "__main__":
    main()

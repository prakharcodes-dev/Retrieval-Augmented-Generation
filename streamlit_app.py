import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

from app.config.settings import settings
from app.vectorstore.chroma_store import ChromaVectorStore
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_retriever import VectorRetriever
from app.generation.answer_generator import RAGGenerator
from app.ingestion.loader import DocumentLoader
from app.ingestion.ingest import ingest_documents


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

    vector_store = get_vector_store()
    retriever = VectorRetriever(vector_store=vector_store)

    with st.sidebar:
        st.header("⚙️ Configuration & Status")

        chunk_count = vector_store.get_count()
        doc_count = len(vector_store.get_existing_hashes())

        col1, col2 = st.columns(2)
        col1.metric("Documents", doc_count)
        col2.metric("Total Chunks", chunk_count)

        st.divider()

        st.subheader("📤 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Document (.pdf, .txt, .md)",
            type=["pdf", "txt", "md", "text"]
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Validating and processing document..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = Path(tmp.name)

                        loader = DocumentLoader()
                        loader.load_file(tmp_path)

                        result = ingest_documents(tmp_path, vector_store=vector_store, reset=False)

                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                        if result["status"] == "success":
                            st.success(f"Successfully ingested '{uploaded_file.name}' ({result['chunks_stored']} chunks).")
                            st.cache_resource.clear()
                            st.rerun()
                        elif result["status"] == "skipped":
                            st.info(f"Document '{uploaded_file.name}' already exists in vector store.")
                        else:
                            st.error(f"Ingestion failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"Document validation/processing error: {str(e)}")

        st.divider()

        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=settings.TOP_K)

        debug_mode = st.toggle("Developer Debug Mode", value=False)

        st.divider()

        if st.button("🗑️ Clear Vector Database", type="secondary"):
            with st.spinner("Resetting vector store..."):
                vector_store.reset()
                st.cache_resource.clear()
                st.success("Vector database cleared successfully!")
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                with st.spinner("Searching documents & generating answer..."):
                    try:
                        retrieved = retriever.retrieve(prompt, top_k=top_k)
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

                    except Exception as e:
                        err_msg = f"An error occurred while generating response: {str(e)}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})


if __name__ == "__main__":
    main()

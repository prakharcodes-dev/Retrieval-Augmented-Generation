import os
import time
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

from app.config import settings
from app.ingestion import ingest_documents
from app.retrieval import ChromaVectorStore, VectorRetriever
from app.generation import RAGGenerator, CitationFormatter

# Page Configuration
st.set_page_config(
    page_title="Ask My Docs — RAG AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Modern UI
st.markdown("""
<style>
    /* Main Background & Font Styling */
    .stApp {
        background-color: #0b0f17;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header Banner Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px 30px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Citation Pill Styling */
    .citation-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        margin: 4px 6px 4px 0;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease-in-out;
    }
    .citation-pill:hover {
        background-color: rgba(56, 189, 248, 0.22);
        border-color: #38bdf8;
    }

    /* Source Reference Container */
    .sources-block {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sources-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom Info Cards */
    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>📚 Ask My Docs <span class="status-badge">🟢 Phase 1 Vector RAG Online</span></h1>
    <p>Upload domain documents (PDF / TXT) to perform semantic vector retrieval and generate grounded answers with source citations.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Setup & Vector Store Initialization
with st.sidebar:
    st.markdown("### ⚙️ RAG Control Panel")

    vector_store = ChromaVectorStore()
    retriever = VectorRetriever(vector_store=vector_store)
    generator = RAGGenerator()

    # Live Database Metrics
    chunk_count = vector_store.get_count()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Indexed Chunks", value=chunk_count)
    with col2:
        st.metric(label="Token Target", value=f"{settings.CHUNK_SIZE}")

    st.markdown("---")
    st.markdown("### 📄 Document Ingestion Hub")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF or TXT documents to chunk, embed, and store in ChromaDB."
    )

    if st.button("🚀 Ingest Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            session_id = f"upload_{int(time.time())}"
            data_dir = Path(f"data/uploads/{session_id}")
            data_dir.mkdir(parents=True, exist_ok=True)

            with st.status("Ingesting documents into ChromaDB...", expanded=True) as status:
                st.write("📁 Saving uploaded file(s)...")
                saved_paths = []
                for uploaded_file in uploaded_files:
                    file_path = data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(file_path)

                st.write("⚡ Extracting text, cleaning formatting & token chunking...")
                result = ingest_documents(path=data_dir, vector_store=vector_store, reset=False)

                if result.get("status") in ("warning", "skipped"):
                    status.update(label=f"ℹ️ {result.get('message')}", state="complete", expanded=False)
                    st.info(result.get("message"))
                else:
                    status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                    st.session_state.messages = []
                    st.toast(f"Indexed {result.get('files_processed')} file(s) ({result.get('chunks_stored')} chunks stored)!", icon="🎉")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Pipeline Settings")
    top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=10, value=settings.TOP_K)
    debug_mode = st.checkbox("Developer Debug Mode", value=os.getenv("DEBUG_RAG", "false").lower() == "true")

    st.markdown("---")
    if st.button("🗑️ Clear Vector Database", type="secondary", use_container_width=True):
        vector_store.reset()
        st.session_state.messages = []
        st.toast("Vector database cleared!", icon="🧹")
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant_response(
    content: str,
    citations: List[Dict[str, Any]] = None,
    retrieved_chunks: List[Dict[str, Any]] = None,
    debug: bool = False
):
    """Renders clean assistant answer with styled citation badges and debug diagnostics."""
    # Strip redundant plain-text "Sources:\n[1] ..." suffix if present in text content
    clean_text = content
    if "\n\nSources:\n" in clean_text:
        clean_text = clean_text.split("\n\nSources:\n")[0]
    elif "\nSources:\n" in clean_text:
        clean_text = clean_text.split("\nSources:\n")[0]

    # Render main markdown answer
    st.markdown(clean_text)

    # Render verified source badges if citations exist
    if citations:
        badges_html = CitationFormatter.format_citations_badges(citations)
        st.markdown(
            f"""
            <div class="sources-block">
                <div class="sources-header">📌 Verified Sources & Page References</div>
                <div>{badges_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Developer Debug Expander
    if debug and retrieved_chunks:
        with st.expander("🛠️ Developer Diagnostics (Debug Mode Only)"):
            st.json({
                "retrieved_count": len(retrieved_chunks),
                "citations": citations or [],
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "source": c.get("source"),
                        "page_number": c.get("page_number"),
                        "similarity_score": c.get("score"),
                        "text_snippet": c.get("text", "")[:150] + "..."
                    }
                    for c in retrieved_chunks
                ]
            })


# Quick Prompt Suggestions (Displayed when DB has documents & chat is clean)
if vector_store.get_count() > 0 and len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Starter Questions:")
    q_col1, q_col2, q_col3 = st.columns(3)

    with q_col1:
        if st.button("📌 What is this document about?", use_container_width=True):
            st.session_state.preset_query = "What is this document about?"
    with q_col2:
        if st.button("🔍 Summarize main topics", use_container_width=True):
            st.session_state.preset_query = "Summarize the main contents and key topics of this document."
    with q_col3:
        if st.button("📋 What key information is provided?", use_container_width=True):
            st.session_state.preset_query = "What key information and details are provided in this document?"

# Handle preset query button clicks
active_query = None
if "preset_query" in st.session_state and st.session_state.preset_query:
    active_query = st.session_state.preset_query
    st.session_state.preset_query = None

# Empty State Welcome Banner
if vector_store.get_count() == 0 and len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #f8fafc; margin-top:0;">👋 Welcome to Ask My Docs</h4>
        <p style="color: #94a3b8;">The document vector database is currently empty. Follow these steps to get started:</p>
        <ol style="color: #cbd5e1; margin-left: 20px;">
            <li>Use the sidebar on the left to upload your <b>PDF</b> or <b>TXT</b> document.</li>
            <li>Click <b>🚀 Ingest Documents</b> to index the content into ChromaDB.</li>
            <li>Ask any question to receive grounded answers with page citations!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_response(
                content=message.get("content", ""),
                citations=message.get("citations", []),
                retrieved_chunks=message.get("retrieved_chunks", []),
                debug=debug_mode
            )
        else:
            st.markdown(message["content"])

# Process User Question Input (from chat input or preset suggestion button)
user_query = st.chat_input("Ask a detailed question about your documents...") or active_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if vector_store.get_count() == 0:
            warning_msg = "⚠️ The document database is empty. Please upload and ingest a document in the sidebar first."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg, "citations": []})
        else:
            with st.spinner("Retrieving semantic context & synthesizing answer..."):
                retrieved_chunks = retriever.retrieve(query=user_query, top_k=top_k)
                result = generator.generate(query=user_query, retrieved_chunks=retrieved_chunks)

            render_assistant_response(
                content=result.answer,
                citations=result.citations,
                retrieved_chunks=result.retrieved_chunks,
                debug=debug_mode
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "citations": result.citations,
                "retrieved_chunks": result.retrieved_chunks
            })

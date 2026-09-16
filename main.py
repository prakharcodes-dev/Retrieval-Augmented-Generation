import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gradio as gr
from app.config import settings
from app.ingestion import ingest_documents
from app.retrieval import ChromaVectorStore, VectorRetriever
from app.generation import RAGGenerator, CitationFormatter

vector_store = ChromaVectorStore()
retriever = VectorRetriever(vector_store=vector_store)
generator = RAGGenerator()


def process_ingestion(files: List[Any], reset_db: bool) -> tuple[str, str]:
    if not files:
        return "⚠️ Please select at least one PDF or TXT file to ingest.", get_db_stats()

    session_id = f"upload_{int(time.time())}"
    data_dir = Path(f"data/uploads/{session_id}")
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file_obj in files:
        file_path_str = getattr(file_obj, "name", str(file_obj))
        dest_path = data_dir / Path(file_path_str).name
        with open(file_path_str, "rb") as f_in:
            with open(dest_path, "wb") as f_out:
                f_out.write(f_in.read())
        saved_paths.append(dest_path)

    result = ingest_documents(path=data_dir, vector_store=vector_store, reset=reset_db)
    status_msg = f"**Status**: {result.get('status', '').upper()}\n\n{result.get('message', '')}\n\n"
    status_msg += f"- **Files Processed**: {result.get('files_processed', 0)}\n"
    status_msg += f"- **Pages Processed**: {result.get('pages_processed', 0)}\n"
    status_msg += f"- **Chunks Stored**: {result.get('chunks_stored', 0)}\n"
    status_msg += f"- **Total DB Chunks**: {result.get('total_chunks_in_db', 0)}"

    return status_msg, get_db_stats()


def handle_query(query: str, top_k: int, history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
    if not query or not query.strip():
        return history, ""

    if vector_store.get_count() == 0:
        warning_msg = "⚠️ The document vector database is empty. Please upload and ingest a document first."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": warning_msg})
        return history, ""

    retrieved_chunks = retriever.retrieve(query=query, top_k=int(top_k))
    result = generator.generate(query=query, retrieved_chunks=retrieved_chunks)

    clean_answer = result.answer
    if "\n\nSources:\n" in clean_answer:
        clean_answer = clean_answer.split("\n\nSources:\n")[0]
    elif "\nSources:\n" in clean_answer:
        clean_answer = clean_answer.split("\nSources:\n")[0]

    citations_html = CitationFormatter.format_citations_badges(result.citations) if result.citations else ""

    formatted_response = clean_answer
    if citations_html:
        formatted_response += f"\n\n---\n**📌 Verified Sources & References**:\n{citations_html}"

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": formatted_response})
    return history, ""


def handle_clear_db() -> tuple[str, str, List[Dict[str, str]]]:
    vector_store.reset()
    return "🧹 Vector database has been completely reset.", get_db_stats(), []


def get_db_stats() -> str:
    count = vector_store.get_count()
    return f"**Indexed Chunks in Vector DB**: `{count}`"


custom_css = """
.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    margin: 4px 4px 4px 0;
    font-weight: 500;
}
"""

with gr.Blocks(title="Ask My Docs — RAG Document Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 Ask My Docs — RAG Document Assistant
        Upload domain documents (PDF / TXT) to perform semantic vector retrieval and generate grounded answers with source citations.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Document Ingestion Hub")
            file_input = gr.File(
                label="Upload PDF or TXT files",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"]
            )
            reset_checkbox = gr.Checkbox(label="Reset database before ingesting", value=False)
            ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary")
            ingest_status = gr.Markdown()
            db_stats = gr.Markdown(value=get_db_stats())
            clear_db_btn = gr.Button("🗑️ Clear Vector Database", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Interactive Q&A Session")
            chatbot = gr.Chatbot(label="Chat History", height=450)
            query_input = gr.Textbox(
                label="Ask a detailed question about your documents...",
                placeholder="What is this document about?",
                lines=2
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=settings.TOP_K,
                    step=1,
                    label="Top-K Context Chunks"
                )
                submit_btn = gr.Button("Submit Question", variant="primary")

    ingest_btn.click(
        fn=process_ingestion,
        inputs=[file_input, reset_checkbox],
        outputs=[ingest_status, db_stats]
    )

    submit_btn.click(
        fn=handle_query,
        inputs=[query_input, top_k_slider, chatbot],
        outputs=[chatbot, query_input]
    )

    query_input.submit(
        fn=handle_query,
        inputs=[query_input, top_k_slider, chatbot],
        outputs=[chatbot, query_input]
    )

    clear_db_btn.click(
        fn=handle_clear_db,
        inputs=[],
        outputs=[ingest_status, db_stats, chatbot]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

# Ask My Docs — Phase 1 RAG Fundamentals

**Ask My Docs** is a domain-specific Retrieval-Augmented Generation (RAG) assistant designed to ingest PDF and TXT documents, store token-bounded text chunks in ChromaDB, perform semantic vector search, and generate grounded answers with verifiable source citations.

---

## 📖 What is RAG & Why is it Useful?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances Large Language Models (LLMs) by retrieving relevant document context from an external vector database before generating an answer.

### Why RAG is Useful:
1. **Prevents Hallucinations**: Constrains the LLM to answer strictly using provided document context.
2. **Domain-Specific Accuracy**: Enables QA against private corporate policies, technical specifications, and proprietary handbooks without retraining or fine-tuning models.
3. **Traceability**: Provides real source citations (e.g., `[1] refund_policy.pdf — Page 4`) mapping directly to original documents.
4. **Refusal Mechanism**: Safely refuses to answer questions when sufficient document evidence is unavailable rather than fabricating information.

---

## 🏗️ Phase 1 Architecture

```
User Document (.pdf / .txt)
           │
           ▼
   [ Document Loader ] (PDF 1-indexed pages / TXT)
           │
           ▼
    [ Text Cleaner ]   (Fixes unspaced font artifacts & normalizes whitespace)
           │
           ▼
   [ Token Chunker ]   (700 tokens target, 100 overlap, metadata attached)
           │
           ▼
 [ Embedding Service ] (OpenAI / SentenceTransformers / Mock)
           │
           ▼
   [ ChromaDB Store ]  (Persistent vector storage)
           │
           ▼
[ Semantic Retriever ] (Top-K similarity search)
           │
           ▼
  [ RAG Generator ]    (Grounded prompt, strict refusal on missing evidence)
           │
           ▼
 [ Citation Formatter] (Clean user answer + real source citations)
```

> **Note**: Phase 1 implements core basic semantic vector RAG. Advanced features such as BM25, hybrid search, reranking, RAGAS, or CI quality gates are **not** implemented in this phase.

---

## 🔑 Core Features & Components

### 1. Document Ingestion (`app/ingestion/loader.py` & `cleaner.py`)
- Supports **PDF** and **TXT** files.
- Preserves 1-indexed `page_number` for PDF documents.
- `TextCleaner` automatically repairs character stream artifacts (such as missing word spaces from PDF streams like `Inprecovidyears...` -> `In pre covid years...`).

### 2. Token-Aware Chunking (`app/ingestion/chunker.py`)
- Uses `tiktoken` to split text into chunks of approximately **700 tokens** with **100 token overlap**.
- Preserves document hierarchy (headings, paragraphs, sentences).
- Attaches complete metadata (`document_id`, `filename`, `file_type`, `source`, `title`, `page_number`, `chunk_id`, `chunk_index`).
- Includes validation checks (`validate_chunks`) for completeness and uniqueness.

### 3. Embedding Service (`app/embeddings/embedding_service.py`)
- Dedicated `EmbeddingService` abstraction exposing `embed_documents` and `embed_query`.
- Configurable via `config.yaml` or `.env` (`openai`, `sentence-transformers`, or `mock`).

### 4. ChromaDB Vector Store (`app/vectorstore/chroma_store.py`)
- Implements `BaseVectorStore` interface over persistent ChromaDB storage (`./vectorstore`).

### 5. Grounded LLM Generation & Refusal (`app/generation/answer_generator.py`)
- Uses system prompt `prompts/rag_answer_v1.txt`.
- Answers strictly using retrieved context excerpts.
- If context is missing or evidence is insufficient, returns exact refusal string:
  > *"I couldn't find sufficient evidence in the provided documents to answer that question."*

### 6. Real Source Citations (`app/generation/citation_formatter.py`)
- Formats verifiable citations from retrieved metadata:
  ```
  Answer:
  Customers can request a refund within 30 days of purchase.

  Sources:
  [1] refund_policy.pdf — Page 4
  ```
- Prevents fake filenames, fabricated page numbers, or unretrieved sources.

### 7. Clean User Interface & Hidden Chunks
- Standard UI (Streamlit & CLI) shows ONLY the clean answer and formatted sources. Internal retrieval chunks, vector arrays, and similarity scores remain hidden from normal view.

---

## ⚙️ Configuration & Environment

Centralized configuration is defined in `config.yaml` and `.env`:

### `config.yaml`
```yaml
chunking:
  chunk_size: 700
  chunk_overlap: 100

retrieval:
  vector_top_k: 5

vector_store:
  provider: chroma
  persist_dir: ./vectorstore
  collection_name: rag_documents

embedding:
  provider: openai
  model: text-embedding-3-small

llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
```

### Environment Variables (`.env`)
Create a `.env` file based on `.env.example`:
```bash
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-gemini-key-here
DEBUG_RAG=false
```

---

## 🚀 Installation & Setup

1. **Activate Virtual Environment**:
   ```bash
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

### Option 1: Web UI (Streamlit)
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`. Upload PDF/TXT files in the sidebar and ask questions.

### Option 2: CLI Commands (`app/main.py`)

- **Ingest Documents**:
  ```bash
  python app/main.py ingest --path data/
  ```

- **Ask a Question**:
  ```bash
  python app/main.py ask "What is the refund policy?"
  ```

- **Ask with Developer Diagnostics**:
  ```bash
  python app/main.py ask "What is the refund policy?" --verbose
  ```

- **Interactive CLI Session**:
  ```bash
  python app/main.py interactive
  ```

- **Reset Database**:
  ```bash
  python app/main.py reset
  ```

---

## 🧪 Running Unit Tests

Run the complete Phase 1 pytest suite:
```bash
pytest -v
```

Tests cover document ingestion, text cleaning, token chunking, vector retrieval, grounded generation, strict refusal, and citation formatting.

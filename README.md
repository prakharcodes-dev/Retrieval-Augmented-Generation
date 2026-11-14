---
title: RAG Document Assistant
emoji: 🚀
colorFrom: pink
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: streamlit_app.py
pinned: false
license: mit
short_description: Production-Ready RAG System with Hybrid Search, Benchmarking, Performance Tracking, and Document Validation.
---

# 📚 RAG Document Assistant

**RAG Document Assistant** is a high-capacity, production-grade Retrieval-Augmented Generation (RAG) platform built with Python, PyMuPDF, ChromaDB, SQLite BM25, and Streamlit/Gradio. It combines dense semantic search and sparse lexical search via Reciprocal Rank Fusion (RRF), page-batched streaming ingestion, strict document validation, an automated RAG evaluation harness, retrieval method comparison, real-time performance tracking, and citation verification.

---

## 🌟 Upgraded Practical Features

The platform provides 5 integrated core capabilities:

1. **RAG Evaluation and Testing System**:
   - Reusable benchmark dataset with 25 diverse test cases covering factual, numerical, comparison, definition, overview, and negative control queries.
   - Comprehensive retrieval metrics: **Precision@K**, **Recall@K**, **Hit Rate**, **Mean Reciprocal Rank (MRR)**, and **Context Precision**.
   - Answer-quality evaluation: **Faithfulness** (context grounding) and **Answer Relevancy**.
   - Downloadable reports in standard **JSON** and **CSV** formats.
   - Operable both via the **Streamlit Web UI** and the command-line interface (`python evaluate.py`).

2. **Retrieval Method Comparison**:
   - Side-by-side benchmarking of **Vector Search (Dense)**, **BM25 Search (Sparse)**, and **Hybrid Search (Dense + Sparse RRF)** across identical test queries.
   - Empirical ranking of methods with automated identification of the best overall, lowest latency, and highest hit-rate algorithms.
   - Comparative accuracy and latency bar charts in the Streamlit UI with exportable comparison reports.

3. **Improved PDF Upload and Document Validation**:
   - Path-traversal sanitization and safe filename normalization.
   - Deep structural validation: checks for standard `%PDF-` header magic bytes, empty/whitespace files, corrupted xref tables, and broken streams.
   - Pre-flight file size enforcement against configured limits (`settings.MAX_FILE_SIZE_MB`).
   - Early detection of password-protected/encrypted documents.
   - Scanned/image-only PDF detection with clear user feedback.
   - SHA-256 hash deduplication preventing redundant processing.
   - Safe temporary staging and guaranteed file descriptor cleanup.

4. **Retrieval Performance Tracking**:
   - Accurate microsecond timing (`time.perf_counter`) of document ingestion, retrieval, reranking/fusion, generation, and total end-to-end question answering.
   - Tracking partitioned by retrieval modality (Dense, Sparse, Hybrid).
   - Real-time performance dashboard with average, P50, and P95 latency percentiles.
   - Ingestion throughput metrics (chunks/sec and pages/sec).
   - Recent query traces log without leaking sensitive document contents or API credentials.

5. **Production Reliability**:
   - Centralized memory guard (`ResourceGuard`) monitoring physical RAM (`psutil`) to prevent OS-level OOM crashes.
   - Grounded universal prompt enforcing strict context adherence and standardized refusal for unsupported queries.
   - Verifiable citations with source document name and page number references.

---

## 🛠️ Technology Stack Actually Used

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **User Interfaces** | Streamlit, Gradio | Interactive Web UIs for document Q&A, evaluation, and latency monitoring |
| **Dense Vector Store** | ChromaDB (SQLite + HNSW) | Persistent embedding storage and nearest-neighbor vector search |
| **Sparse Lexical Index** | SQLite BM25 (`BM25Index`) | Deterministic BM25 keyword search synchronized with Chroma chunk IDs |
| **Retrieval Fusion** | Reciprocal Rank Fusion (RRF) | Rank-based score aggregation: `1 / (rrf_k + rank)` |
| **Document Processing** | PyMuPDF (fitz), PyPDF | Safe multi-page PDF extraction and streaming page slicing |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) / Mock | Dense vector representation generation |
| **LLM Generation** | Google Gemini API (`gemini-3.8-flash`), Groq (`qwen3.8-27b`), Qwen LoRA | Context-grounded response generation |
| **Evaluation Suite** | Custom Metric Engine, Pandas | Precision@K, Recall@K, Hit Rate, MRR, Context Precision, Faithfulness |
| **Testing** | Pytest, ReportLab | Comprehensive automated unit test suite (84 tests) |
| **Resource Guard** | `psutil`, `threading` | System RAM monitoring, garbage collection, and lazy-loading locks |

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Document Ingestion Flow                         │
└────────────────────────────────────────────────────────────────────────┘
  Uploaded Document (PDF / TXT / MD)
          │
          ▼
  [ DocumentValidator ]  ──► (Sanitize filename, check size, %PDF- header,
          │                   encryption, empty text, SHA-256 deduplication)
          ▼
  [ Page-Batched Streaming Loop ] (Safe context manager handles)
          ├──► Text Cleaning (cleaner.py: unspaced text, whitespace)
          ├──► Token Chunking (chunker.py: 700 tokens, 100 overlap)
          ├──► Vector Embeddings (Gemini Embeddings / Mock)
          ├──► Incremental Chroma Persistence (chroma.sqlite3 + HNSW)
          ├──► Synchronized SQLite BM25 Upsert (bm25.sqlite3)
          └──► ResourceGuard Garbage Collection (RAM safety)

┌────────────────────────────────────────────────────────────────────────┐
│                        Query & Retrieval Flow                          │
└────────────────────────────────────────────────────────────────────────┘
  User Query / Evaluation Sample
          │
          ├──────────────────────────┬─────────────────────────┐
          ▼                          ▼                         ▼
   [ Vector Search ]          [ BM25 Search ]          [ Hybrid RRF ]
   Dense cosine/L2 match      Sparse BM25 tokens       Reciprocal Rank Fusion
   (Top-K candidates)         (Top-K candidates)       1 / (rrf_k + rank)
          │                          │                         │
          └──────────────────────────┴─────────────────────────┘
                                     │
                                     ▼
                            [ Top-K Ranked Context ]
                                     │
                                     ▼
                          [ Universal RAG Prompt ]
                                     │
                                     ▼
                       [ Gemini 3.8 Flash Generation ]
                         (Groq fallback if configured)
                                     │
                                     ▼
                         [ Cited Answer + Sources ]
                                     │
                                     ▼
                    [ Performance Tracker & Dashboard ]
```

---

## 📁 File Structure & Storage Locations

```text
d:\RAG\
├── app/
│   ├── config/                     # Application configuration & YAML settings
│   │   └── settings.py
│   ├── core/                       # Core safety, exceptions, & performance tracking
│   │   ├── exceptions.py           # Subsystem exception hierarchy
│   │   ├── performance_tracker.py  # Latency, percentiles, throughput tracking
│   │   ├── resource_guard.py       # RAM/VRAM resource monitor
│   │   └── state.py                # Application state machine
│   ├── embeddings/                 # Dense embedding providers (Gemini, OpenAI, Mock)
│   ├── evaluation/                 # RAG evaluation & comparison system
│   │   ├── comparator.py           # 3-way retrieval comparator (Vector vs BM25 vs Hybrid)
│   │   ├── dataset.py              # EvaluationDataset & EvaluationSample schema
│   │   ├── evaluator.py            # RAGEvaluator engine & EvaluationReport
│   │   └── metrics.py              # Precision@K, Recall@K, Hit Rate, MRR, Faithfulness
│   ├── generation/                 # LLM generation & citation formatters
│   ├── ingestion/                  # Document ingestion & validation
│   │   ├── chunker.py              # Deterministic token chunker
│   │   ├── cleaner.py              # Text cleaning & spacing repairs
│   │   ├── ingest.py               # Streaming page-batch ingestion loop
│   │   ├── loader.py               # Safe PDF/TXT document loaders
│   │   └── validator.py            # Upload validator, sanitization & corruption checks
│   ├── retrieval/                  # Retrieval implementations
│   │   ├── bm25_index.py           # SQLite BM25 sparse index
│   │   └── vector_retriever.py     # Hybrid RRF retriever
│   ├── security/                   # Password gate access control
│   └── vectorstore/                # ChromaDB vector store wrapper
│
├── data/
│   ├── documents/                  # Reference documents (e.g. sample_travel_guide.pdf)
│   └── evaluation/                 # Benchmark evaluation datasets
│       └── default_eval_dataset.json # 25 benchmark test cases
│
├── vectorstore/                    # Persistent vector database & BM25 database
│   ├── chroma.sqlite3              # Chroma vector metadata & document text
│   └── bm25.sqlite3                # Synchronized SQLite BM25 term index
│
├── prompts/                        # Strict RAG system prompts
│   ├── rag_answer_v1.txt
│   └── rag_prompt.txt
│
├── tests/                          # 84 automated pytest tests
│   ├── test_access_control.py
│   ├── test_bm25_hybrid.py
│   ├── test_chunking.py
│   ├── test_citations.py
│   ├── test_document_validation.py # Upload & corruption validation tests
│   ├── test_embedding_configuration.py
│   ├── test_evaluation_dataset.py  # Dataset schema & loader tests
│   ├── test_evaluation_metrics.py  # Precision, Recall, Hit Rate, MRR tests
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_performance_tracking.py# Latency & throughput tests
│   ├── test_reliability_and_capacity.py
│   ├── test_retrieval.py
│   └── test_retrieval_comparison.py# 3-way retrieval comparison tests
│
├── evaluate.py                     # CLI tool for RAG evaluation & comparison
├── ingest.py                       # CLI tool for document ingestion
├── app.py                          # CLI interactive Q&A assistant
├── main.py                         # Gradio Web UI entrypoint
├── streamlit_app.py                # Streamlit Web UI entrypoint (4 tabs)
├── config.yaml                     # Central configuration parameters
├── requirements.txt                # Package dependencies
└── .env.example                    # Environment variable template
```

---

## 🔍 Retrieval Methods Explained

### 1. Vector Search (Dense Semantic Retrieval)
- **Mechanism**: Encodes user queries into dense numerical vectors using Gemini Embeddings (`gemini-embedding-001`, 768 dimensions) and computes nearest-neighbor cosine/L2 distance across ChromaDB HNSW indexes.
- **Strengths**: Captures conceptual meaning, synonyms, paraphrasing, and thematic relevance (e.g., matching "where to stay" with "accommodation guides").
- **Limitations**: May struggle with exact alphanumeric codes, technical acronyms, or rare entity names absent from embedding vocabulary.

### 2. BM25 Search (Sparse Lexical Retrieval)
- **Mechanism**: Tokenizes query terms and scores indexed chunks in SQLite using the Okapi BM25 ranking function ($k_1=1.5$, $b=0.75$), calculating term frequency and inverse document frequency (IDF).
- **Strengths**: Excels at precise keyword matching, exact entity lookups (e.g., "Tirumala Nayak Palace"), and numeric queries.
- **Limitations**: Ineffective when queries use synonyms or phrasing different from the source document.

### 3. Hybrid Search (Reciprocal Rank Fusion)
- **Mechanism**: Executes dense and sparse queries concurrently, extracting independent candidate sets. Fuses candidates using Reciprocal Rank Fusion (RRF):
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k_{\text{rrf}} + \text{rank}_m(d)}$$
  where $k_{\text{rrf}} = 60$.
- **Strengths**: Combines semantic breadth with lexical precision. Eliminates single-method blind spots and yields the highest overall Hit Rate and MRR.

---

## 📊 RAG Evaluation Methodology & Metrics

The evaluation suite tests retrieval and answer generation using ground truth test datasets (`EvaluationDataset`).

### Benchmark Dataset Schema
Each sample contains:
- `id`: Unique identifier (e.g., `eval_001`).
- `question`: User query string.
- `expected_answer`: Ground truth answer text.
- `relevant_docs`: Target source document filenames (e.g., `sample_travel_guide.pdf`).
- `expected_context_snippets`: Specific text passages that must be retrieved for relevance.
- `category`: Query intent (`factual`, `numerical`, `comparison`, `definition`, `overview`, `negative_control`).

### Retrieval Metrics

1. **Precision@K**:
   $$\text{Precision}@K = \frac{\text{Relevant Chunks in Top } K}{K}$$
   Measures context efficiency and noise filtering.

2. **Recall@K**:
   $$\text{Recall}@K = \frac{\text{Relevant Target Units Found in Top } K}{\text{Total Expected Relevant Units}}$$
   Measures completeness of context retrieval.

3. **Hit Rate (@K)**:
   $$\text{Hit Rate} = \begin{cases} 1.0 & \text{if at least 1 relevant chunk is in Top } K \\ 0.0 & \text{otherwise} \end{cases}$$
   Measures how reliably the user receives at least one useful reference.

4. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{Rank of First Relevant Chunk}}$$
   Measures how close to the top the most relevant chunk is positioned.

5. **Context Precision**:
   $$\text{Context Precision} = \frac{\sum_{i=1}^K (\text{Precision}@i \times \text{Rel}(i))}{\text{Total Relevant Chunks in Top } K}$$
   Rewards systems that place the most relevant chunks in early positions within the context window.

### Answer Quality Metrics

1. **Faithfulness**:
   Evaluates whether claims and sentences in the generated answer are grounded in the retrieved context, preventing hallucination.
2. **Answer Relevancy**:
   Measures token overlap and semantic alignment between the generated response and the original user question or ground truth.

---

## 🚀 How to Run the Application & Features

### 1. Web UI (Streamlit - Recommended)
```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

The sidebar includes a **Clear Vector Database** button. One click immediately removes all documents from the active Chroma collection and SQLite BM25 index. It is available to every app visitor, so anyone with access to the public app can erase the shared index. Back up the `vectorstore/` directory before using it.

#### Available UI Tabs:
- **💬 Chat Assistant**: Natural document conversation with citation expanders, developer chunk inspection, and real-time latency badges.
- **🔬 RAG Evaluation**: Select test category, adjust Top-K, optionally enable answer quality evaluation, run benchmarks, view per-query tables, and download JSON/CSV reports.
- **⚖️ Retrieval Comparison**: Compare Vector Search vs BM25 Search vs Hybrid Search with accuracy and latency charts.
- **⚡ Performance Monitor**: View live latency summary cards (Avg, P50, P95), method breakdowns, ingestion throughput, and recent query traces.

---

### 2. Command-Line Evaluation (`evaluate.py`)

Run default RAG retrieval evaluation:
```powershell
.venv\Scripts\python.exe evaluate.py
```

Run 3-way retrieval method comparison:
```powershell
.venv\Scripts\python.exe evaluate.py --compare --top-k 5
```

Run full evaluation including answer generation quality (Faithfulness & Relevancy):
```powershell
.venv\Scripts\python.exe evaluate.py --generate --top-k 5
```

Export evaluation reports to JSON and CSV:
```powershell
.venv\Scripts\python.exe evaluate.py --compare --output report.json --csv report.csv
```

---

### 3. Document Ingestion CLI (`ingest.py`)
```powershell
.venv\Scripts\python.exe ingest.py --path data/documents/sample_travel_guide.pdf
```

---

### 4. Interactive CLI Q&A (`app.py`)
```powershell
.venv\Scripts\python.exe app.py ask "What historical monuments are covered?"
```

---

## 🧪 Running Automated Tests

Execute the complete 84-test test suite:
```powershell
.venv\Scripts\python.exe -m pytest
```

Execute specific feature test suites:
```powershell
# Evaluation metrics & dataset tests
.venv\Scripts\python.exe -m pytest tests/test_evaluation_metrics.py tests/test_evaluation_dataset.py

# 3-way retrieval comparison tests
.venv\Scripts\python.exe -m pytest tests/test_retrieval_comparison.py

# PDF validation & security tests
.venv\Scripts\python.exe -m pytest tests/test_document_validation.py

# Performance tracking tests
.venv\Scripts\python.exe -m pytest tests/test_performance_tracking.py
```

---

## 🔑 Environment Variables & Streamlit Cloud Secrets

Configure variables in `.env` (local) or in **Streamlit Community Cloud → Settings → Secrets**:

```toml
# Answer Generation API Keys
GEMINI_API_KEY = "your-gemini-api-key"
GROQ_API_KEY = "your-groq-api-key-for-fallback"
OPENAI_API_KEY = "your-openai-api-key"

# Retrieval & Provider Overrides (Optional)
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.8-flash"
GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_PROVIDER = "auto"
```

> [!CAUTION]
> Never commit actual API keys or secret passwords to GitHub. Use `.env.example` as a template.

---

## ⚠️ Known Limitations & Best Practices

1. **Scanned / Image-Only PDFs**: Text extraction relies on PyMuPDF and PyPDF digital text streams. Non-OCR scanned images contain no extractable text; the system detects this and warns the user.
2. **Shared Password Gate**: The application uses a shared gate password for access control; it does not provide individual multi-tenant user isolation.
3. **Local Embedding Vector Dimension**: The vector store index is tied to the embedding model's dimensionality. When changing embedding providers, back up `vectorstore/` and run `python migrate_embeddings.py`.
4. **Offline Evaluation**: The default evaluation mode runs fast, offline, and deterministically without incurring API tokens or external costs. Answer generation evaluation can be optionally toggled when credentials are configured.

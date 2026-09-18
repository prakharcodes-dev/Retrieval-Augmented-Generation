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
short_description: Production-Ready RAG System with Page-Batched Streaming, Memory Guards, Lazy LLM, and Citation Verification.
---

# 📚 RAG Document Assistant

**RAG Document Assistant** is a high-capacity, production-ready Retrieval-Augmented Generation (RAG) system built with Python, PyMuPDF, ChromaDB, and Streamlit/Gradio. It features page-batched streaming ingestion, centralized memory guarding (`ResourceGuard`), deterministic chunking, lazy LLM loading, and verifiable citations.

---

## 📁 File Storage Locations & Directory Structure

Here is where files and datasets are stored across the project:

```text
d:\RAG\
├── app/                           # Core Application Source Code
│   ├── config/                    # Configuration settings & YAML parsers
│   ├── core/                      # Core reliability, state tracking & resource guard
│   │   ├── exceptions.py          # Subsystem exception hierarchy
│   │   ├── resource_guard.py      # RAM/VRAM/Workload monitor (psutil/cuda)
│   │   └── state.py               # Application state machine (READY, INGESTING, etc.)
│   ├── embeddings/                # Vector embedding services (OpenAI/SentenceTransformers/Mock)
│   ├── generation/                # Answer generator & citation formatters (Qwen+LoRA / CPU / Mock)
│   ├── ingestion/                 # Document loaders, cleaners, token chunkers & batched ingest
│   ├── retrieval/                 # Hybrid vector retriever & keyword scorers
│   └── vectorstore/               # ChromaDB store & targeted hash lookup
│
├── data/                          # Physical Document & Upload Storage
│   ├── documents/                 # Sample documents directory (e.g. sample_travel_guide.pdf)
│   └── uploads/                   # Temporary upload directories for web/UI files
│
├── vectorstore/                   # Persistent Vector Database (ChromaDB SQLite & HNSW files)
│   ├── chroma.sqlite3             # Main SQLite database storing metadata & document text
│   └── <uuid-collections>/        # HNSW vector index binary files
│
├── prompts/                       # System Prompts
│   ├── rag_answer_v1.txt          # Default strict context Q&A system prompt
│   └── rag_prompt.txt             # Alternative prompt template
│
├── tests/                         # Pytest Test Suite
│   ├── test_chunking.py           # Token chunking & metadata validation tests
│   ├── test_citations.py          # Citation extraction & formatting tests
│   ├── test_generation.py         # LLM answer generation & refusal tests
│   ├── test_ingestion.py          # Document loader & deduplication tests
│   ├── test_reliability_and_capacity.py # Batched ingestion, memory guard & capacity tests
│   └── test_retrieval.py          # Vector store & retriever tests
│
├── .env                           # Local environment variables & API keys
├── config.yaml                    # Centralized system parameters
├── app.py                         # CLI application entrypoint
├── ingest.py                      # Dedicated CLI document ingestion tool
├── main.py                        # Gradio Web UI entrypoint
├── streamlit_app.py               # Streamlit Web UI entrypoint
└── requirements.txt               # Python package dependencies
```

### Storage Summary:
1. **Uploaded Files**: Streamlit/Gradio uploads are written to temporary operating system buffers or `data/uploads/` and processed immediately.
2. **Persistent Vector Store**: Chunks, document text, and vector embeddings are persisted locally in `vectorstore/` (ChromaDB).
3. **Temporary Processing Files**: Handled strictly via deterministic context managers (`safe_pymupdf_open` / `safe_pypdf_open`) and cleaned up immediately after processing.

---

## 🛠️ User Interface Guide & Sidebar Metrics

### 1. Free RAM (MB) Metric
- **What it displays**: Real-time available system physical memory (RAM) in Megabytes.
- **Source**: Queried directly from OS kernel APIs via `psutil.virtual_memory().available`.
- **Purpose**: Used by `ResourceGuard` to verify safety before performing heavy ingestion, embedding, or LLM inference. If free RAM drops below 300 MB, the system halts operations safely to prevent OS-level crashes.

### 2. Top-K Retrieved Chunks Slider
- **What it does**: Controls how many of the top matching passages are retrieved from ChromaDB to construct each answer.
- **Storage vs. Retrieval**: Storage capacity is unbounded (thousands of chunks can be stored in the vector database). Top-K only controls retrieval context size per query.

#### Quick Top-K Decision Cheat Sheet:
| Question Type | Recommended Top-K | Example Questions | Why |
| :--- | :---: | :--- | :--- |
| **Fact / Pinpoint Detail** | **3 – 5** *(Default)* | *"What is the check-in time?"*<br>*"What is the contact email?"* | Fast, precise, focused answer from a single snippet. |
| **Lists & Procedures** | **6 – 10** | *"What are all the rules for refunds?"*<br>*"List the steps to register."* | Captures steps scattered across 2–4 pages. |
| **Comparisons & Deep Search** | **10 – 15** | *"Compare Plan A vs Plan B."*<br>*"Find all mentions of risk factors."* | Gathers details from different chapters/sections. |
| **Full Document Overview** | **15 – 20** | *"Give me a complete summary of this file."* | Provides maximum broad context across the entire PDF. |

---

## 🏗️ Production Reliability Architecture

```text
PDF Document
     │
     ▼
[ ResourceGuard & PDF Validation ] ──► (Lightweight integrity check)
     │
     ▼
[ Page-Batched Streaming Loop ]    ──► (Default: 20-50 pages per slice)
     │
     ├──► Text Cleaning (cleaner.py)
     ├──► Token Chunking & Sequential Indexing (chunker.py: 700 tokens, 100 overlap)
     ├──► Vector Embedding (embedding_service.py)
     ├──► Incremental Chroma Persist (chroma_store.py)
     └──► Memory Release & GC (ResourceGuard.collect_garbage())
     │
     ▼
[ Vector Store Ready ]             ──► (Targeted hash queries prevent RAM bloat)
     │
     ▼
[ User Query ]                     ──► (Hybrid Vector + Keyword Retrieval)
     │
     ▼
[ Lazy LLM Loading & Generation ]  ──► (Qwen 2.5 3B + LoRA / CPU fallback)
```

---

## 🔑 Key Features & Crash Prevention

1. **Page-Batched Streaming Ingestion**: Removes artificial page/file size caps (like the old 150-page limit). Documents of any page length are processed in memory-safe streaming page slices.
2. **Centralized Resource Guard**: Before every heavy memory step, `ResourceGuard` checks free system RAM (`psutil`) and process memory, raising `ResourceUnsafeError` before memory is exhausted.
3. **Safe PDF Handle Lifecycle**: PyMuPDF (`pymupdf.open`) and PyPDF (`pypdf.PdfReader`) handles are managed by context managers (`safe_pymupdf_open`) to prevent `ValueError: document closed` errors.
4. **Targeted Metadata Queries**: `has_file_hash()` uses targeted Chroma queries (`collection.get(where={"file_hash": file_hash})`) rather than loading entire DB metadatas into memory.
5. **Lazy LLM Isolation**: Model weights are loaded only on the first user query (not at app startup or PDF upload), guarded by thread locks and VRAM/RAM checks.
6. **Streamlit Rerun Safety**: State tracking (`OperationState`) and hash deduplication prevent duplicate file ingestions or duplicate model allocations on Streamlit reruns.

---

## ⚙️ Configuration (`config.yaml`)

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
  provider: qwen_lora
  base_model: Qwen/Qwen2.5-3B-Instruct
  adapter_path: D:\Training\trained_model
  temperature: 0.0
```

---

## 🚀 How to Run the Application

### 1. Web UI (Streamlit - Recommended)
```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
*Or using the virtualenv streamlit binary:*
```powershell
.venv\Scripts\streamlit.exe run streamlit_app.py
```

### 2. CLI Document Ingestion
```powershell
.venv\Scripts\python.exe ingest.py --path data/documents/sample_travel_guide.pdf --reset
```

### 3. CLI Interactive Q&A
```powershell
.venv\Scripts\python.exe app.py ask "What is this document about?"
```

### 4. Gradio Web UI
```powershell
.venv\Scripts\python.exe main.py
```

---

## 🧪 Running Unit Tests

Run the complete 32-test pytest suite:
```powershell
.venv\Scripts\python.exe -m pytest
```

All 32 tests (covering document loading, cleaning, chunking, embedding, vector retrieval, answer generation, citations, memory guards, and high chunk capacity) will execute and pass.

# 🚀 SmartChunk — Project Progress & Architecture Report

**Status:** ✅ Production-Ready · Live Server & Browser Verified · 100% Offline Capable  
**Date:** 2026-10-28  
**Workspace:** `c:\Users\user\OneDrive\New folder\OneDrive\Desktop\SmartChunk`  

---

## 📌 Executive Summary

SmartChunk is a production-grade, 100% offline-capable, structure-aware, and self-healing RAG (Retrieval-Augmented Generation) pipeline. It eliminates chunk-boundary mutilation (e.g. broken tables, split code blocks, severed headings) and implements an autonomous multi-agent diagnostic & repair loop.

All core layers (Backend, Agents, Chunkers, Search Engine, Ingestion, and Modern Glassmorphic Frontend) are fully implemented. All Python dependencies (`torch`, `sentence-transformers`, `chromadb`, `fastapi`, `rank-bm25`, `pypdf`, `beautifulsoup4`, `langgraph`) are installed and verified.

---

## 📁 Directory & File Architecture (48+ Files)

```
SmartChunk/
├── PROGRESS.md                    # This progress tracker & handover guide
├── requirements.txt               # All core Python dependencies
├── run.bat                        # 1-Click portable launcher script
│
├── samples/                       # Test datasets for chunking & self-healing evaluation
│   ├── sample_architecture.md     # Rich doc with nested headings, code blocks, tables & callouts
│   └── sample_faq.md              # Compact FAQ doc with structured Q&A pairs
│
├── backend/
│   ├── __init__.py
│   ├── config.py                  # Global settings, paths (outputs/models, outputs/chroma_db), thresholds
│   ├── main.py                    # FastAPI app, modern lifespan handler, static mount, /api/ingest
│   ├── models.py                  # Pydantic schemas (Chunks, Diagnostics, AgentState, SearchResults)
│   │
│   ├── agents/                    # Multi-Agent Self-Healing & Q&A Workflow
│   │   ├── __init__.py
│   │   ├── state.py               # TypedDict state schemas for Agent graphs
│   │   ├── graph.py               # LangGraph compiled state machines + sequential fallback orchestrator
│   │   ├── query_rewriter.py      # HyDE & keyword query expansion
│   │   ├── retriever.py           # Hybrid retrieval node
│   │   ├── reviewer.py            # Context relevance & coverage grader
│   │   ├── diagnoser.py           # Autonomous root-cause classifier
│   │   ├── fixer.py               # Targeted chunk healing & re-embedding agent
│   │   ├── evaluator.py           # Pre/post healing score validator
│   │   ├── answerer.py            # Synthesizer with inline markdown citations
│   │   ├── explainer.py           # Diagnostic reasoning trace generator
│   │   └── guardian.py            # Hallucination guardrail & faithfulness checker
│   │
│   ├── chunking/                  # Chunking Engine
│   │   ├── __init__.py
│   │   ├── naive.py               # Fixed-size RecursiveCharacterTextSplitter (500 chars, 50 overlap)
│   │   ├── structure_aware.py     # Markdown AST block parser: atomic tables, atomic code blocks, heading trails
│   │   ├── parent_child.py        # Hierarchical parent-child splitter (Small-to-Big retrieval)
│   │   └── stats.py               # Boundary mutilation & statistical comparative analytics
│   │
│   ├── healing/                   # Diagnostic & Self-Healing Engine
│   │   ├── __init__.py
│   │   ├── diagnostics.py         # Table integrity, code syntax, sentence boundary & heading orphan checks
│   │   ├── repair.py              # Automated stitchers, re-parsers, and contextual repair actions
│   │   └── regression.py          # Regression testing test suite on repaired chunks
│   │
│   ├── ingestion/                 # Document Loading & Offline Embeddings
│   │   ├── __init__.py
│   │   ├── loader.py              # Parsers for Markdown, PDF (pypdf), HTML (BeautifulSoup), and TXT
│   │   └── embeddings.py          # Local SentenceTransformers (all-MiniLM-L6-v2) cached in outputs/models/
│   │
│   ├── search/                    # Dual-Index Search Engine
│   │   ├── __init__.py
│   │   ├── vector_store.py        # ChromaDB PersistentClient vector database (outputs/chroma_db)
│   │   ├── bm25.py                # Rank-BM25 tokenized lexical inverted index
│   │   └── hybrid.py              # Reciprocal Rank Fusion (RRF) combining vector + sparse BM25
│   │
│   ├── llm/                       # Modular LLM Provider Layer
│   │   ├── __init__.py
│   │   └── client.py              # Fallback-aware LLM client (Local/Ollama, Open-source, or Rule-based Mock)
│   │
│   └── routes/                    # FastAPI REST API Endpoints
│       ├── __init__.py
│       ├── chunk.py               # POST /api/chunk (Naive vs Structure-aware dual processing)
│       ├── loop.py                # POST /api/loop/run (Autonomous self-healing loop execution)
│       ├── run_status.py          # GET /api/loop/status (Real-time progress polling)
│       ├── review.py              # POST /api/review (Human-in-the-loop approval & override)
│       ├── ask.py                 # POST /api/ask (Multi-agent RAG Q&A with citations)
│       └── agents_route.py        # GET /api/agents/graph (Workflow topology inspection)
│
└── frontend/                      # Premium Dark-Mode Glassmorphism Dashboard
    ├── index.html                 # 4-Tab interface: Studio, Boundary Inspector, Diagnostics, Ask KB
    ├── style.css                  # Custom styling (HSL tokens, glassmorphism, responsive grid, glow effects)
    └── app.js                     # Tab routing, file upload, chunk diffing, heatmaps, live chat & trace tree
```

---

## ⚡ What Has Been Built & Tested

| Module | Features Completed | Status |
| :--- | :--- | :---: |
| **Python Environment** | Dependencies installed (`fastapi`, `uvicorn`, `chromadb`, `sentence-transformers`, `torch`, `rank-bm25`, `pypdf`, `beautifulsoup4`, `langgraph`, `pydantic`); verified working | ✅ Verified |
| **ChromaDB Integration** | Modernized `VectorStore` to support `chromadb.PersistentClient` with backward compatibility fallback | ✅ Verified |
| **FastAPI Modernization** | Migrated deprecated startup/shutdown events in `backend/main.py` to modern `lifespan` context manager | ✅ Complete |
| **Agent Resilience** | `backend/agents/graph.py` equipped with compiled LangGraph state machines + sequential execution fallback | ✅ Complete |
| **Document Loaders** | Support for `.md`, `.pdf`, `.html`, `.txt` with text normalisation | ✅ Complete |
| **Naive Chunking** | Fixed-size splitting with configurable chunk size and overlap | ✅ Complete |
| **Structure-Aware Chunking** | Markdown block parser keeping tables & code blocks intact; hierarchical `# Heading` breadcrumbs | ✅ Complete |
| **Parent-Child Chunking** | Large parent containers + small searchable child chunks linked by ID | ✅ Complete |
| **Statistical Analytics** | Table breakage count, code mutilation count, size variance, heading retention rate | ✅ Complete |
| **Diagnostic Engine** | 4-point automated flaw detection: split tables, truncated code, orphaned headings, cut sentences | ✅ Complete |
| **Self-Healing Loop** | Dynamic chunk re-merging, AST re-parsing, parent context expansion | ✅ Complete |
| **Dual Search Index** | ChromaDB vector search + BM25 sparse search combined via Reciprocal Rank Fusion (RRF) | ✅ Complete |
| **Multi-Agent RAG Graph** | LangGraph workflow: Rewriter ➔ Retriever ➔ Reviewer ➔ Diagnoser ➔ Fixer ➔ Answerer ➔ Guardian | ✅ Complete |
| **Sample Test Suite** | Created `sample_architecture.md` and `sample_faq.md` in `samples/` for end-to-end testing | ✅ Complete |
| **Frontend UI** | Dark glassmorphic design system, tab switcher, dual comparison viewer, boundary heatmap, chat UI | ✅ Complete |
| **1-Click Launch Script** | `run.bat` upgraded to portable relative paths (`%~dp0`), auto-detects dependencies and launches server | ✅ Complete |
| **End-to-End API Tests** | Validated `GET /`, `POST /api/chunk`, `POST /api/ingest`, `POST /ask`, `POST /loop`, `GET /run/{id}/status` | ✅ Verified |
| **Browser UI Validation** | Headless browser execution verified tab switching, file dropzone, stats pills, heatmap & chat responses | ✅ Verified |
| **Minimalist Dark Theme** | Black & Dark Blue palette (`#06080f` / `#0c101c`), zero animations, no emojis, clean text avatars, removed footer & v1.0 badge | ✅ Complete |

---

## 🔄 Recent Updates & Fixes

1. **ChromaDB Constructor Modernization** (`backend/search/vector_store.py`):
   - Migrated from legacy `chroma_db_impl="duckdb+parquet"` to `chromadb.PersistentClient(path=str(CHROMA_DIR))`.
   - Added automated fallback handling to prevent crashes across different ChromaDB releases.

2. **FastAPI Lifespan Implementation** (`backend/main.py`):
   - Replaced deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` with FastAPI 0.104+ `@asynccontextmanager` `lifespan` pattern.

3. **LangGraph Agent Workflow Resilience** (`backend/agents/graph.py`):
   - Built a sequential execution fallback for `ask_pipeline` and `healing_loop` to ensure zero-downtime execution even in environments where graph compilation needs schema adjustments.

4. **Environment & Dependency Verification**:
   - Added `pydantic>=2.0.0` to `requirements.txt`.
   - Ran `pip install -r requirements.txt` and verified all packages import cleanly (`ALL IMPORTS OK`).

5. **Sample Datasets Added** (`samples/`):
   - `samples/sample_architecture.md`: A 9.5KB markdown document with complex tables, code blocks, bullet lists, and nested headings.
   - `samples/sample_faq.md`: A structured FAQ markdown document for testing small-to-big retrieval and heading retention.

6. **Script Portability** (`run.bat`):
   - Removed hardcoded local paths and standardized on relative paths so `run.bat` can run from any folder on any machine.

---

## 🛠️ How to Run

### Option A: 1-Click Launch (Recommended)
Double-click `run.bat` or run from terminal:
```cmd
cd "c:\Users\user\OneDrive\New folder\OneDrive\Desktop\SmartChunk"
run.bat
```
This script will:
1. Validate Python 3.10+ installation.
2. Confirm all dependencies are present.
3. Cache/verify the offline embedding model (`sentence-transformers/all-MiniLM-L6-v2`) in `outputs/models/`.
4. Launch the Uvicorn web server at `http://127.0.0.1:8000`.
5. Automatically open the SmartChunk dashboard in your default browser.

### Option B: Manual Launch
```cmd
cd "c:\Users\user\OneDrive\New folder\OneDrive\Desktop\SmartChunk"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🎯 Next Steps

1. **Start the Web App**: Run `run.bat` or execute the manual uvicorn command above.
2. **Upload Samples**: Upload `samples/sample_architecture.md` via the **Chunking Studio** tab to inspect Naive vs Structure-Aware chunking side-by-side.
3. **Execute Self-Healing**: Navigate to the **Diagnostics & Healing** tab to observe automated detection of mutilated boundaries and their repair traces.
4. **Test RAG Q&A**: In the **Ask Knowledge Base** tab, query terms like `"How does the self-healing loop detect broken tables?"` to see reciprocal rank fusion retrieval and grounded generation with citations.

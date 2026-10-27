# SmartChunk Architecture Guide

SmartChunk is a production-grade, offline-capable, structure-aware, self-healing RAG pipeline. This document describes its internal architecture, components, and design decisions.

## Pipeline Overview

The system ingests documents through a multi-stage pipeline:
1. **Document Loading** — Reads raw files (Markdown, PDF, HTML, TXT)
2. **Structure-Aware Chunking** — Parses Markdown AST to preserve code blocks and tables
3. **Parent-Child Indexing** — Links fine-grained child chunks to full parent sections
4. **Dual Indexing** — Stores embeddings in ChromaDB and term frequencies in BM25
5. **Self-Healing Diagnostics** — Automatically detects and repairs retrieval failures

## Chunking Engine

### Naive Chunker

The naive chunker uses a fixed-size `RecursiveCharacterTextSplitter` from LangChain with the following configuration:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

This approach is fast but has critical flaws:
- Code blocks may be split mid-function
- Markdown tables are truncated at arbitrary row boundaries
- Heading context is lost after the first chunk

### Structure-Aware Chunker

The structure-aware chunker parses the document into typed blocks:

| Block Type  | Detection Method       | Splitting Rule              |
|-------------|------------------------|-----------------------------|
| heading     | Regex `^#{1,6}\s+`     | Triggers new chunk section  |
| code_block  | Fenced `` ``` ``       | **Never split**             |
| table       | Pipe-delimited rows    | **Never split**             |
| paragraph   | Contiguous text lines  | Grouped under nearest heading |
| blank       | Empty lines            | Discarded                   |

Each chunk carries its **hierarchical heading path**, for example:
`# Architecture > ## Chunking Engine > ### Structure-Aware Chunker`

### Parent-Child Indexing

The parent-child strategy uses a two-tier index:

```python
def build_parent_child_index(parent_chunks, *, child_size=200, child_overlap=30):
    """
    Parent = full structure-aware section (large context window).
    Child  = smaller sub-chunks for precise vector matching.
    
    Atomic blocks (code, tables) are kept whole even in children.
    Each child carries metadata.parent_id → parent chunk.
    """
    parents, children = [], []
    for parent in parent_chunks:
        blocks = parse_blocks(parent["text"])
        for block in blocks:
            if block.type in ("code_block", "table"):
                children.append(make_child(block, parent_id=parent["id"]))
            else:
                for fragment in sub_split(block.content, child_size):
                    children.append(make_child(fragment, parent_id=parent["id"]))
    return parents, children
```

## Search Engine

### Dense Vector Search

ChromaDB stores document embeddings using the `all-MiniLM-L6-v2` sentence transformer model. The model produces 384-dimensional vectors and is cached locally in `outputs/models/` for fully offline operation.

Search uses cosine similarity:
```
score = 1 - cosine_distance(query_embedding, chunk_embedding)
```

### BM25 Sparse Search

BM25 (Best Matching 25) is a probabilistic ranking function used for lexical matching. It excels at finding exact terminology that dense embeddings might miss.

The BM25 formula:

```
score(D, Q) = Σ IDF(qi) · [ f(qi, D) · (k1 + 1) ] / [ f(qi, D) + k1 · (1 - b + b · |D|/avgdl) ]
```

Where:
- `f(qi, D)` = term frequency of query term qi in document D
- `|D|` = document length
- `avgdl` = average document length
- `k1 = 1.5`, `b = 0.75` (default parameters)

### Hybrid Search with Reciprocal Rank Fusion

When vocabulary mismatch is detected, the system switches to hybrid search combining dense and sparse results using Reciprocal Rank Fusion (RRF):

```python
def reciprocal_rank_fusion(results_lists, k=60):
    scores = {}
    for results in results_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc.id] = scores.get(doc.id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## Self-Healing Diagnostics

### Root Cause Classification

When a query fails to retrieve the expected chunk, the diagnoser classifies the failure:

| Root Cause      | Condition                                    | Auto-Repair Action     |
|-----------------|----------------------------------------------|------------------------|
| PARENT_NEEDED   | Child chunk too small; parent would match    | Switch to parent granularity |
| VOCAB_MISMATCH  | BM25 matches but dense does not              | Enable hybrid search (RRF) |
| TABLE_SPLIT     | Expected chunk has truncated table           | Re-chunk with table protection |
| CODE_SPLIT      | Expected chunk has unclosed code fence       | Re-chunk with code protection |
| CROSS_REF_MISS  | Answer spans multiple sections               | Expand context window (top-k++) |

### Evaluation Pipeline

The evaluation runs in two phases:

1. **Dev Split (70%)**: Measures baseline hit rate, diagnoses failures, applies repairs
2. **Held-Out Split (30%)**: Verifies that repairs caused zero regressions

```python
def regression_check(holdout_queries, searcher, old_config, new_config):
    regressions = 0
    for query in holdout_queries:
        old_hit = expected_id in searcher.search(query, config=old_config)
        new_hit = expected_id in searcher.search(query, config=new_config)
        if old_hit and not new_hit:
            regressions += 1
    return regressions == 0  # Must be zero
```

## Multi-Agent Pipeline

The system uses LangGraph state machines to coordinate nine specialized agents:

| Agent          | Role                                         | Default Mode      |
|----------------|----------------------------------------------|--------------------|
| QueryRewriter  | Expands and normalizes user queries          | Deterministic      |
| Retriever      | Executes dense/BM25/hybrid search            | Deterministic      |
| Reviewer       | Grades chunk relevance and coverage          | Deterministic      |
| Diagnoser      | Classifies retrieval failures                | Deterministic      |
| Fixer          | Applies targeted chunk repairs               | Deterministic      |
| Evaluator      | Measures hit rates on dev/holdout splits     | Deterministic      |
| Guardian       | Enforces safety invariants                   | Deterministic      |
| Answerer       | Generates answer with citations              | LLM (with fallback)|
| Explainer      | Produces execution trace summary             | Deterministic      |

### Guardian Invariants

The Guardian agent enforces three critical safety checks at every pipeline step:

1. **No Data Leakage**: Query text must not appear in chunk metadata
2. **Token Budget**: Total packed context must not exceed `MAX_CONTEXT_TOKENS` (4096)
3. **Citation Validity**: Every cited chunk ID must exist in retrieved results

```python
def guardian_check(state):
    errors = []
    # Invariant 1: data leakage
    for chunk in state.chunks:
        if state.query in chunk.metadata.values():
            errors.append("DATA_LEAK")
    # Invariant 2: token budget
    if sum(len(c.text) for c in state.chunks) // 4 > 4096:
        errors.append("TOKEN_BUDGET")
    # Invariant 3: citation validity
    chunk_ids = {c.id for c in state.chunks}
    for citation in state.citations:
        if citation.chunk_id not in chunk_ids:
            errors.append("CITATION_INVALID")
    return len(errors) == 0
```

## API Reference

### POST /api/chunk

Upload a file and receive side-by-side chunking comparison.

**Request**: `multipart/form-data` with `file` field.

**Response**:
```json
{
  "source": "architecture.md",
  "text_length": 8421,
  "naive_chunks": [...],
  "structure_aware_chunks": [...],
  "parents": [...],
  "children": [...],
  "stats": {
    "naive": { "chunk_count": 17, "split_code_blocks": 3, "split_tables": 2 },
    "structure_aware": { "chunk_count": 12, "split_code_blocks": 0, "split_tables": 0 }
  }
}
```

### POST /ask

Ask the knowledge base a question with full citation tracing.

**Request**:
```json
{
  "query": "How does hybrid search work?",
  "top_k": 5,
  "search_mode": "hybrid"
}
```

### POST /loop

Start the self-healing evaluation and diagnosis loop.

### GET /run/{run_id}/status

Check the status of a running evaluation loop.

### POST /review/{run_id}

Submit human review boundary decisions.

## Configuration

All configuration is managed through environment variables and `backend/config.py`:

| Variable                    | Default               | Description                           |
|-----------------------------|-----------------------|---------------------------------------|
| SMARTCHUNK_LLM_URL          | (empty)               | OpenAI-compatible LLM endpoint        |
| SMARTCHUNK_MODEL_DEFAULT    | none                  | Default model (none = deterministic)  |
| SMARTCHUNK_MODEL_ANSWERER   | none                  | Model for the Answerer agent          |
| NAIVE_CHUNK_SIZE            | 500                   | Fixed chunk size for naive chunker    |
| NAIVE_CHUNK_OVERLAP         | 50                    | Overlap between naive chunks          |
| CHILD_CHUNK_SIZE            | 200                   | Child chunk size for parent-child     |
| MAX_CONTEXT_TOKENS          | 4096                  | Maximum tokens in packed context      |

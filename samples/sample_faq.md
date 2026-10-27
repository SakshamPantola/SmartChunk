# RAG Pipeline FAQ

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. Instead of relying solely on a language model's parametric knowledge, RAG first retrieves relevant document chunks and then uses them as context for generating answers.

## Why use structure-aware chunking?

Traditional fixed-size chunking treats documents as flat text. This causes several problems:

- **Code blocks get split mid-function**, making retrieved code unusable
- **Tables lose rows**, making data incomplete
- **Heading context is lost**, making it unclear what section a chunk belongs to

Structure-aware chunking solves these problems by parsing the document's structure (headings, code fences, pipe tables) and respecting those boundaries.

## How does the self-healing loop work?

The self-healing loop follows a four-step process:

1. **Evaluate**: Run a set of test queries against the current index
2. **Diagnose**: Classify why specific queries failed to retrieve expected chunks
3. **Repair**: Apply targeted fixes based on the diagnosis
4. **Verify**: Re-run queries to confirm improvements without regressions

### Diagnosis Categories

```python
class RootCause(Enum):
    PARENT_NEEDED = "PARENT_NEEDED"     # Child chunk too small
    VOCAB_MISMATCH = "VOCAB_MISMATCH"   # Dense embedding missed terminology
    TABLE_SPLIT = "TABLE_SPLIT"         # Table was truncated
    CODE_SPLIT = "CODE_SPLIT"           # Code block was split
    CROSS_REF_MISS = "CROSS_REF_MISS"   # Answer spans multiple sections
```

### Repair Actions

| Diagnosis       | Repair Action    | Effect                              |
|-----------------|------------------|-------------------------------------|
| PARENT_NEEDED   | USE_PARENT       | Switch retrieval to parent chunks   |
| VOCAB_MISMATCH  | HYBRID_SEARCH    | Enable BM25 + Dense fusion         |
| TABLE_SPLIT     | RECHUNK_TABLE    | Re-parse with table protection     |
| CODE_SPLIT      | RECHUNK_CODE     | Re-parse with code protection      |
| CROSS_REF_MISS  | EXPAND_CONTEXT   | Increase top-k and context window  |

## What are the Guardian invariants?

The Guardian agent enforces three safety rules:

1. **No data leakage**: The user's query text must never appear verbatim in chunk metadata fields
2. **Token budget compliance**: The total packed context must stay within the configured token limit (default: 4096 tokens)
3. **Citation integrity**: Every citation reference in the answer must map to an actual retrieved chunk ID

## Can SmartChunk work offline?

Yes! SmartChunk is designed for 100% offline operation:

- **Embeddings**: Uses a local `all-MiniLM-L6-v2` model cached in `outputs/models/`
- **Vector store**: ChromaDB stores data locally in `outputs/chroma_db/`
- **LLM fallback**: When no LLM is configured, the answerer returns verbatim context chunks with formatted citations
- **No internet required**: After initial model download, everything runs locally

## How do I connect an LLM?

Set these environment variables before starting the server:

```bash
export SMARTCHUNK_LLM_URL="http://localhost:11434"  # Ollama
export SMARTCHUNK_MODEL_DEFAULT="llama3"             # Model name
export SMARTCHUNK_MODEL_ANSWERER="llama3"            # Per-agent override
```

The system will use the LLM for richer answer generation while maintaining the deterministic fallback as a safety net.

"""
Answerer agent node.

Budget-packs retrieved chunks and generates an answer.
In deterministic mode: returns formatted verbatim chunks with citations.
"""

from __future__ import annotations

import time
from backend.config import MAX_CONTEXT_TOKENS, APPROX_CHARS_PER_TOKEN
from backend.llm.client import get_llm_client


def _budget_pack(chunks: list[dict], max_tokens: int) -> list[dict]:
    """Keep as many chunks as fit within *max_tokens*."""
    packed: list[dict] = []
    used = 0
    for c in chunks:
        est = len(c.get("text", "")) // APPROX_CHARS_PER_TOKEN
        if used + est > max_tokens:
            break
        packed.append(c)
        used += est
    return packed


def answerer_node(state: dict) -> dict:
    """LangGraph node — generate answer from retrieved chunks."""
    t0 = time.perf_counter()

    chunks = state.get("retrieved_chunks", [])
    query = state.get("rewritten_query") or state.get("query", "")
    null_query = state.get("null_query", False)

    if null_query or not chunks:
        answer = "No relevant information found in the knowledge base."
        citations: list[dict] = []
    else:
        packed = _budget_pack(chunks, MAX_CONTEXT_TOKENS)

        # Build citations
        citations = []
        for c in packed:
            meta = c.get("metadata", {})
            citations.append(
                {
                    "chunk_id": c["id"],
                    "source": meta.get("source", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "line_start": meta.get("line_start", 0),
                    "line_end": meta.get("line_end", 0),
                    "score": c.get("score", 0.0),
                    "text_preview": c.get("text", "")[:150],
                }
            )

        llm = get_llm_client()
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}: {c.get('metadata',{}).get('source','')}]\n{c['text']}"
            for i, c in enumerate(packed)
        )
        prompt = (
            f"Answer the following question using ONLY the provided context chunks.\n"
            f"Cite chunks by number.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}"
        )
        answer = llm.generate(prompt, agent="answerer", context_chunks=packed)

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "answerer",
        "action": "generate",
        "detail": f"chunks_used={len(chunks)} null={null_query}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "trace": state.get("trace", []) + [trace_step],
    }

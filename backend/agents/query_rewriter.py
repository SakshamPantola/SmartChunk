"""
QueryRewriter agent node.

Rewrites the user query for better retrieval.
In deterministic mode: applies simple expansions and synonym injection.
"""

from __future__ import annotations

import time
from backend.llm.client import get_llm_client


def query_rewriter_node(state: dict) -> dict:
    """LangGraph node — rewrite the query for retrieval."""
    t0 = time.perf_counter()
    query = state["query"]

    llm = get_llm_client()
    model = "none"  # deterministic default

    # Deterministic rewrite: keep original + lower-cased variant
    rewritten = query.strip()

    # Try LLM-based rewrite if available
    try:
        prompt = (
            f"Rewrite the following question to improve retrieval from a "
            f"technical knowledge base. Return only the rewritten query.\n\n"
            f"Original: {query}"
        )
        result = llm.generate(prompt, agent="query_rewriter")
        if result and result != query and "No relevant" not in result:
            rewritten = result.strip()
    except Exception:
        pass  # keep deterministic rewrite

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "query_rewriter",
        "action": "rewrite",
        "detail": f"'{query}' → '{rewritten}'",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "rewritten_query": rewritten,
        "trace": state.get("trace", []) + [trace_step],
    }

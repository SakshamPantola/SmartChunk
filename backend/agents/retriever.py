"""
Retriever agent node.

Performs dense, BM25, or hybrid search depending on RetrievalConfig.
Checks the null-query threshold (calibrated p95) and flags below-threshold
queries.
"""

from __future__ import annotations

import time
from backend.config import DEFAULT_NULL_THRESHOLD


def retriever_node(state: dict, *, searcher=None) -> dict:
    """LangGraph node — retrieve top-k chunks."""
    t0 = time.perf_counter()

    config = state.get("retrieval_config", {})
    query = state.get("rewritten_query") or state["query"]
    top_k = config.get("top_k", 5)
    mode = config.get("search_mode", "dense")
    null_threshold = config.get("null_threshold", DEFAULT_NULL_THRESHOLD)

    results: list[dict] = []
    null_query = False

    if searcher is not None:
        results = searcher.search(query, top_k=top_k, mode=mode)

        # Null-query gating
        if results and results[0].get("score", 0) < null_threshold:
            null_query = True

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "retriever",
        "action": "search",
        "detail": f"mode={mode} top_k={top_k} results={len(results)} null={null_query}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "retrieved_chunks": results,
        "null_query": null_query,
        "trace": state.get("trace", []) + [trace_step],
    }

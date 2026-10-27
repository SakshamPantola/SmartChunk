"""
Evaluator agent node (loop graph).

Runs queries against dev or held-out splits and measures hit rate.
"""

from __future__ import annotations

import time
from backend.models import RetrievalConfig


def evaluator_node(state: dict, *, searcher=None, phase: str = "dev") -> dict:
    """
    LangGraph node — evaluate retrieval quality.

    *phase* is ``"dev"`` or ``"holdout"``.
    """
    t0 = time.perf_counter()

    config = RetrievalConfig(**state.get("retrieval_config", {}))
    queries = state.get(f"{phase}_queries", state.get("dev_queries", []))

    hits = 0
    failed: list[dict] = []

    for q in queries:
        query_text = q["query"]
        expected = q.get("expected_chunk_id", "")

        if searcher:
            results = searcher.search(
                query_text, top_k=config.top_k, mode=config.search_mode
            )
            retrieved_ids = [r["id"] for r in results]
        else:
            retrieved_ids = []

        if expected in retrieved_ids:
            hits += 1
        else:
            failed.append(
                {
                    "query": query_text,
                    "expected_chunk_id": expected,
                    "retrieved_ids": retrieved_ids,
                }
            )

    total = len(queries) or 1
    hit_rate = round(hits / total, 4)

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "evaluator",
        "action": f"eval_{phase}",
        "detail": f"hit_rate={hit_rate} hits={hits}/{len(queries)}",
        "duration_ms": round(elapsed, 1),
    }

    update: dict = {
        **state,
        "trace": state.get("trace", []) + [trace_step],
    }

    if phase == "dev":
        if not state.get("dev_hit_rate_before"):
            update["dev_hit_rate_before"] = hit_rate
        else:
            update["dev_hit_rate_after"] = hit_rate
        update["failed_queries"] = failed
    else:
        if not state.get("holdout_hit_rate_before"):
            update["holdout_hit_rate_before"] = hit_rate
        else:
            update["holdout_hit_rate_after"] = hit_rate

    return update

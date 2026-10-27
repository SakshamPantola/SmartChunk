"""
Guardian agent node.

Deterministic safety assertions at each pipeline step:
  1. No data leakage (query text not in chunk metadata).
  2. Token budget packing (total context ≤ MAX_CONTEXT_TOKENS).
  3. Every citation must exist in retrieved chunk metadata.
"""

from __future__ import annotations

import time
from backend.config import MAX_CONTEXT_TOKENS, APPROX_CHARS_PER_TOKEN


def guardian_node(state: dict) -> dict:
    """LangGraph node — validate invariants."""
    t0 = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    chunks = state.get("retrieved_chunks", [])
    citations = state.get("citations", [])
    query = state.get("query", "")

    # ── Invariant 1: no data leakage ────────────────────────────────
    for c in chunks:
        meta = c.get("metadata", {})
        for v in meta.values():
            if isinstance(v, str) and query and query.lower() in v.lower():
                errors.append(f"DATA_LEAK: query found in metadata of chunk {c['id']}")

    # ── Invariant 2: token budget ───────────────────────────────────
    total_chars = sum(len(c.get("text", "")) for c in chunks)
    est_tokens = total_chars // APPROX_CHARS_PER_TOKEN
    if est_tokens > MAX_CONTEXT_TOKENS:
        errors.append(
            f"TOKEN_BUDGET: packed {est_tokens} tokens > limit {MAX_CONTEXT_TOKENS}"
        )

    # ── Invariant 3: citation validity ──────────────────────────────
    chunk_ids = {c["id"] for c in chunks}
    for cit in citations:
        cid = cit.get("chunk_id", cit) if isinstance(cit, dict) else str(cit)
        if cid not in chunk_ids:
            errors.append(f"CITATION_INVALID: '{cid}' not in retrieved chunks")

    passed = len(errors) == 0

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "guardian",
        "action": "validate",
        "detail": f"passed={passed} errors={len(errors)}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "guardian_passed": passed,
        "errors": errors,
        "trace": state.get("trace", []) + [trace_step],
    }

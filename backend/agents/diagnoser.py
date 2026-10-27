"""
Diagnoser agent node (loop graph).

Wraps healing.diagnostics for the LangGraph state machine.
"""

from __future__ import annotations

import time
from backend.healing.diagnostics import diagnose_failures


def diagnoser_node(state: dict, *, child_store=None, parent_store=None, bm25_index=None) -> dict:
    """LangGraph node — classify root causes of failed retrievals."""
    t0 = time.perf_counter()

    failed = state.get("failed_queries", [])
    diagnoses_raw: list = []

    if child_store and parent_store and bm25_index and failed:
        diagnoses_obj = diagnose_failures(
            failed, child_store, parent_store, bm25_index
        )
        diagnoses_raw = [d.model_dump() for d in diagnoses_obj]

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "diagnoser",
        "action": "classify",
        "detail": f"failures={len(failed)} diagnoses={len(diagnoses_raw)}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "diagnoses": diagnoses_raw,
        "trace": state.get("trace", []) + [trace_step],
    }

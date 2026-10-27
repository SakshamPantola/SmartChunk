"""
Shared state definition for the LangGraph multi-agent pipeline.

Two graph variants share the same ``PipelineState``:
  * **ask** — query → rewrite → retrieve → guard → answer → explain
  * **loop** — evaluate → diagnose → fix → evaluate → guard (regression)
"""

from __future__ import annotations

from typing import Any, Optional, Annotated
from operator import add

from backend.models import (
    RetrievalConfig,
    RunStatus,
    Diagnosis,
    RepairRecord,
    TraceStep,
    Citation,
)


def _replace(a, b):
    """Reducer: always take the newer value."""
    return b


def _merge_list(a: list, b: list) -> list:
    """Reducer: append."""
    return (a or []) + (b or [])


class PipelineState(dict):
    """
    LangGraph state container.

    Fields are accessed as dict keys.  Default values are set in
    ``initial_state()``.
    """
    pass


def initial_ask_state(query: str, config: RetrievalConfig | None = None) -> dict:
    return {
        "mode": "ask",
        "query": query,
        "rewritten_query": "",
        "retrieval_config": (config or RetrievalConfig()).model_dump(),
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "trace": [],
        "null_query": False,
        "errors": [],
        "guardian_passed": True,
    }


def initial_loop_state(
    queries: list[dict],
    dev_queries: list[dict],
    holdout_queries: list[dict],
    config: RetrievalConfig | None = None,
    run_id: str = "",
) -> dict:
    return {
        "mode": "loop",
        "run_id": run_id,
        "all_queries": queries,
        "dev_queries": dev_queries,
        "holdout_queries": holdout_queries,
        "retrieval_config": (config or RetrievalConfig()).model_dump(),
        "failed_queries": [],
        "diagnoses": [],
        "repairs": [],
        "dev_hit_rate_before": 0.0,
        "dev_hit_rate_after": 0.0,
        "holdout_hit_rate_before": 0.0,
        "holdout_hit_rate_after": 0.0,
        "regressions": 0,
        "trace": [],
        "review_needed": False,
        "review_boundaries": [],
        "review_decisions": [],
        "status": RunStatus.RUNNING.value,
        "errors": [],
        "guardian_passed": True,
    }

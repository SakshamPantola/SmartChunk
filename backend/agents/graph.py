"""
LangGraph state-machine orchestrator.

Two compiled graphs:
  * ``ask_graph``  — Query → Rewrite → Retrieve → Guard → Answer → Explain
  * ``loop_graph`` — Eval(dev) → Diagnose → Fix → Eval(dev) → Guard(regression)
"""

from __future__ import annotations

import logging
import uuid
import asyncio
from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END

from backend.agents.state import initial_ask_state, initial_loop_state
from backend.agents.query_rewriter import query_rewriter_node
from backend.agents.retriever import retriever_node
from backend.agents.guardian import guardian_node
from backend.agents.answerer import answerer_node
from backend.agents.explainer import explainer_node
from backend.agents.diagnoser import diagnoser_node
from backend.agents.fixer import fixer_node
from backend.agents.evaluator import evaluator_node
from backend.agents.reviewer import reviewer_node
from backend.models import RetrievalConfig, RunStatus, HealingReport

log = logging.getLogger(__name__)

# ── Run store (in-memory) ────────────────────────────────────────────

_runs: dict[str, dict] = {}
_review_events: dict[str, asyncio.Event] = {}


def get_run(run_id: str) -> dict | None:
    return _runs.get(run_id)


def set_run(run_id: str, data: dict):
    _runs[run_id] = data


# ═══════════════════════════════════════════════════════════════════════
#  ASK GRAPH
# ═══════════════════════════════════════════════════════════════════════


def build_ask_graph(searcher=None):
    """Build and compile the ask pipeline graph."""

    graph = StateGraph(dict)

    # Nodes
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node(
        "retriever",
        partial(retriever_node, searcher=searcher),
    )
    graph.add_node("guardian", guardian_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("explainer", explainer_node)

    # Edges: linear pipeline
    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "guardian")
    graph.add_edge("guardian", "answerer")
    graph.add_edge("answerer", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile()


def run_ask(query: str, *, searcher=None, config: RetrievalConfig | None = None) -> dict:
    """Execute the ask pipeline synchronously and return the final state."""
    graph = build_ask_graph(searcher=searcher)
    state = initial_ask_state(query, config)
    result = graph.invoke(state)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  LOOP GRAPH  (self-healing)
# ═══════════════════════════════════════════════════════════════════════


def build_loop_graph(searcher=None, child_store=None, parent_store=None, bm25_index=None):
    """Build and compile the self-healing loop graph."""

    graph = StateGraph(dict)

    # Nodes
    graph.add_node(
        "eval_dev_before",
        partial(evaluator_node, searcher=searcher, phase="dev"),
    )
    graph.add_node(
        "diagnoser",
        partial(diagnoser_node, child_store=child_store, parent_store=parent_store, bm25_index=bm25_index),
    )
    graph.add_node("fixer", fixer_node)
    graph.add_node(
        "eval_dev_after",
        partial(evaluator_node, searcher=searcher, phase="dev"),
    )
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("guardian", guardian_node)

    # Edges
    graph.set_entry_point("eval_dev_before")
    graph.add_edge("eval_dev_before", "diagnoser")
    graph.add_edge("diagnoser", "fixer")
    graph.add_edge("fixer", "eval_dev_after")
    graph.add_edge("eval_dev_after", "reviewer")

    # Conditional: if review needed, pause; otherwise continue
    def after_reviewer(state):
        if state.get("review_needed"):
            return "end_for_review"
        return "guardian"

    graph.add_conditional_edges(
        "reviewer",
        after_reviewer,
        {"end_for_review": END, "guardian": "guardian"},
    )
    graph.add_edge("guardian", END)

    return graph.compile()


async def start_loop(
    queries: list[dict],
    *,
    dev_ratio: float = 0.7,
    searcher=None,
    child_store=None,
    parent_store=None,
    bm25_index=None,
) -> str:
    """
    Start a self-healing loop in the background.

    Returns *run_id*.
    """
    run_id = str(uuid.uuid4())[:8]

    # Split queries into dev / holdout
    split_idx = max(1, int(len(queries) * dev_ratio))
    dev_queries = queries[:split_idx]
    holdout_queries = queries[split_idx:]

    state = initial_loop_state(
        queries=queries,
        dev_queries=dev_queries,
        holdout_queries=holdout_queries,
        run_id=run_id,
    )

    set_run(run_id, {"status": RunStatus.RUNNING.value, "state": state, "report": None})
    _review_events[run_id] = asyncio.Event()

    # Run in background
    asyncio.create_task(_run_loop(run_id, searcher, child_store, parent_store, bm25_index))
    return run_id


async def _run_loop(run_id, searcher, child_store, parent_store, bm25_index):
    """Background task for the self-healing loop."""
    try:
        run_data = _runs[run_id]
        state = run_data["state"]

        graph = build_loop_graph(
            searcher=searcher,
            child_store=child_store,
            parent_store=parent_store,
            bm25_index=bm25_index,
        )
        result = graph.invoke(state)

        # Check if review is needed
        if result.get("review_needed"):
            run_data["status"] = RunStatus.AWAITING_REVIEW.value
            run_data["state"] = result
            run_data["pending_reviews"] = result.get("review_boundaries", [])

            # Wait for human review
            await _review_events[run_id].wait()
            _review_events[run_id].clear()

            # After review, re-run with review decisions applied
            result["review_needed"] = False
            result = graph.invoke(result)

        # Build report
        report = HealingReport(
            run_id=run_id,
            dev_hit_rate_before=result.get("dev_hit_rate_before", 0),
            dev_hit_rate_after=result.get("dev_hit_rate_after", 0),
            holdout_hit_rate_before=result.get("holdout_hit_rate_before", 0),
            holdout_hit_rate_after=result.get("holdout_hit_rate_after", 0),
            diagnoses=[],
            repairs=[],
            regressions=result.get("regressions", 0),
            status=RunStatus.DONE,
        )

        run_data["status"] = RunStatus.DONE.value
        run_data["state"] = result
        run_data["report"] = report.model_dump()

    except Exception as exc:
        log.exception("Loop failed: %s", exc)
        _runs[run_id]["status"] = RunStatus.FAILED.value
        _runs[run_id]["error"] = str(exc)


def submit_review(run_id: str, decisions: list[dict]):
    """Submit human review decisions and resume the loop."""
    run_data = _runs.get(run_id)
    if run_data:
        run_data["state"]["review_decisions"] = decisions
        run_data["state"]["review_needed"] = False
        if run_id in _review_events:
            _review_events[run_id].set()

"""
POST /loop — start self-healing evaluation & diagnosis loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.models import LoopRequest
from backend.agents.graph import start_loop

router = APIRouter()


@router.post("/loop")
async def start_healing_loop(req: LoopRequest, request: Request):
    """
    Start a closed-loop self-healing evaluation.

    If ``queries`` is empty, a set of demo queries is used.
    """
    app_state = request.app.state

    queries = [
        {"query": q, "expected_chunk_id": eid}
        for q, eid in zip(req.queries, req.expected_chunk_ids)
    ] if req.queries and req.expected_chunk_ids else _demo_queries()

    run_id = await start_loop(
        queries,
        dev_ratio=req.dev_ratio,
        searcher=getattr(app_state, "searcher", None),
        child_store=getattr(app_state, "child_store", None),
        parent_store=getattr(app_state, "parent_store", None),
        bm25_index=getattr(app_state, "bm25_index", None),
    )

    return {"run_id": run_id, "status": "running"}


def _demo_queries() -> list[dict]:
    """Fallback demo queries when none are provided."""
    return [
        {"query": "How does the chunking pipeline work?", "expected_chunk_id": "struct-0"},
        {"query": "What is parent-child indexing?", "expected_chunk_id": "struct-1"},
        {"query": "Explain the self-healing loop", "expected_chunk_id": "struct-2"},
        {"query": "What are the API endpoints?", "expected_chunk_id": "struct-3"},
        {"query": "How does BM25 sparse search work?", "expected_chunk_id": "struct-4"},
    ]

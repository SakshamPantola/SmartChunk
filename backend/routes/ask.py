"""
POST /ask — Q&A endpoint with retrieval, budget packing, citations, and trace.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.models import AskRequest, AskResponse, RetrievalConfig
from backend.agents.graph import run_ask

router = APIRouter()


@router.post("/ask")
async def ask(req: AskRequest, request: Request):
    """
    Ask the knowledge base a question.

    Returns answer, metadata citations, and full agent execution trace.
    """
    app_state = request.app.state
    searcher = getattr(app_state, "searcher", None)

    config = RetrievalConfig(
        search_mode=req.search_mode,
        top_k=req.top_k,
    )

    result = run_ask(req.query, searcher=searcher, config=config)

    return AskResponse(
        query=req.query,
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        trace=result.get("trace", []),
        null_query=result.get("null_query", False),
        retrieval_config=config,
    ).model_dump()

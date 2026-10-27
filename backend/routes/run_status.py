"""
GET /run/{run_id}/status — returns run state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.agents.graph import get_run

router = APIRouter()


@router.get("/run/{run_id}/status")
async def run_status(run_id: str):
    """
    Return current run state: running, awaiting_review, done, or failed.
    """
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")

    resp = {
        "run_id": run_id,
        "status": run.get("status", "unknown"),
    }
    if run.get("report"):
        resp["report"] = run["report"]
    if run.get("pending_reviews"):
        resp["pending_reviews"] = run["pending_reviews"]
    if run.get("error"):
        resp["error"] = run["error"]

    # Include trace if available
    state = run.get("state", {})
    if state.get("trace"):
        resp["trace"] = state["trace"]

    return resp

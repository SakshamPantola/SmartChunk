"""
POST /review/{run_id} — submit human review decisions and resume pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import ReviewRequest
from backend.agents.graph import get_run, submit_review

router = APIRouter()


@router.post("/review/{run_id}")
async def post_review(run_id: str, req: ReviewRequest):
    """
    Submit boundary review decisions (accept/merge) for a paused run.
    """
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if run.get("status") != "awaiting_review":
        raise HTTPException(
            400,
            f"Run '{run_id}' is not awaiting review (status={run.get('status')})"
        )

    decisions = [d.model_dump() for d in req.decisions]
    submit_review(run_id, decisions)

    return {"run_id": run_id, "status": "resumed", "decisions_applied": len(decisions)}

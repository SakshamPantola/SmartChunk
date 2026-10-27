"""
Reviewer agent node.

Handles human-in-the-loop review at low-confidence chunk boundaries.
When review is needed, sets ``review_needed=True`` and pauses the pipeline.
"""

from __future__ import annotations

import time


def reviewer_node(state: dict) -> dict:
    """
    LangGraph node — check if human review is needed.

    For the loop graph, reviews chunk boundary decisions.
    For the ask graph, reviews low-confidence answers.
    """
    t0 = time.perf_counter()

    chunks = state.get("retrieved_chunks", [])
    boundaries: list[dict] = []

    # Identify low-confidence boundaries between consecutive chunks
    for i in range(len(chunks) - 1):
        curr = chunks[i]
        next_c = chunks[i + 1]
        curr_meta = curr.get("metadata", {})
        next_meta = next_c.get("metadata", {})

        # Flag if consecutive chunks share the same heading path
        # (potential merge candidate)
        if (
            curr_meta.get("heading_path")
            and curr_meta.get("heading_path") == next_meta.get("heading_path")
        ):
            boundaries.append(
                {
                    "chunk_a_id": curr["id"],
                    "chunk_b_id": next_c["id"],
                    "heading_path": curr_meta.get("heading_path", ""),
                    "reason": "Same heading — consider merging",
                }
            )

    review_needed = len(boundaries) > 0

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "reviewer",
        "action": "boundary_check",
        "detail": f"boundaries={len(boundaries)} review_needed={review_needed}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "review_needed": review_needed,
        "review_boundaries": boundaries,
        "trace": state.get("trace", []) + [trace_step],
    }

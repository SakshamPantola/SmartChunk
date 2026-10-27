"""
Explainer agent node.

Produces a human-readable summary of the agent execution trace.
"""

from __future__ import annotations

import time


def explainer_node(state: dict) -> dict:
    """LangGraph node — summarise the execution trace."""
    t0 = time.perf_counter()

    trace = state.get("trace", [])
    summary_parts: list[str] = []
    for step in trace:
        agent = step.get("agent", "?")
        action = step.get("action", "?")
        detail = step.get("detail", "")
        ms = step.get("duration_ms", 0)
        summary_parts.append(f"{agent}.{action} ({ms:.0f}ms) — {detail}")

    trace_step = {
        "agent": "explainer",
        "action": "summarise",
        "detail": f"{len(trace)} steps traced",
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
    }

    return {
        **state,
        "trace": trace + [trace_step],
    }

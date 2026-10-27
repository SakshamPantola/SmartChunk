"""
Fixer agent node (loop graph).

Applies deterministic auto-repairs based on diagnoses.
"""

from __future__ import annotations

import time
from backend.models import Diagnosis, RetrievalConfig
from backend.healing.repair import apply_repair


def fixer_node(state: dict) -> dict:
    """LangGraph node — apply auto-repair actions."""
    t0 = time.perf_counter()

    diagnoses_raw = state.get("diagnoses", [])
    config = RetrievalConfig(**state.get("retrieval_config", {}))
    repairs: list[dict] = []

    for d_raw in diagnoses_raw:
        diag = Diagnosis(**d_raw)
        new_cfg, record = apply_repair(diag, config)
        config = new_cfg
        repairs.append(record.model_dump())

    elapsed = (time.perf_counter() - t0) * 1000
    trace_step = {
        "agent": "fixer",
        "action": "repair",
        "detail": f"applied {len(repairs)} fixes → config={config.model_dump()}",
        "duration_ms": round(elapsed, 1),
    }

    return {
        **state,
        "retrieval_config": config.model_dump(),
        "repairs": repairs,
        "trace": state.get("trace", []) + [trace_step],
    }

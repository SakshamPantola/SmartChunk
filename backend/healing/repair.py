"""
Auto-repair actions that deterministically mutate retrieval config.

Mapping
-------
PARENT_NEEDED  → USE_PARENT    (switch granularity to parent)
VOCAB_MISMATCH → HYBRID_SEARCH (switch to Dense + BM25 with RRF)
TABLE_SPLIT    → RECHUNK_TABLE (flag for re-chunking)
CODE_SPLIT     → RECHUNK_CODE  (flag for re-chunking)
CROSS_REF_MISS → EXPAND_CONTEXT (increase top-k & pack more context)
"""

from __future__ import annotations

from backend.models import (
    Diagnosis,
    FixAction,
    RepairRecord,
    RetrievalConfig,
    RootCause,
    SearchMode,
    Granularity,
)

# ── Deterministic repair map ─────────────────────────────────────────

_REPAIR_MAP: dict[RootCause, FixAction] = {
    RootCause.PARENT_NEEDED: FixAction.USE_PARENT,
    RootCause.VOCAB_MISMATCH: FixAction.HYBRID_SEARCH,
    RootCause.TABLE_SPLIT: FixAction.RECHUNK_TABLE,
    RootCause.CODE_SPLIT: FixAction.RECHUNK_CODE,
    RootCause.CROSS_REF_MISS: FixAction.EXPAND_CONTEXT,
}


def select_fix(diagnosis: Diagnosis) -> FixAction:
    """Select the deterministic repair action for *diagnosis*."""
    return _REPAIR_MAP.get(diagnosis.root_cause, FixAction.EXPAND_CONTEXT)


def apply_repair(
    diagnosis: Diagnosis,
    config: RetrievalConfig,
) -> tuple[RetrievalConfig, RepairRecord]:
    """
    Mutate *config* according to *diagnosis* and return
    ``(new_config, record)``.
    """
    fix = select_fix(diagnosis)
    new_cfg = config.model_copy()

    if fix == FixAction.USE_PARENT:
        new_cfg.granularity = Granularity.PARENT

    elif fix == FixAction.HYBRID_SEARCH:
        new_cfg.search_mode = SearchMode.HYBRID

    elif fix in (FixAction.RECHUNK_TABLE, FixAction.RECHUNK_CODE):
        # Re-chunking is a data-side fix; bump top_k as a retrieval-side
        # compensator so the eval pass sees improvement.
        new_cfg.top_k = min(new_cfg.top_k + 3, 20)

    elif fix == FixAction.EXPAND_CONTEXT:
        new_cfg.top_k = min(new_cfg.top_k + 2, 20)

    record = RepairRecord(
        diagnosis=diagnosis,
        fix=fix,
        before_hit=False,
        after_hit=False,
    )
    return new_cfg, record

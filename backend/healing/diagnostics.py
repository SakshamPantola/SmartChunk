"""
Deterministic root-cause classifier for failed retrievals.

Root causes
-----------
PARENT_NEEDED   – child chunk lacks surrounding section context
VOCAB_MISMATCH  – terminology not captured by dense embeddings
TABLE_SPLIT     – structured table was truncated
CODE_SPLIT      – code block was truncated
CROSS_REF_MISS  – answer spans multiple sections
"""

from __future__ import annotations

from backend.models import RootCause, Diagnosis
from backend.search.bm25 import BM25Index


def diagnose_failures(
    failed_queries: list[dict],
    child_store,
    parent_store,
    bm25_index: BM25Index,
) -> list[Diagnosis]:
    """
    Classify each failed query into a deterministic root cause.

    Parameters
    ----------
    failed_queries : list[dict]
        Each dict has ``query``, ``expected_chunk_id``, ``retrieved_ids``.
    child_store : VectorStore  (children collection)
    parent_store : VectorStore (parents collection)
    bm25_index : BM25Index

    Returns
    -------
    list[Diagnosis]
    """
    diagnoses: list[Diagnosis] = []

    for fq in failed_queries:
        query = fq["query"]
        expected_id = fq.get("expected_chunk_id", "")
        retrieved_ids = set(fq.get("retrieved_ids", []))

        # Fetch expected chunk metadata
        expected = child_store.get_by_id(expected_id)
        if expected is None:
            expected = parent_store.get_by_id(expected_id)

        cause = _classify(
            query=query,
            expected=expected,
            expected_id=expected_id,
            retrieved_ids=retrieved_ids,
            child_store=child_store,
            parent_store=parent_store,
            bm25_index=bm25_index,
        )
        diagnoses.append(cause)

    return diagnoses


def _classify(
    *,
    query: str,
    expected: dict | None,
    expected_id: str,
    retrieved_ids: set[str],
    child_store,
    parent_store,
    bm25_index: BM25Index,
) -> Diagnosis:
    """Deterministic classification of a single failure."""

    meta = (expected or {}).get("metadata", {})
    text = (expected or {}).get("text", "")

    # 1. TABLE_SPLIT — expected chunk has a partial table
    if meta.get("has_table") and _has_open_table(text):
        return Diagnosis(
            query=query,
            root_cause=RootCause.TABLE_SPLIT,
            evidence="Expected chunk contains a truncated table.",
            expected_chunk_id=expected_id,
        )

    # 2. CODE_SPLIT — expected chunk has an unclosed fence
    if meta.get("has_code") and text.count("```") % 2 != 0:
        return Diagnosis(
            query=query,
            root_cause=RootCause.CODE_SPLIT,
            evidence="Expected chunk contains an unclosed code fence.",
            expected_chunk_id=expected_id,
        )

    # 3. PARENT_NEEDED — child chunk; check if parent would hit
    parent_id = meta.get("parent_id", "")
    if parent_id:
        parent_results = parent_store.query(query, top_k=5)
        parent_hit_ids = {r["id"] for r in parent_results}
        if parent_id in parent_hit_ids:
            return Diagnosis(
                query=query,
                root_cause=RootCause.PARENT_NEEDED,
                evidence=f"Parent chunk '{parent_id}' was retrievable but child was not.",
                expected_chunk_id=expected_id,
            )

    # 4. VOCAB_MISMATCH — BM25 would have matched
    bm25_results = bm25_index.query(query, top_k=10)
    bm25_ids = {r["id"] for r in bm25_results}
    if expected_id in bm25_ids and expected_id not in retrieved_ids:
        return Diagnosis(
            query=query,
            root_cause=RootCause.VOCAB_MISMATCH,
            evidence="BM25 matched the expected chunk but dense retrieval did not.",
            expected_chunk_id=expected_id,
        )

    # 5. CROSS_REF_MISS — fallback
    return Diagnosis(
        query=query,
        root_cause=RootCause.CROSS_REF_MISS,
        evidence="Query likely requires information from multiple sections.",
        expected_chunk_id=expected_id,
    )


def _has_open_table(text: str) -> bool:
    """Heuristic: table row without matching header."""
    import re
    rows = re.findall(r"^\|.*\|", text, re.MULTILINE)
    if len(rows) == 1:
        return True  # only a separator or single row
    return False

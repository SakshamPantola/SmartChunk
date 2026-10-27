"""
Regression guard — verifies repairs on the held-out split.

The rule is simple: **zero regressions allowed**.
A regression = a query that *was* hitting before the repair but
*stops* hitting after.
"""

from __future__ import annotations

from backend.models import RetrievalConfig
from backend.search.hybrid import HybridSearcher


def regression_check(
    holdout_queries: list[dict],
    searcher: HybridSearcher,
    old_config: RetrievalConfig,
    new_config: RetrievalConfig,
) -> dict:
    """
    Run *holdout_queries* under both configs and compare.

    Parameters
    ----------
    holdout_queries : list[dict]
        Each dict has ``query`` and ``expected_chunk_id``.

    Returns
    -------
    dict with ``regressions`` (int), ``improvements`` (int),
    ``before_hits``, ``after_hits``, ``passed`` (bool).
    """
    before_hits = 0
    after_hits = 0
    regressions = 0
    improvements = 0
    details: list[dict] = []

    for hq in holdout_queries:
        query = hq["query"]
        expected = hq.get("expected_chunk_id", "")

        # Old config
        old_results = searcher.search(
            query, top_k=old_config.top_k, mode=old_config.search_mode.value
        )
        old_hit = any(r["id"] == expected for r in old_results)
        if old_hit:
            before_hits += 1

        # New config
        new_results = searcher.search(
            query, top_k=new_config.top_k, mode=new_config.search_mode.value
        )
        new_hit = any(r["id"] == expected for r in new_results)
        if new_hit:
            after_hits += 1

        if old_hit and not new_hit:
            regressions += 1
        elif not old_hit and new_hit:
            improvements += 1

        details.append(
            {
                "query": query,
                "expected": expected,
                "old_hit": old_hit,
                "new_hit": new_hit,
            }
        )

    total = len(holdout_queries) or 1
    return {
        "regressions": regressions,
        "improvements": improvements,
        "before_hits": before_hits,
        "after_hits": after_hits,
        "before_hit_rate": round(before_hits / total, 4),
        "after_hit_rate": round(after_hits / total, 4),
        "passed": regressions == 0,
        "details": details,
    }

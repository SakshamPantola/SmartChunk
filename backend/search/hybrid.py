"""
Hybrid search: Dense (ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion.
"""

from __future__ import annotations

from backend.search.vector_store import VectorStore
from backend.search.bm25 import BM25Index


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF.

    ``score_rrf = Σ 1 / (k + rank)`` across all lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results, start=1):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_map:
                doc_map[did] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for did, score in ranked:
        entry = dict(doc_map[did])
        entry["score"] = score
        out.append(entry)
    return out


class HybridSearcher:
    """
    Combines dense vector search and BM25 sparse search via RRF.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        Run search in *mode* (``dense``, ``bm25``, or ``hybrid``).
        """
        if mode == "dense":
            return self.vector_store.query(query, top_k=top_k)
        elif mode == "bm25":
            return self.bm25_index.query(query, top_k=top_k)
        else:
            # hybrid — fetch more candidates, then fuse
            dense_results = self.vector_store.query(query, top_k=top_k * 2)
            sparse_results = self.bm25_index.query(query, top_k=top_k * 2)
            fused = reciprocal_rank_fusion([dense_results, sparse_results])
            return fused[:top_k]


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    """Convenience functional wrapper around HybridSearcher."""
    searcher = HybridSearcher(vector_store=vector_store, bm25_index=bm25_index)
    return searcher.search(query=query, top_k=top_k, mode=mode)

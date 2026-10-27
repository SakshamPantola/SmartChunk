"""
BM25 sparse index for vocabulary-level matching.

Uses ``rank_bm25`` and keeps everything in-memory
(rebuilt on each ingestion — fast enough for local use).
"""

from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 sparse index over a list of chunk dicts."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ── build ────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """(Re-)build the index from *chunks* (each must have ``text`` and ``id``)."""
        self._chunks = list(chunks)
        self._corpus = [_tokenize(c["text"]) for c in self._chunks]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    index_chunks = build
    upsert_chunks = build

    # ── query ────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """
        Return the ``top_k`` best BM25 matches for *text*.

        Each result is ``{id, text, score, metadata}``.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # argsort descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                break
            c = self._chunks[idx]
            results.append(
                {
                    "id": c["id"],
                    "text": c["text"],
                    "score": float(scores[idx]),
                    "metadata": c.get("metadata", {}),
                }
            )
        return results

    def count(self) -> int:
        return len(self._chunks)

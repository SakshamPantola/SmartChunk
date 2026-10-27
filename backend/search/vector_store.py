"""
ChromaDB vector store wrapper — offline, local-only storage.

Two collections are maintained:
  * ``parents``  – full section chunks
  * ``children`` – smaller sub-chunks with ``parent_id`` in metadata
"""

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_DIR
from backend.ingestion.embeddings import embed_texts


class VectorStore:
    """Thin wrapper around a persistent ChromaDB client."""

    def __init__(self, collection_name: str = "children"):
        try:
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        except (AttributeError, TypeError, Exception):
            try:
                self._client = chromadb.Client(
                    Settings(
                        persist_directory=str(CHROMA_DIR),
                        anonymized_telemetry=False,
                    )
                )
            except Exception:
                self._client = chromadb.Client()

        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    # ── write ────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Insert or update chunks. Returns count upserted."""
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c.get("metadata", {}))
            # ChromaDB metadata must be flat str/int/float/bool
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""
                elif isinstance(v, (list, dict)):
                    m[k] = str(v)
            m["heading_path"] = c.get("heading_path", "")
            m["token_count"] = c.get("token_count", 0)
            metas.append(m)

        embeddings = embed_texts(texts)
        emb_lists = [e.tolist() for e in embeddings]

        self._col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=emb_lists,
        )
        return len(ids)

    # ── read ─────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Embed *text* and return the ``top_k`` nearest neighbours.

        Each result is a dict with ``id, text, score, metadata``.
        """
        emb = embed_texts([text])[0].tolist()
        kwargs: dict = {
            "query_embeddings": [emb],
            "n_results": min(top_k, self._col.count() or 1),
        }
        if where:
            kwargs["where"] = where
        raw = self._col.query(**kwargs)

        results = []
        for i in range(len(raw["ids"][0])):
            results.append(
                {
                    "id": raw["ids"][0][i],
                    "text": raw["documents"][0][i] if raw["documents"] else "",
                    "score": 1 - (raw["distances"][0][i] if raw["distances"] else 0),
                    "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                }
            )
        return results

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Fetch a single chunk by id."""
        res = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return {
            "id": res["ids"][0],
            "text": res["documents"][0] if res["documents"] else "",
            "metadata": res["metadatas"][0] if res["metadatas"] else {},
        }

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch multiple chunks by their ids."""
        if not ids:
            return []
        res = self._col.get(ids=ids, include=["documents", "metadatas"])
        out = []
        for i, cid in enumerate(res["ids"]):
            out.append(
                {
                    "id": cid,
                    "text": res["documents"][i] if res["documents"] else "",
                    "metadata": res["metadatas"][i] if res["metadatas"] else {},
                }
            )
        return out

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        """Delete all documents in the collection."""
        self._client.delete_collection(self._collection_name)
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

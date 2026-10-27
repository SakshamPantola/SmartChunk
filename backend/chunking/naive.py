"""
Naive fixed-size recursive chunker.
Uses LangChain's RecursiveCharacterTextSplitter (500 chars, 50 overlap).
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import NAIVE_CHUNK_SIZE, NAIVE_CHUNK_OVERLAP


def naive_chunk(
    text: str,
    *,
    chunk_size: int = NAIVE_CHUNK_SIZE,
    chunk_overlap: int = NAIVE_CHUNK_OVERLAP,
    source: str = "",
) -> list[dict]:
    """
    Split *text* into fixed-size chunks with overlap.

    Returns a list of chunk dicts ready for downstream indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)

    chunks: list[dict] = []
    search_offset = 0

    for idx, fragment in enumerate(splits):
        pos = text.find(fragment, search_offset)
        if pos == -1:
            pos = search_offset

        token_est = max(1, len(fragment) // 4)
        chunks.append(
            {
                "id": f"naive-{idx}",
                "text": fragment,
                "index": idx,
                "method": "naive",
                "token_count": token_est,
                "metadata": {
                    "source": source,
                    "heading_path": "",
                    "has_code": "```" in fragment,
                    "has_table": "|" in fragment and fragment.count("|") >= 3,
                    "line_start": text[:pos].count("\n"),
                    "line_end": text[:pos].count("\n") + fragment.count("\n"),
                    "parent_id": None,
                },
            }
        )
        search_offset = pos + 1

    return chunks

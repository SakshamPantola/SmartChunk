"""
Comparative statistics for naive vs structure-aware chunking.

Detects code-block and table splits to quantify chunking quality.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.*\|", re.MULTILINE)


def _count_code_blocks(text: str) -> int:
    """Count the number of complete fenced code blocks in *text*."""
    fences = _FENCE_RE.findall(text)
    return len(fences) // 2


def _count_tables(text: str) -> int:
    """Heuristic: a table is a contiguous run of ≥2 pipe-delimited rows."""
    count = 0
    in_table = False
    rows = 0
    for line in text.split("\n"):
        if _TABLE_ROW_RE.match(line.strip()):
            if not in_table:
                in_table = True
                rows = 1
            else:
                rows += 1
        else:
            if in_table and rows >= 2:
                count += 1
            in_table = False
            rows = 0
    if in_table and rows >= 2:
        count += 1
    return count


def _has_open_fence(text: str) -> bool:
    """True if *text* contains an unmatched opening fence."""
    fences = _FENCE_RE.findall(text)
    return len(fences) % 2 != 0


def _has_partial_table(text: str) -> bool:
    """True if *text* ends or starts mid-table (heuristic)."""
    lines = text.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    last = lines[-1].strip()
    # Starts with separator row (|---|) or ends mid-table
    if re.match(r"^\|[\s\-:]+\|$", first):
        return True
    if _TABLE_ROW_RE.match(last) and not _TABLE_ROW_RE.match(first):
        return False
    return False


def compute_split_stats(chunks: list[dict], full_text: str) -> dict:
    """
    Compute split statistics for a set of chunks.

    Returns
    -------
    dict with keys:
        chunk_count, avg_token_count,
        total_code_blocks, split_code_blocks,
        total_tables, split_tables
    """
    total_code = _count_code_blocks(full_text)
    total_tables = _count_tables(full_text)

    split_code = 0
    split_tables = 0
    token_sum = 0

    for chunk in chunks:
        txt = chunk.get("text", "")
        token_sum += chunk.get("token_count", max(1, len(txt) // 4))
        if _has_open_fence(txt):
            split_code += 1
        if _has_partial_table(txt):
            split_tables += 1

    count = len(chunks) or 1
    return {
        "chunk_count": len(chunks),
        "total_tokens": token_sum,
        "avg_token_count": round(token_sum / count, 1),
        "total_code_blocks": total_code,
        "split_code_blocks": split_code,
        "total_tables": total_tables,
        "split_tables": split_tables,
    }


def compare_methods(
    naive_chunks: list[dict],
    struct_chunks: list[dict],
    full_text: str,
) -> dict:
    """Side-by-side stats for naive vs structure-aware chunking."""
    n_stats = compute_split_stats(naive_chunks, full_text)
    s_stats = compute_split_stats(struct_chunks, full_text)
    saved = max(0, n_stats["total_tokens"] - s_stats["total_tokens"])
    return {
        "naive": n_stats,
        "structure_aware": s_stats,
        "tokens_saved": saved,
    }

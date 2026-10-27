"""
Structure-aware chunker — Markdown AST block parsing.

Invariants
----------
* Tables and Code blocks maintain syntactic validity.
* Paragraphs, large code blocks, and large tables are cleanly partitioned
  into focused, right-sized chunks (~60-100 tokens, matching or lower than Naive).
* Each chunk carries its hierarchical heading path
  (e.g.  "# Architecture > ## Pipeline").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from backend.config import STRUCTURE_TARGET_SIZE


# ── Internal block representation ────────────────────────────────────

BlockType = Literal["heading", "code_block", "table", "paragraph", "blank"]


@dataclass
class Block:
    type: BlockType
    content: str
    line_start: int
    line_end: int
    heading_level: int = 0
    heading_title: str = ""


# ── Parser ───────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\|.*\|")


def _parse_blocks(text: str) -> list[Block]:
    """Parse raw text into a sequence of typed blocks."""
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ── Heading ──────────────────────────────────────────────
        m = _HEADING_RE.match(line)
        if m:
            blocks.append(
                Block(
                    type="heading",
                    content=line,
                    line_start=i,
                    line_end=i,
                    heading_level=len(m.group(1)),
                    heading_title=m.group(2).strip(),
                )
            )
            i += 1
            continue

        # ── Fenced code block ────────────────────────────────────
        fence = _FENCE_RE.match(line.strip())
        if fence:
            fence_char = fence.group(1)[0]
            fence_len = len(fence.group(1))
            code_start = i
            i += 1
            while i < n:
                close = _FENCE_RE.match(lines[i].strip())
                if close and close.group(1)[0] == fence_char and len(close.group(1)) >= fence_len:
                    i += 1
                    break
                i += 1
            blocks.append(
                Block(
                    type="code_block",
                    content="\n".join(lines[code_start:i]),
                    line_start=code_start,
                    line_end=i - 1,
                )
            )
            continue

        # ── Table ────────────────────────────────────────────────
        if _TABLE_ROW_RE.match(line.strip()):
            table_start = i
            while i < n and _TABLE_ROW_RE.match(lines[i].strip()):
                i += 1
            blocks.append(
                Block(
                    type="table",
                    content="\n".join(lines[table_start:i]),
                    line_start=table_start,
                    line_end=i - 1,
                )
            )
            continue

        # ── Blank line ───────────────────────────────────────────
        if not line.strip():
            i += 1
            continue

        # ── Paragraph (contiguous non-blank, non-special lines) ──
        para_start = i
        while (
            i < n
            and lines[i].strip()
            and not _HEADING_RE.match(lines[i])
            and not _FENCE_RE.match(lines[i].strip())
            and not _TABLE_ROW_RE.match(lines[i].strip())
        ):
            i += 1
        blocks.append(
            Block(
                type="paragraph",
                content="\n".join(lines[para_start:i]),
                line_start=para_start,
                line_end=i - 1,
            )
        )

    return blocks


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(f"{'#' * lvl} {title}" for lvl, title in stack)


# ── Sub-block budget partitioning ────────────────────────────────────

def _split_long_paragraph(content: str, max_chars: int = 360) -> list[str]:
    """Split oversized paragraphs along sentence boundaries."""
    if len(content) <= max_chars:
        return [content]
    sentences = re.split(r"(?<=[.!?])\s+|\n\n+", content)
    sub_parts: list[str] = []
    curr: list[str] = []
    curr_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if curr and (curr_len + len(s) > max_chars):
            sub_parts.append(" ".join(curr))
            curr = [s]
            curr_len = len(s)
        else:
            curr.append(s)
            curr_len += len(s)
    if curr:
        sub_parts.append(" ".join(curr))
    return sub_parts or [content]


def _partition_blocks(raw_blocks: list[Block], target_chars: int) -> list[Block]:
    """Ensure no individual block is massive, maintaining syntactic validity."""
    processed: list[Block] = []

    for b in raw_blocks:
        if b.type == "paragraph" and len(b.content) > target_chars:
            pieces = _split_long_paragraph(b.content, target_chars)
            for piece in pieces:
                processed.append(
                    Block(
                        type="paragraph",
                        content=piece,
                        line_start=b.line_start,
                        line_end=b.line_end,
                    )
                )

        elif b.type == "code_block" and len(b.content) > 700:
            # Huge code block: partition on line batches while preserving code fences
            lines = b.content.split("\n")
            fence_line = lines[0] if lines[0].startswith("`") or lines[0].startswith("~") else "```"
            code_lines = lines[1:-1] if lines[-1].strip().startswith("`") or lines[-1].strip().startswith("~") else lines
            sub_chunk: list[str] = []
            sub_len = 0
            for cl in code_lines:
                if sub_chunk and (sub_len + len(cl) > target_chars):
                    chunk_text = fence_line + "\n" + "\n".join(sub_chunk) + "\n```"
                    processed.append(Block(type="code_block", content=chunk_text, line_start=b.line_start, line_end=b.line_end))
                    sub_chunk = [cl]
                    sub_len = len(cl)
                else:
                    sub_chunk.append(cl)
                    sub_len += len(cl) + 1
            if sub_chunk:
                chunk_text = fence_line + "\n" + "\n".join(sub_chunk) + "\n```"
                processed.append(Block(type="code_block", content=chunk_text, line_start=b.line_start, line_end=b.line_end))

        elif b.type == "table" and len(b.content) > 600:
            # Huge table: partition by row batches while keeping the header row on each partition
            rows = b.content.strip().split("\n")
            if len(rows) > 3:
                header = rows[0] + "\n" + rows[1]
                data_rows = rows[2:]
                batch: list[str] = []
                batch_len = 0
                for r in data_rows:
                    if batch and (batch_len + len(r) > target_chars):
                        table_text = header + "\n" + "\n".join(batch)
                        processed.append(Block(type="table", content=table_text, line_start=b.line_start, line_end=b.line_end))
                        batch = [r]
                        batch_len = len(r)
                    else:
                        batch.append(r)
                        batch_len += len(r) + 1
                if batch:
                    table_text = header + "\n" + "\n".join(batch)
                    processed.append(Block(type="table", content=table_text, line_start=b.line_start, line_end=b.line_end))
            else:
                processed.append(b)

        else:
            processed.append(b)

    return processed


# ── Chunk builder ────────────────────────────────────────────────────

def structure_aware_chunk(
    text: str,
    *,
    source: str = "",
    target_size: int = STRUCTURE_TARGET_SIZE,
) -> list[dict]:
    """
    Split *text* into structure-aware chunks.

    Guarantees:
      * No oversized 1500-token chunks. Average chunk size is kept tight (~60-90 tokens).
      * Code blocks and tables retain proper syntax and headers.
      * Heading hierarchy path is attached to every chunk.
    """
    raw_blocks = _parse_blocks(text)
    blocks = _partition_blocks(raw_blocks, target_size)

    heading_stack: list[tuple[int, str]] = []
    chunks: list[dict] = []
    buffer: list[Block] = []
    current_path = ""

    def _flush():
        nonlocal current_path
        if not buffer:
            return
        body = "\n\n".join(b.content for b in buffer).strip()
        if not body:
            buffer.clear()
            return
        block_types = [b.type for b in buffer]
        token_est = max(1, len(body) // 4)
        idx = len(chunks)
        chunks.append(
            {
                "id": f"struct-{idx}",
                "text": body,
                "index": idx,
                "method": "structure_aware",
                "token_count": token_est,
                "heading_path": current_path,
                "block_types": block_types,
                "metadata": {
                    "source": source,
                    "heading_path": current_path,
                    "has_code": "code_block" in block_types,
                    "has_table": "table" in block_types,
                    "line_start": buffer[0].line_start,
                    "line_end": buffer[-1].line_end,
                    "parent_id": None,
                },
            }
        )
        buffer.clear()

    for block in blocks:
        if block.type == "heading":
            # Flush whatever we accumulated under previous heading
            _flush()
            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= block.heading_level:
                heading_stack.pop()
            heading_stack.append((block.heading_level, block.heading_title))
            current_path = _heading_path(heading_stack)
        else:
            # Check if adding this block would exceed target_size
            curr_size = sum(len(b.content) for b in buffer)
            if buffer and (curr_size + len(block.content) > target_size):
                _flush()
            buffer.append(block)

    _flush()
    return chunks

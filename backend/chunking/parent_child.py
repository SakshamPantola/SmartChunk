"""
Parent-child indexing for structure-aware chunks.

* **Parent** = full section text (the structure-aware chunk as-is).
* **Child**  = smaller sub-chunks within the parent for precise vector
  search, linked back via ``parent_id``.

Atomic code blocks and tables are never split even in child chunks.
"""

from __future__ import annotations

from backend.config import CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
from backend.chunking.structure_aware import _parse_blocks, Block


def _sub_split_text(text: str, max_size: int, overlap: int) -> list[str]:
    """
    Naively split *text* into pieces of at most *max_size* chars
    with *overlap*, breaking only at paragraph boundaries (``\\n\\n``)
    or sentence boundaries (``. ``).
    """
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    # Try paragraph splits first
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > max_size and current:
            parts.append(current.strip())
            # overlap: keep tail of previous
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para).strip() if tail else para
        else:
            current = candidate
    if current.strip():
        parts.append(current.strip())

    return parts if parts else [text]


def build_parent_child_index(
    parent_chunks: list[dict],
    *,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> tuple[list[dict], list[dict]]:
    """
    Given *parent_chunks* (structure-aware), produce ``(parents, children)``.

    Each child chunk carries ``metadata.parent_id`` pointing to its parent.
    Atomic blocks (code blocks, tables) are kept whole in a single child.
    """
    parents: list[dict] = []
    children: list[dict] = []
    child_counter = 0

    for parent in parent_chunks:
        parent_id = parent["id"]
        parent_text = parent["text"]
        meta = dict(parent.get("metadata", {}))

        # Re-parse the parent text so we can protect atomic blocks
        blocks = _parse_blocks(parent_text)
        atomic_types = {"code_block", "table"}

        # Split non-atomic blocks into child fragments; keep atomic whole
        fragments: list[str] = []
        prose_buf = ""

        for blk in blocks:
            if blk.type in atomic_types:
                # flush prose
                if prose_buf.strip():
                    fragments.extend(
                        _sub_split_text(prose_buf.strip(), child_size, child_overlap)
                    )
                    prose_buf = ""
                # atomic block → one child
                fragments.append(blk.content)
            else:
                prose_buf += blk.content + "\n\n"

        if prose_buf.strip():
            fragments.extend(
                _sub_split_text(prose_buf.strip(), child_size, child_overlap)
            )

        if not fragments:
            fragments = [parent_text]

        parent_entry = {
            **parent,
            "id": parent_id,
            "child_ids": [],
        }

        for frag in fragments:
            cid = f"child-{child_counter}"
            child_counter += 1
            token_est = max(1, len(frag) // 4)
            child_entry = {
                "id": cid,
                "text": frag,
                "index": child_counter - 1,
                "method": "structure_aware",
                "token_count": token_est,
                "heading_path": parent.get("heading_path", ""),
                "metadata": {
                    **meta,
                    "parent_id": parent_id,
                    "has_code": "```" in frag,
                    "has_table": "|" in frag and frag.count("|") >= 3,
                },
            }
            children.append(child_entry)
            parent_entry["child_ids"].append(cid)

        parents.append(parent_entry)

    return parents, children

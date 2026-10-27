"""
POST /api/chunk — upload file, return naive vs structure-aware chunks & stats.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from backend.ingestion.loader import load_file
from backend.chunking.naive import naive_chunk
from backend.chunking.structure_aware import structure_aware_chunk
from backend.chunking.parent_child import build_parent_child_index
from backend.chunking.stats import compare_methods

router = APIRouter()


@router.post("/api/chunk")
async def chunk_file(file: UploadFile = File(...)):
    """
    Upload a file and receive side-by-side chunking comparison.

    Returns naive chunks, structure-aware chunks, parent-child index,
    and comparative split statistics.
    """
    # Save uploaded file to temp
    suffix = Path(file.filename or "upload.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text, meta = load_file(tmp_path)
        source = file.filename or "upload"

        # Run both chunkers
        naive_chunks = naive_chunk(text, source=source)
        struct_chunks = structure_aware_chunk(text, source=source)

        # Parent-child index
        parents, children = build_parent_child_index(struct_chunks)

        # Comparative stats
        stats = compare_methods(naive_chunks, struct_chunks, text)

        return {
            "source": source,
            "text_length": len(text),
            "naive_chunks": naive_chunks,
            "structure_aware_chunks": struct_chunks,
            "parents": parents,
            "children": children,
            "stats": stats,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)

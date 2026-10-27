"""
SmartChunk — FastAPI application entry point.

Assembles all routes, mounts the frontend, and initialises
search indexes on startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chunk import router as chunk_router
from backend.routes.loop import router as loop_router
from backend.routes.run_status import router as run_status_router
from backend.routes.review import router as review_router
from backend.routes.ask import router as ask_router
from backend.routes.agents_route import router as agents_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("smartchunk")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="SmartChunk",
    description="Structure-aware, self-healing RAG pipeline",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────
app.include_router(chunk_router)
app.include_router(loop_router)
app.include_router(run_status_router)
app.include_router(review_router)
app.include_router(ask_router)
app.include_router(agents_router)


# ── Startup — lazy-init search indexes ───────────────────────────────
@app.on_event("startup")
async def _startup():
    log.info("SmartChunk starting up …")
    # Search indexes will be initialised on first ingestion
    # to avoid blocking startup when no data exists yet.
    app.state.child_store = None
    app.state.parent_store = None
    app.state.bm25_index = None
    app.state.searcher = None
    log.info("Ready — http://127.0.0.1:8000")


# ── Ingest helper (called after /api/chunk to populate indexes) ──────
@app.post("/api/ingest")
async def ingest_to_index(payload: dict):
    """
    Ingest chunked data into the vector + BM25 indexes.

    Expects ``children`` and ``parents`` lists in the payload body.
    """
    from backend.search.vector_store import VectorStore
    from backend.search.bm25 import BM25Index
    from backend.search.hybrid import HybridSearcher

    children = payload.get("children", [])
    parents = payload.get("parents", [])

    # Initialise stores if needed
    if app.state.child_store is None:
        app.state.child_store = VectorStore("children")
        app.state.parent_store = VectorStore("parents")
        app.state.bm25_index = BM25Index()

    app.state.child_store.upsert_chunks(children)
    app.state.parent_store.upsert_chunks(parents)

    all_chunks = children + parents
    app.state.bm25_index.build(all_chunks)

    app.state.searcher = HybridSearcher(
        app.state.child_store, app.state.bm25_index
    )

    return {
        "indexed_children": len(children),
        "indexed_parents": len(parents),
        "total_bm25": app.state.bm25_index.count(),
    }


# ── Frontend serving ─────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

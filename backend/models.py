"""
Pydantic models shared across the SmartChunk API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class ChunkMethod(str, Enum):
    NAIVE = "naive"
    STRUCTURE_AWARE = "structure_aware"


class RootCause(str, Enum):
    PARENT_NEEDED = "PARENT_NEEDED"
    VOCAB_MISMATCH = "VOCAB_MISMATCH"
    TABLE_SPLIT = "TABLE_SPLIT"
    CODE_SPLIT = "CODE_SPLIT"
    CROSS_REF_MISS = "CROSS_REF_MISS"


class FixAction(str, Enum):
    USE_PARENT = "USE_PARENT"
    HYBRID_SEARCH = "HYBRID_SEARCH"
    RECHUNK_TABLE = "RECHUNK_TABLE"
    RECHUNK_CODE = "RECHUNK_CODE"
    EXPAND_CONTEXT = "EXPAND_CONTEXT"


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    DONE = "done"
    FAILED = "failed"


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    MERGE = "merge"


class SearchMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class Granularity(str, Enum):
    CHILD = "child"
    PARENT = "parent"


# ── Chunk models ─────────────────────────────────────────────────────

class ChunkMeta(BaseModel):
    source: str = ""
    heading_path: str = ""
    has_code: bool = False
    has_table: bool = False
    line_start: int = 0
    line_end: int = 0
    parent_id: Optional[str] = None


class Chunk(BaseModel):
    id: str
    text: str
    index: int
    method: ChunkMethod
    metadata: ChunkMeta = Field(default_factory=ChunkMeta)
    token_count: int = 0


# ── Chunk statistics ─────────────────────────────────────────────────

class SplitStats(BaseModel):
    chunk_count: int = 0
    avg_token_count: float = 0.0
    total_code_blocks: int = 0
    split_code_blocks: int = 0
    total_tables: int = 0
    split_tables: int = 0


class ChunkCompareResponse(BaseModel):
    naive: list[Chunk] = []
    structure_aware: list[Chunk] = []
    naive_stats: SplitStats = Field(default_factory=SplitStats)
    structure_stats: SplitStats = Field(default_factory=SplitStats)


# ── Diagnostics & healing ────────────────────────────────────────────

class Diagnosis(BaseModel):
    query: str
    root_cause: RootCause
    evidence: str = ""
    expected_chunk_id: str = ""


class RepairRecord(BaseModel):
    diagnosis: Diagnosis
    fix: FixAction
    before_hit: bool = False
    after_hit: bool = False


class HealingReport(BaseModel):
    run_id: str
    dev_hit_rate_before: float = 0.0
    dev_hit_rate_after: float = 0.0
    holdout_hit_rate_before: float = 0.0
    holdout_hit_rate_after: float = 0.0
    diagnoses: list[Diagnosis] = []
    repairs: list[RepairRecord] = []
    regressions: int = 0
    status: RunStatus = RunStatus.RUNNING


# ── Retrieval config ─────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    search_mode: SearchMode = SearchMode.DENSE
    granularity: Granularity = Granularity.CHILD
    top_k: int = 5
    null_threshold: float = 0.25


# ── Review ───────────────────────────────────────────────────────────

class BoundaryReview(BaseModel):
    chunk_id: str
    decision: ReviewDecision


class ReviewRequest(BaseModel):
    decisions: list[BoundaryReview]


# ── Ask / Q&A ────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: str
    source: str = ""
    heading_path: str = ""
    line_start: int = 0
    line_end: int = 0
    score: float = 0.0
    text_preview: str = ""


class TraceStep(BaseModel):
    agent: str
    action: str
    detail: str = ""
    duration_ms: float = 0.0


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: SearchMode = SearchMode.DENSE


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = []
    trace: list[TraceStep] = []
    null_query: bool = False
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)


# ── Agent config ─────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    name: str
    model: str = "none"
    enabled: bool = True


class AgentConfigUpdate(BaseModel):
    agents: list[AgentConfig]


# ── Loop ─────────────────────────────────────────────────────────────

class LoopRequest(BaseModel):
    """Starts a self-healing evaluation loop."""
    queries: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    dev_ratio: float = 0.7


class LoopStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    report: Optional[HealingReport] = None
    pending_reviews: list[dict[str, Any]] = []

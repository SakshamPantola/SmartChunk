"""
SmartChunk configuration — paths, env-vars, defaults.
Everything is offline-first; remote LLM is opt-in.
"""

import os
from pathlib import Path

# Disable HF symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
CHROMA_DIR = OUTPUTS_DIR / "chroma_db"
DATA_DIR = OUTPUTS_DIR / "data"

# ── Embedding model ─────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Chunking defaults ───────────────────────────────────────────────
NAIVE_CHUNK_SIZE = 500
NAIVE_CHUNK_OVERLAP = 50
STRUCTURE_TARGET_SIZE = 350
CHILD_CHUNK_SIZE = 200
CHILD_CHUNK_OVERLAP = 30

# ── Retrieval ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
NULL_QUERY_PERCENTILE = 95
DEFAULT_NULL_THRESHOLD = 0.25

# ── LLM / agents ────────────────────────────────────────────────────
LLM_URL: str = os.getenv("SMARTCHUNK_LLM_URL", "")
DEFAULT_MODEL: str = os.getenv("SMARTCHUNK_MODEL_DEFAULT", "none")

AGENT_NAMES: list[str] = [
    "retriever", "diagnoser", "fixer", "evaluator",
    "guardian", "reviewer", "query_rewriter", "answerer", "explainer",
]


def get_agent_model(agent: str) -> str:
    """Return the model name for *agent*, falling back to DEFAULT_MODEL."""
    return os.getenv(f"SMARTCHUNK_MODEL_{agent.upper()}", DEFAULT_MODEL)


# ── Token budget ─────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS = 4096
APPROX_CHARS_PER_TOKEN = 4

# ── Ensure output dirs exist on import ──────────────────────────────
for _d in (OUTPUTS_DIR, MODELS_DIR, CHROMA_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

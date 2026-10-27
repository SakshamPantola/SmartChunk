"""
File loaders for Markdown, PDF, HTML, and plain-text.

Each loader returns ``(text, metadata_dict)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_markdown(path: Path) -> Tuple[str, dict]:
    text = _read_text(path)
    return text, {"source": path.name, "format": "markdown"}


def load_txt(path: Path) -> Tuple[str, dict]:
    text = _read_text(path)
    return text, {"source": path.name, "format": "txt"}


def load_html(path: Path) -> Tuple[str, dict]:
    from bs4 import BeautifulSoup

    raw = _read_text(path)
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text, {"source": path.name, "format": "html"}


def load_pdf(path: Path) -> Tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    text = "\n\n".join(pages)
    return text, {"source": path.name, "format": "pdf", "pages": len(reader.pages)}


_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_txt,
    ".text": load_txt,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_file(path: str | Path) -> Tuple[str, dict]:
    """
    Auto-detect file type and return ``(text, metadata)``.

    Raises ``ValueError`` for unsupported extensions.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(_LOADERS)}"
        )
    return loader(p)

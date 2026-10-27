"""
LLM client with graceful fallback.

Default mode (model='none') is 100 % deterministic — no remote calls.
When an LLM endpoint is configured it's used for richer
generation; if the call fails the client silently falls back to
verbatim context chunks with metadata citations.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from groq import Groq
from backend.config import get_agent_model

log = logging.getLogger(__name__)

_client: Optional["LLMClient"] = None


class LLMClient:
    """Stateless LLM caller with fallback using Groq."""

    def __init__(self, default_model: str = "none"):
        self.default_model = default_model
        # Teri Groq API Key yahan embed kar di gayi hai
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

    # ── public API ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        agent: str = "",
        context_chunks: list[dict] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate text for *prompt*.

        * If model is ``'none'`` → deterministic fallback (no remote call).
        * Otherwise try the remote LLM; on failure fall back.
        """
        effective_model = model or (get_agent_model(agent) if agent else self.default_model)

        if effective_model == "none" or not self.groq_client:
            return self._deterministic_fallback(prompt, context_chunks)

        try:
            return self._call_remote(
                prompt,
                model=effective_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            log.warning("LLM call failed (%s), falling back to deterministic", exc)
            return self._deterministic_fallback(prompt, context_chunks)

    # ── internals ────────────────────────────────────────────────────

    def _call_remote(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the Groq API endpoint."""
        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content

    @staticmethod
    def _deterministic_fallback(
        prompt: str,
        context_chunks: list[dict] | None,
    ) -> str:
        """Return verbatim context chunks with metadata citations."""
        if not context_chunks:
            return "No relevant information found in the knowledge base."

        parts: list[str] = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source", "unknown")
            heading = meta.get("heading_path", "")
            line = meta.get("line_start", "?")
            citation = f"[{src}"
            if heading:
                citation += f" | {heading}"
            citation += f" | L{line}]"

            parts.append(f"**Chunk {i}** {citation}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)


def get_llm_client() -> LLMClient:
    """Return (or create) the global LLMClient singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
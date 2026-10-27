"""
Cloud Gemini embeddings to prevent Render OOM crashes.
"""

from __future__ import annotations

import os
import numpy as np
from google import genai

# Teri Gemini API key yahan set kar di hai
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def embed_texts(
    texts: list[str],
    *,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of strings using Gemini and return an (N, dim) float32 array.
    """
    # Gemini API list of texts ko direct handle kar leti hai
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=texts
    )
    embeddings = [e.values for e in response.embeddings]
    return np.asarray(embeddings, dtype=np.float32)

def embed_single(text: str) -> list[float]:
    """Embed a single string and return a flat list of floats."""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values
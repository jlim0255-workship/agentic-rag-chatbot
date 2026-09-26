"""Ollama HTTP client (local only). Grows at M2 with chat and embed calls."""
from __future__ import annotations

import httpx


class OllamaError(Exception):
    """Ollama is unreachable or returned an error."""


def normalise_model_name(name: str) -> str:
    """Ollama lists untagged models as '<name>:latest'."""
    return name if ":" in name else f"{name}:latest"


def list_models(base_url: str, timeout_s: float) -> list[str]:
    """Names of the models installed in Ollama (GET /api/tags)."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaError(f"Ollama unreachable at {base_url}: {e}") from e
    return [m["name"] for m in response.json().get("models", [])]


def embed(base_url: str, model: str, texts: list[str], timeout_s: float) -> list[list[float]]:
    """POST /api/embed. truncate=False: an over-long text is an error, never silently cut."""
    try:
        response = httpx.post(f"{base_url.rstrip('/')}/api/embed",
                              json={"model": model, "input": texts, "truncate": False},
                              timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama embed failed ({e.response.status_code}): {e.response.text}") from e
    except httpx.HTTPError as e:
        raise OllamaError(f"Ollama unreachable at {base_url}: {e}") from e
    vectors = response.json().get("embeddings", [])
    if len(vectors) != len(texts):
        raise OllamaError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
    return vectors

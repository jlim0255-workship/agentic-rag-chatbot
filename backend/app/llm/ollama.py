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

"""Ollama HTTP client (local only)."""
from __future__ import annotations

from typing import Any

import httpx


class OllamaError(Exception):
    """Ollama is unreachable or returned an error."""


class OllamaTimeout(OllamaError):
    """A call took longer than its timeout."""


def normalise_model_name(name: str) -> str:
    """Ollama lists untagged models as '<name>:latest'."""
    return name if ":" in name else f"{name}:latest"


def _request(method: str, base_url: str, path: str, timeout_s: float, json: Any = None) -> dict:
    try:
        response = httpx.request(method, f"{base_url.rstrip('/')}{path}", json=json, timeout=timeout_s)
        response.raise_for_status()
    except httpx.TimeoutException as e:
        raise OllamaTimeout(f"Ollama call {path} took longer than {timeout_s:.0f}s") from e
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama {path} failed ({e.response.status_code}): {e.response.text}") from e
    except httpx.HTTPError as e:
        raise OllamaError(f"Ollama unreachable at {base_url}: {e}") from e
    return response.json()


def list_models(base_url: str, timeout_s: float) -> list[str]:
    """Names of the models installed in Ollama (GET /api/tags)."""
    return [m["name"] for m in _request("GET", base_url, "/api/tags", timeout_s).get("models", [])]


def embed(base_url: str, model: str, texts: list[str], timeout_s: float) -> list[list[float]]:
    """POST /api/embed. truncate=False: an over-long text is an error, never silently cut."""
    body = _request("POST", base_url, "/api/embed", timeout_s,
                    {"model": model, "input": texts, "truncate": False})
    vectors = body.get("embeddings", [])
    if len(vectors) != len(texts):
        raise OllamaError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
    return vectors


def chat(base_url: str, model: str, prompt: str, timeout_s: float, num_ctx: int,
         format: dict | None = None) -> str:
    """One non-streamed chat turn. Thinking off and temperature 0: fast and repeatable.
    `format` is a JSON schema: Ollama then constrains the output to it."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    if format is not None:
        payload["format"] = format
    body = _request("POST", base_url, "/api/chat", timeout_s, payload)
    return body["message"]["content"]

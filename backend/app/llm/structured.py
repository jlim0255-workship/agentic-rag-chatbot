"""Structured LLM output (PRD 5.6): JSON schema -> validate -> one retry -> caller's fallback."""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm import ollama

T = TypeVar("T", bound=BaseModel)


def call_structured(prompt: str, schema: type[T], settings: Settings, timeout_s: float,
                    json_schema: dict | None = None) -> tuple[T | None, int]:
    """Returns (parsed output or None if invalid twice, number of LLM calls made).

    `json_schema` overrides the schema sent to Ollama, e.g. to restrict a field to an enum of valid values.
    None means: take the node's fallback. Timeouts and connection errors are not retried; they propagate.
    """
    for attempt in (1, 2):
        text = ollama.chat(settings.ollama_base_url, settings.fast_model, prompt, timeout_s,
                           settings.llm_num_ctx, format=json_schema or schema.model_json_schema())
        try:
            return schema.model_validate_json(text), attempt
        except ValidationError:
            continue
    return None, 2

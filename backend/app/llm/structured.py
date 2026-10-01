"""Structured LLM output (PRD 5.6): JSON schema -> validate -> one retry -> caller's fallback."""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm import ollama

T = TypeVar("T", bound=BaseModel)
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    """Without enforced `format` (Ollama Cloud) models wrap JSON in a ```json fence; the JSON itself is fine."""
    m = FENCE.match(text)
    return m.group(1) if m else text


def call_structured(prompt: str, schema: type[T], settings: Settings, timeout_s: float,
                    json_schema: dict | None = None) -> tuple[T | None, int]:
    """Returns (parsed output or None if invalid twice, number of LLM calls made).

    `json_schema` overrides the schema sent to Ollama, e.g. to restrict a field to an enum of valid values.
    None means: take the node's fallback. Timeouts and connection errors are not retried; they propagate.
    """
    fmt = json_schema or schema.model_json_schema()
    # The schema is also spelled out in the prompt: local Ollama enforces `format`, but Ollama Cloud ignores it
    # and the model then answers in prose or fenced JSON (found when evaluating gemma4:31b-cloud).
    prompt = f"{prompt}\n\nReply only with a JSON object matching this JSON schema:\n{json.dumps(fmt)}"
    for attempt in (1, 2):
        text = ollama.chat(settings.ollama_base_url, settings.fast_model, prompt, timeout_s,
                           settings.llm_num_ctx, format=fmt)
        try:
            return schema.model_validate_json(strip_fence(text)), attempt
        except ValidationError:
            continue
    return None, 2

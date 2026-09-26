"""Traditional mode (PRD 10.3): retrieve once with the original question -> generate -> return.

Yields (event, payload) pairs in the SSE order of PRD 11.2. Errors propagate; the API turns
them into an `error` event.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from app.config import Settings
from app.db import connection
from app.graph.generate import NOT_FOUND, cited_ids, generate
from app.llm.ollama import OllamaTimeout
from app.retrieval import search

Event = tuple[str, Any]

UNCHECKED_LABEL = "Not checked: traditional mode has no source check"
NOT_FOUND_TEXT = "Not found in the provided documents."


def trace(node: str, status: str, summary: str, data: dict | None = None) -> Event:
    return "trace", {"node": node, "status": status, "summary": summary, "data": data or {}}


def run_traditional(question: str, settings: Settings) -> Iterator[Event]:
    started = time.monotonic()
    llm_calls = 0

    def remaining() -> float:
        return settings.question_budget_s - (time.monotonic() - started)

    def finish(status: str) -> Iterator[Event]:
        yield "metrics", {"latency_ms": round((time.monotonic() - started) * 1000), "llm_calls": llm_calls,
                          "retries": 0, "regenerations": 0}
        yield "done", {"status": status}

    try:
        yield trace("retrieve", "start", "Retrieving…")
        with connection.connect(settings.database_url, settings.health_timeout_s) as conn:
            chunks = search.retrieve(conn, settings, question,
                                     timeout_s=min(settings.llm_call_timeout_s, remaining()))
        yield trace("retrieve", "end", f"{len(chunks)} chunks",
                    {"chunks": [{"id": c.id, "score": round(c.score, 4)} for c in chunks]})

        if remaining() <= 0:
            raise OllamaTimeout("question budget used up before generation")
        yield trace("generate", "start", "Writing the answer…")
        llm_calls += 1
        answer = generate(question, chunks, settings, timeout_s=min(settings.llm_call_timeout_s, remaining()))
        yield trace("generate", "end", "Answer written")
    except OllamaTimeout as e:
        yield trace("timeout", "end", str(e))
        yield from finish("timeout")
        return

    by_id = {c.id: c for c in chunks}
    citations = [{"chunk_id": i, "document": by_id[i].short_name, "section": by_id[i].label,
                  "page": by_id[i].pages} for i in cited_ids(answer) if i in by_id]
    # The model sometimes writes a sentence and then NOT_FOUND; without citations that is a not-found.
    if not answer or (NOT_FOUND in answer and not citations):
        yield "answer", {"text": NOT_FOUND_TEXT, "label": UNCHECKED_LABEL}
        yield "citations", []
        yield from finish("not_found")
        return

    answer = answer.replace(NOT_FOUND, "").strip()
    yield "answer", {"text": answer, "label": UNCHECKED_LABEL}
    yield "citations", citations
    yield from finish("answered")

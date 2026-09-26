"""FastAPI app. Run (from backend/): uvicorn app.api.main:app --reload"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app.api.health import HealthReport, build_health_report
from app.config import Settings, get_settings
from app.db import connection, meta
from app.graph.traditional import run_traditional

logger = logging.getLogger(__name__)


class EmbeddingModelMismatch(RuntimeError):
    pass


def verify_embedding_model(settings: Settings) -> None:
    """Refuse to start if stored vectors came from a different embedding model (PRD 8.3).

    A DB that is down is not a reason to refuse: /health reports it and the API recovers when it is up.
    """
    try:
        with connection.connect(settings.database_url, settings.health_timeout_s) as conn:
            check = meta.check_embedding_in_db(conn, settings.embed_model, settings.embed_dim)
    except psycopg.Error as e:
        logger.warning("Database unreachable at startup; embedding model not checked: %s", e)
        return
    if check.status == "mismatch":
        raise EmbeddingModelMismatch(
            "Refusing to start: " + "; ".join(check.problems)
            + ". Re-run ingestion with the configured model, or restore the previous EMBED_MODEL/EMBED_DIM."
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    verify_embedding_model(get_settings())
    yield


app = FastAPI(title="Act 854 Agentic RAG", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["agentic", "traditional"] = "traditional"


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_events(req: ChatRequest, settings: Settings) -> Iterator[str]:
    """Format pipeline events as SSE. Any failure becomes one `error` event, then the stream closes."""
    try:
        if req.mode == "agentic":
            raise NotImplementedError("Agentic mode is not built yet (milestone M4); use traditional mode.")
        for event, data in run_traditional(req.question, settings):
            yield sse(event, data)
    except Exception as e:  # noqa: BLE001 - the client must always get an event, never a hang
        logger.exception("chat failed")
        yield sse("error", {"message": str(e) or type(e).__name__})


@app.post("/chat")
def chat(req: ChatRequest, settings: Settings = Depends(get_settings)) -> StreamingResponse:
    return StreamingResponse(chat_events(req, settings), media_type="text/event-stream")


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, settings: Settings = Depends(get_settings)) -> dict:
    """Full chunk text and metadata, for the citation view."""
    with connection.connect(settings.database_url, settings.health_timeout_s) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            row = cur.execute(
                """
                SELECT c.id, c.document_id, d.title AS document_title, d.short_name, c.part, c.part_title,
                       c.section, c.subsection, c.heading, c.page_start, c.page_end, c.text
                FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = %s
                """, [chunk_id]).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no chunk {chunk_id!r}")
    return row


@app.get("/health", response_model=HealthReport)
def health(response: Response, settings: Settings = Depends(get_settings)) -> HealthReport:
    report = build_health_report(settings)
    if report.status != "ok":
        response.status_code = 503
    return report

"""FastAPI app. Run (from backend/): uvicorn app.api.main:app --reload"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from fastapi import Depends, FastAPI, Response

from app.api.health import HealthReport, build_health_report
from app.config import Settings, get_settings
from app.db import connection, meta

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


@app.get("/health", response_model=HealthReport)
def health(response: Response, settings: Settings = Depends(get_settings)) -> HealthReport:
    report = build_health_report(settings)
    if report.status != "ok":
        response.status_code = 503
    return report

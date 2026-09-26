"""/health: DB reachable, Ollama reachable, configured models installed, embedding model matches."""
from __future__ import annotations

from typing import Literal

import psycopg
from pydantic import BaseModel

from app.config import Settings
from app.db import connection, meta
from app.llm import ollama


class DatabaseStatus(BaseModel):
    ok: bool
    error: str | None = None


class OllamaStatus(BaseModel):
    ok: bool
    base_url: str
    missing_models: list[str] = []
    error: str | None = None


class ModelsInUse(BaseModel):
    """Shown in the UI sidebar (PRD 12.4)."""

    fast_model: str
    answer_model: str
    embed_model: str
    embed_dim: int


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    database: DatabaseStatus
    ollama: OllamaStatus
    models: ModelsInUse
    embedding: meta.EmbeddingCheck


def check_database(settings: Settings) -> tuple[DatabaseStatus, meta.EmbeddingCheck]:
    try:
        with connection.connect(settings.database_url, settings.health_timeout_s) as conn:
            embedding = meta.check_embedding_in_db(conn, settings.embed_model, settings.embed_dim)
    except psycopg.Error as e:
        return DatabaseStatus(ok=False, error=str(e).strip()), meta.EmbeddingCheck(status="unknown")
    return DatabaseStatus(ok=True), embedding


def check_ollama(settings: Settings) -> OllamaStatus:
    try:
        installed = {ollama.normalise_model_name(m)
                     for m in ollama.list_models(settings.ollama_base_url, settings.health_timeout_s)}
    except ollama.OllamaError as e:
        return OllamaStatus(ok=False, base_url=settings.ollama_base_url, error=str(e))
    wanted = dict.fromkeys([settings.fast_model, settings.answer_model, settings.embed_model])
    missing = [m for m in wanted if ollama.normalise_model_name(m) not in installed]
    return OllamaStatus(ok=True, base_url=settings.ollama_base_url, missing_models=missing)


def build_health_report(settings: Settings) -> HealthReport:
    database, embedding = check_database(settings)
    ollama_status = check_ollama(settings)
    healthy = (database.ok and ollama_status.ok and not ollama_status.missing_models
               and embedding.status != "mismatch")
    return HealthReport(
        status="ok" if healthy else "degraded",
        database=database,
        ollama=ollama_status,
        models=ModelsInUse(fast_model=settings.fast_model, answer_model=settings.answer_model,
                           embed_model=settings.embed_model, embed_dim=settings.embed_dim),
        embedding=embedding,
    )

"""Which embedding model produced the stored vectors, and whether it matches config.

Vectors from different embedding models are not comparable. If EMBED_MODEL changes without
re-ingesting, search silently returns the wrong chunks, so the API checks this at startup.
"""
from __future__ import annotations

from typing import Literal

import psycopg
from pydantic import BaseModel

EMBED_MODEL_KEY = "embed_model"
EMBED_DIM_KEY = "embed_dim"


class EmbeddingCheck(BaseModel):
    status: Literal["match", "mismatch", "not_ingested", "unknown"]
    stored_model: str | None = None
    stored_dim: int | None = None
    column_dim: int | None = None
    problems: list[str] = []


def read_embedding_meta(conn: psycopg.Connection) -> tuple[str | None, int | None]:
    """(model, dim) recorded by the embed step, or (None, None) before the first embed."""
    if not conn.execute("SELECT to_regclass('ingest_meta') IS NOT NULL").fetchone()[0]:
        return None, None
    rows = dict(conn.execute(
        "SELECT key, value FROM ingest_meta WHERE key = ANY(%s)", [[EMBED_MODEL_KEY, EMBED_DIM_KEY]]
    ).fetchall())
    dim = rows.get(EMBED_DIM_KEY)
    return rows.get(EMBED_MODEL_KEY), int(dim) if dim is not None else None


def write_embedding_meta(conn: psycopg.Connection, model: str, dim: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_meta (key, value) VALUES (%s, %s), (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
        """,
        [EMBED_MODEL_KEY, model, EMBED_DIM_KEY, str(dim)],
    )


def embedding_column_dim(conn: psycopg.Connection) -> int | None:
    """Declared size of chunks.embedding (pgvector stores it as the type modifier), or None if absent."""
    row = conn.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = to_regclass('chunks') AND attname = 'embedding' AND NOT attisdropped
        """
    ).fetchone()
    return row[0] if row else None


def check_embedding(configured_model: str, configured_dim: int, stored_model: str | None,
                    stored_dim: int | None, column_dim: int | None) -> EmbeddingCheck:
    problems: list[str] = []
    if column_dim is not None and column_dim != configured_dim:
        problems.append(f"chunks.embedding is vector({column_dim}) but EMBED_DIM={configured_dim}")
    if stored_model is not None and stored_model != configured_model:
        problems.append(f"stored vectors were made by {stored_model!r} but EMBED_MODEL={configured_model!r}")
    if stored_dim is not None and stored_dim != configured_dim:
        problems.append(f"stored vectors have {stored_dim} dimensions but EMBED_DIM={configured_dim}")

    if problems:
        status = "mismatch"
    elif stored_model is None:
        status = "not_ingested"
    else:
        status = "match"
    return EmbeddingCheck(status=status, stored_model=stored_model, stored_dim=stored_dim,
                          column_dim=column_dim, problems=problems)


def check_embedding_in_db(conn: psycopg.Connection, configured_model: str, configured_dim: int) -> EmbeddingCheck:
    stored_model, stored_dim = read_embedding_meta(conn)
    return check_embedding(configured_model, configured_dim, stored_model, stored_dim,
                           embedding_column_dim(conn))

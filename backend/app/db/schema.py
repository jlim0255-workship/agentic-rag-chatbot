"""Create the database schema (PRD 8.3). Safe to run repeatedly.

Usage (from backend/):
    python -m app.db.schema

Created from Python rather than a .sql file because the vector column size comes from
settings.EMBED_DIM, which a plain SQL init script cannot read.
"""
from __future__ import annotations

import psycopg

from app.config import get_settings
from app.db import meta
from app.db.connection import connect

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           text PRIMARY KEY,
    title        text NOT NULL,
    short_name   text NOT NULL UNIQUE,
    version_date date NOT NULL,
    source_url   text
);

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    document_id     text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    part            text,
    part_title      text,
    section         text NOT NULL,
    subsection      text,
    heading         text NOT NULL,
    page_start      int  NOT NULL,
    page_end        int  NOT NULL,
    text            text NOT NULL,
    contextual_text text NOT NULL,
    embedding       vector({dim}),
    CHECK (page_start <= page_end)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key        text PRIMARY KEY,
    value      text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
"""


class SchemaMismatch(Exception):
    """The existing chunks.embedding column size differs from EMBED_DIM."""


def create_schema(conn: psycopg.Connection, embed_dim: int) -> None:
    conn.execute(SCHEMA_SQL.format(dim=int(embed_dim)))
    column_dim = meta.embedding_column_dim(conn)
    if column_dim != embed_dim:
        raise SchemaMismatch(
            f"chunks.embedding is vector({column_dim}) but EMBED_DIM={embed_dim}. "
            "EMBED_MODEL/EMBED_DIM changed: run DROP TABLE chunks; then re-run this script and ingestion."
        )


def main() -> None:
    settings = get_settings()
    with connect(settings.database_url, settings.health_timeout_s) as conn:
        create_schema(conn, settings.embed_dim)
    print(f"schema ready (embedding vector({settings.embed_dim}))")


if __name__ == "__main__":
    main()

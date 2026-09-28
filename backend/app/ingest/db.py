"""Database access for ingestion."""
from __future__ import annotations

import psycopg

from app.config import get_settings
from app.db import meta
from app.db.connection import connect

from .models import Chunk, DocumentCard


def get_connection() -> psycopg.Connection:
    """Connection from settings.DATABASE_URL. Use as `with get_connection() as conn:`,
    which commits on success and rolls back on an exception."""
    settings = get_settings()
    return connect(settings.database_url, settings.health_timeout_s)


def to_vector_literal(vector: list[float]) -> str:
    """pgvector's text form, '[0.1,0.2,...]', cast with ::vector in SQL (no extra driver package)."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def upsert_document(conn: psycopg.Connection, cfg: DocumentCard) -> None:
    """INSERT ... ON CONFLICT (id) DO UPDATE into `documents` (id, title, short_name, version_date, source_url)."""
    conn.execute(
        """
        INSERT INTO documents (id, title, short_name, version_date, source_url)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title, short_name = EXCLUDED.short_name,
            version_date = EXCLUDED.version_date, source_url = EXCLUDED.source_url
        """,
        [cfg.id, cfg.title, cfg.short_name, cfg.version_date, cfg.source_url],
    )


def replace_chunks(conn: psycopg.Connection, document_id: str, chunks: list[Chunk]) -> None:
    """Make `chunks` for this document exactly equal `chunks` (idempotent re-runs).

    Upsert by id, and delete this document's rows whose id is no longer produced.
    Existing embeddings are cleared when contextual_text changes, so the embed step redoes them.
    """
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (id, document_id, part, part_title, section, subsection, heading,
                                page_start, page_end, text, contextual_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id, part = EXCLUDED.part, part_title = EXCLUDED.part_title,
                section = EXCLUDED.section, subsection = EXCLUDED.subsection, heading = EXCLUDED.heading,
                page_start = EXCLUDED.page_start, page_end = EXCLUDED.page_end, text = EXCLUDED.text,
                contextual_text = EXCLUDED.contextual_text,
                embedding = CASE WHEN chunks.contextual_text = EXCLUDED.contextual_text
                                 THEN chunks.embedding END
            """,
            [[c.id, document_id, c.part, c.part_title, c.section, c.subsection, c.heading,
              c.page_start, c.page_end, c.text, c.contextual_text] for c in chunks],
        )
        cur.execute("DELETE FROM chunks WHERE document_id = %s AND NOT (id = ANY(%s))",
                    [document_id, [c.id for c in chunks]])


def fetch_chunks_to_embed(conn: psycopg.Connection, reembed_all: bool) -> list[tuple[str, str]]:
    """Return (id, contextual_text) for chunks with embedding IS NULL, or all chunks if reembed_all."""
    where = "" if reembed_all else "WHERE embedding IS NULL"
    return conn.execute(f"SELECT id, contextual_text FROM chunks {where} ORDER BY id").fetchall()


def write_embeddings(conn: psycopg.Connection, rows: list[tuple[str, list[float]]]) -> None:
    """UPDATE chunks SET embedding = ... WHERE id = ... (committed with the caller's transaction)."""
    with conn.cursor() as cur:
        cur.executemany("UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                        [[to_vector_literal(v), id_] for id_, v in rows])


def record_embedding_model(conn: psycopg.Connection, model: str, dim: int) -> None:
    """Record which embedding model produced the stored vectors, in `ingest_meta` (PRD 8.3).

    The API refuses to start if this does not match settings.EMBED_MODEL.
    """
    meta.write_embedding_meta(conn, model, dim)


def stored_embedding_model(conn: psycopg.Connection) -> str | None:
    return meta.read_embedding_meta(conn)[0]

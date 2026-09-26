"""Database access for ingestion. PLACEHOLDERS: to be implemented with Claude Code.

Marked TODO(db) rather than TODO(Jlim) on purpose: CLAUDE.md tells Claude Code never to
fill TODO(Jlim), and these are the parts you want it to fill.
"""
from __future__ import annotations

from .models import Chunk, DocumentConfig


def get_connection():
    """Return a DB connection from settings.DATABASE_URL (backend/app/config.py)."""
    # TODO(db): driver choice is a new dependency; ask before adding (CLAUDE.md).
    raise NotImplementedError


def upsert_document(conn, cfg: DocumentConfig) -> None:
    """INSERT ... ON CONFLICT (id) DO UPDATE into `documents` (id, title, short_name, version_date, source_url)."""
    # TODO(db)
    raise NotImplementedError


def replace_chunks(conn, document_id: str, chunks: list[Chunk]) -> None:
    """Make `chunks` for this document exactly equal `chunks` (idempotent re-runs).

    Upsert by id, and delete this document's rows whose id is no longer produced.
    Existing embeddings must be cleared when contextual_text changes, so the embed step redoes them.
    """
    # TODO(db)
    raise NotImplementedError


def fetch_chunks_to_embed(conn, reembed_all: bool) -> list[tuple[str, str]]:
    """Return (id, contextual_text) for chunks with embedding IS NULL, or all chunks if reembed_all."""
    # TODO(db)
    raise NotImplementedError


def write_embeddings(conn, rows: list[tuple[str, list[float]]]) -> None:
    """UPDATE chunks SET embedding = ... WHERE id = ... in one transaction."""
    # TODO(db)
    raise NotImplementedError


def record_embedding_model(conn, model: str, dim: int) -> None:
    """Record which embedding model produced the stored vectors (PRD 8.3).

    The API refuses to start if this does not match settings.EMBED_MODEL.
    Where this lives (a metadata table?) is not in the PRD data model yet; see open questions.
    """
    # TODO(db)
    raise NotImplementedError

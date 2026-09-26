"""Step 3 of ingestion: embed chunks already in the DB and write their vectors.

Usage (from backend/):
    python -m app.ingest.run_embed            # only chunks with no embedding
    python -m app.ingest.run_embed --all      # re-embed everything (after changing EMBED_MODEL)

Reads contextual_text (header + text), never text alone. Model, dimension, URL and batch
size come from settings (backend/app/config.py); nothing is hardcoded (CLAUDE.md).
Talks only to local Ollama and Postgres, so it respects the offline rule.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from . import db

# from app.config import settings   # TODO(Jlim): exists after M0; needs OLLAMA_BASE_URL,
#                                   # EMBED_MODEL, EMBED_DIM, EMBED_BATCH_SIZE, LLM_CALL_TIMEOUT_S


def embed_texts(texts: list[str], base_url: str, model: str, timeout_s: float) -> list[list[float]]:
    """POST {base_url}/api/embed with {"model": model, "input": texts}; return response["embeddings"].

    Standard library urllib, so no new dependency. If backend/app/llm gets an Ollama client
    wrapper at M0/M2, switch to it so timeouts and errors are handled in one place.

    Documents are embedded WITHOUT an instruction prefix. (Qwen3-Embedding expects an
    instruction on the query side only; that belongs in retrieval, not here.)
    """
    # TODO(Jlim): implement. Raise a clear error if the response count != len(texts).
    ...


def check_dims(vectors: list[list[float]], expected_dim: int) -> None:
    """Fail loudly if any vector length != EMBED_DIM (model and column dim are tied, PRD 8.3)."""
    # TODO(Jlim): implement.
    ...


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="re-embed every chunk")
    args = parser.parse_args()

    conn = db.get_connection()
    rows = db.fetch_chunks_to_embed(conn, reembed_all=args.all)
    print(f"{len(rows)} chunks to embed")

    # TODO(Jlim): loop over rows in batches of settings.EMBED_BATCH_SIZE:
    #   vectors = embed_texts([text for _, text in batch], ...)
    #   check_dims(vectors, settings.EMBED_DIM)
    #   db.write_embeddings(conn, list(zip([id for id, _ in batch], vectors)))
    # then db.record_embedding_model(conn, settings.EMBED_MODEL, settings.EMBED_DIM)
    ...


if __name__ == "__main__":
    main()

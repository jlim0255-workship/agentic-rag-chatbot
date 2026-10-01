"""Step 3 of ingestion: embed chunks already in the DB and write their vectors.

Usage (from backend/):
    python -m app.ingest.run_embed            # only chunks with no embedding
    python -m app.ingest.run_embed --all      # re-embed everything (after changing EMBED_MODEL)

Reads contextual_text (header + text), never text alone. Model, dimension, URL and batch
size come from settings (backend/app/config.py); nothing is hardcoded (CLAUDE.md).
Talks only to local Ollama and Postgres, so it respects the offline rule.

Documents are embedded WITHOUT an instruction prefix. (Qwen3-Embedding expects an
instruction on the query side only; that belongs in retrieval, not here.)
"""
from __future__ import annotations

import argparse

from app.config import get_settings
from app.llm import ollama

from . import db


def check_dims(vectors: list[list[float]], expected_dim: int) -> None:
    """Fail loudly if any vector length != EMBED_DIM (model and column dim are tied)."""
    bad = {len(v) for v in vectors if len(v) != expected_dim}
    if bad:
        raise ValueError(f"embedding model returned vectors of size {sorted(bad)}, but EMBED_DIM={expected_dim}. "
                         "EMBED_MODEL and EMBED_DIM do not match.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="re-embed every chunk")
    args = parser.parse_args()
    settings = get_settings()

    # One transaction: vectors and the model record are committed together, or not at all.
    with db.get_connection() as conn:
        stored = db.stored_embedding_model(conn)
        if stored and stored != settings.embed_model and not args.all:
            raise SystemExit(f"Stored vectors came from {stored!r} but EMBED_MODEL={settings.embed_model!r}. "
                             "Embedding only the new chunks would mix two models; run with --all.")

        rows = db.fetch_chunks_to_embed(conn, reembed_all=args.all)
        print(f"{len(rows)} chunks to embed with {settings.embed_model}")
        for start in range(0, len(rows), settings.embed_batch_size):
            batch = rows[start:start + settings.embed_batch_size]
            vectors = ollama.embed(settings.ollama_base_url, settings.embed_model,
                                   [text for _, text in batch], settings.embed_timeout_s)
            check_dims(vectors, settings.embed_dim)
            db.write_embeddings(conn, [(id_, v) for (id_, _), v in zip(batch, vectors)])
            print(f"  {start + len(batch)}/{len(rows)}")
        db.record_embedding_model(conn, settings.embed_model, settings.embed_dim)
    print("done")


if __name__ == "__main__":
    main()

"""Ingestion steps 1-3: read the Markdown, chunk, insert. Embedding is step 4 (run_embed.py).

Usage (from backend/):
    python -m app.ingest.run_chunk --dump ../data/chunks.jsonl
    python -m app.ingest.run_chunk --no-db          # no database needed

Input: data/llamaparsed/<doc>.md (--corpus-dir), one per document: LlamaParse output corrected by hand,
with the document card in the +++ block at the top. See reader.py for the format and its checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import REPO_ROOT

from .chunker import make_block_chunk, make_chunks
from .models import Chunk
from .reader import Document, read_document

CORPUS_DIR = REPO_ROOT / "data" / "llamaparsed"


def load_documents(corpus_dir: Path) -> list[Document]:
    """Every <doc>.md in corpus_dir, sorted by id."""
    return [read_document(p) for p in sorted(corpus_dir.glob("*.md"))]


def chunk_document(doc: Document) -> list[Chunk]:
    """All chunks for one document, in document order."""
    chunks = make_chunks(doc.card, doc.units)
    if "long_title" in doc.blocks:
        block, lines = doc.blocks["long_title"]
        chunks.insert(0, make_block_chunk(doc.card, lines, block, "long_title"))
    if "Schedule" in doc.blocks:
        block, lines = doc.blocks["Schedule"]
        chunks.append(make_block_chunk(doc.card, lines, block, "Schedule"))
    return chunks


def sanity_check(chunks: list[Chunk]) -> None:
    """Cheap assertions before anything touches the DB."""
    seen: set[str] = set()
    for c in chunks:
        if c.id in seen:
            raise ValueError(f"duplicate chunk id {c.id}")
        seen.add(c.id)
        if not (c.section and c.document_id and c.text.strip()):
            raise ValueError(f"chunk {c.id} has an empty section, document_id or text")
        if c.page_start > c.page_end:
            raise ValueError(f"chunk {c.id}: page_start {c.page_start} > page_end {c.page_end}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--dump", type=Path, help="write all chunks as JSONL for spot checks")
    parser.add_argument("--no-db", action="store_true", help="skip the DB insert")
    args = parser.parse_args()

    docs = load_documents(args.corpus_dir)
    all_chunks: dict[str, list[Chunk]] = {}
    for doc in docs:
        chunks = chunk_document(doc)
        sanity_check(chunks)
        all_chunks[doc.card.id] = chunks
        print(f"{doc.card.id}: {len(chunks)} chunks")

    if args.dump:
        with args.dump.open("w", encoding="utf-8") as f:
            for chunks in all_chunks.values():
                for c in chunks:
                    f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    if args.no_db:
        return

    from . import db  # imported late so --no-db works without a DB driver installed

    # One transaction for everything: a failure leaves the database as it was.
    with db.get_connection() as conn:
        for doc in docs:
            db.upsert_document(conn, doc.card)
            db.replace_chunks(conn, doc.card.id, all_chunks[doc.card.id])
    print("inserted into the database; run app.ingest.run_embed next")


if __name__ == "__main__":
    main()

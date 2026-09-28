"""Ingestion steps 1-4: extract, outline, chunk, insert. Embedding is step 5 (run_embed.py).

Usage (from backend/):
    python -m app.ingest.run_chunk --corpus-dir ../data/corpus --dump ../data/chunks.jsonl
    python -m app.ingest.run_chunk --corpus-dir ../data/corpus --no-db                  # no database needed
    python -m app.ingest.run_chunk --corpus-dir ../data/corpus --no-db --write-outline  # new or changed PDF

Files, per document <doc> (see data/corpus/<doc>.toml for the card):
    data/work/<doc>.lines.txt      step 1 output: every PDF line with its printed page (not committed)
    data/corpus/<doc>.outline.md   step 2 output: the table of contents, reviewed and committed
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from .chunker import make_block_chunk, make_chunks
from .extract import between, find_line, load_lines, write_lines
from .models import Chunk, DocumentCard
from .outline import from_arrangement, from_body, read_markdown, to_markdown
from .structure import OutlineMismatch, detect_units


def load_cards(corpus_dir: Path) -> list[DocumentCard]:
    """Every <doc>.toml in corpus_dir, sorted by id. `pdf` is made absolute (it is relative to the card)."""
    cards = []
    for path in sorted(corpus_dir.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        data["pdf"] = str(path.resolve().parent / data["pdf"])
        cards.append(DocumentCard(id=path.stem, **data))
    return cards


def outline_path(corpus_dir: Path, card: DocumentCard) -> Path:
    return corpus_dir / f"{card.id}.outline.md"


def chunk_document(card: DocumentCard, corpus_dir: Path, write_outline: bool = False,
                   work_dir: Path | None = None) -> list[Chunk]:
    """All chunks for one document, in document order. Raises on any mismatch."""
    # 1. extract
    lines = load_lines(card)
    if work_dir:
        write_lines(lines, work_dir / f"{card.id}.lines.txt") # write the extracted line under /data/work
    body = between(lines, card.body_starts_after, card.body_ends_before)

    # 2. outline: built from the PDF, then compared with (or written to) the committed file
    if card.toc_starts_after:
        outline = from_arrangement(between(lines, card.toc_starts_after, card.toc_ends_before))
        source = f"the arrangement of sections (after {card.toc_starts_after!r})"
    else:
        outline = from_body(body)
        source = "the heading line above each numbered unit in the body"
    path = outline_path(corpus_dir, card)
    if write_outline:
        path.write_text(to_markdown(card.title, source, outline), encoding="utf-8")
    elif not path.exists() or read_markdown(path) != outline:
        raise OutlineMismatch(f"{card.id}: the outline built from the PDF differs from {path.name} (or it is "
                              "missing). Re-run with --write-outline, review the diff against the PDF, commit.")

    # 3. structure + chunks
    chunks = make_chunks(card, detect_units(body, outline))

    # Blocks outside the body: long title first (above the body), Schedule last (after it).
    def block_lines(block, search_from: int = 0):
        start = find_line(lines, block.first_line, search_from)
        return lines[start:find_line(lines, block.last_line, start) + 1]

    if card.long_title:
        chunks.insert(0, make_block_chunk(card, block_lines(card.long_title), card.long_title, "long_title"))
    if card.schedule:
        body_end = find_line(lines, card.body_ends_before, find_line(lines, card.body_starts_after))
        chunks.append(make_block_chunk(card, block_lines(card.schedule, body_end), card.schedule, "Schedule"))
    return chunks


# def sanity_check(chunks: list[Chunk]) -> None:
#     """Cheap assertions from PRD 8.4 before anything touches the DB."""
#     seen: set[str] = set()
#     for c in chunks:
#         if c.id in seen:
#             raise ValueError(f"duplicate chunk id {c.id}")
#         seen.add(c.id)
#         if not (c.section and c.document_id and c.text.strip()):
#             raise ValueError(f"chunk {c.id} has an empty section, document_id or text")
#         if c.page_start > c.page_end:
#             raise ValueError(f"chunk {c.id}: page_start {c.page_start} > page_end {c.page_end}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--dump", type=Path, help="write all chunks as JSONL for spot checks")
    parser.add_argument("--no-db", action="store_true", help="skip the DB insert")
    parser.add_argument("--write-outline", action="store_true",
                        help="(re)write data/corpus/<doc>.outline.md from the PDF instead of checking it")
    args = parser.parse_args()

    cards = load_cards(args.corpus_dir)
    work_dir = args.corpus_dir.parent / "work"
    all_chunks: dict[str, list[Chunk]] = {}
    for card in cards:
        chunks = chunk_document(card, args.corpus_dir, args.write_outline, work_dir)
        # sanity_check(chunks)
        all_chunks[card.id] = chunks
        print(f"{card.id}: {len(chunks)} chunks")
    if args.write_outline:
        print(f"outlines written to {args.corpus_dir}: review them against the PDFs before committing")

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
        for card in cards:
            db.upsert_document(conn, card)
            db.replace_chunks(conn, card.id, all_chunks[card.id])
    print("inserted into the database; run app.ingest.run_embed next")


if __name__ == "__main__":
    main()

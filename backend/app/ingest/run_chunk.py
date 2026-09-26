"""Step 1+2 of ingestion: parse and chunk every configured document, then insert into the DB.

Usage (from backend/):
    python -m app.ingest.run_chunk --corpus-dir ../data/corpus --dump ../data/chunks.jsonl
    python -m app.ingest.run_chunk --corpus-dir ../data/corpus --dump ../data/chunks.jsonl --no-db

--no-db runs everything except the insert, so chunking can be checked before the DB exists.
Embedding is a separate script (run_embed.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import make_block_chunk, make_chunks
from .extract import find_line, load_lines
from .models import Chunk, ChecklistEntry, DocumentConfig
from .structure import detect_units


def load_configs(corpus_dir: Path) -> list[DocumentConfig]:
    """Every <id>.json in corpus_dir that is not a checklist (*.toc.json).

    pdf_path and checklist_path are relative to the config file, so any working directory works.
    """
    configs = []
    for path in sorted(corpus_dir.glob("*.json")):
        if path.name.endswith(".toc.json"):
            continue
        cfg = DocumentConfig.model_validate_json(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        cfg.pdf_path = str(base / cfg.pdf_path)
        cfg.checklist_path = str(base / cfg.checklist_path)
        configs.append(cfg)
    return sorted(configs, key=lambda c: c.id)


def load_checklist(path: str) -> list[ChecklistEntry]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ChecklistEntry.model_validate(e) for e in entries]


def chunk_document(cfg: DocumentConfig) -> list[Chunk]:
    """Full chunk pipeline for one document. Raises on any checklist mismatch.

      1. lines = load_lines(cfg)
      2. body = lines between body_start_after and body_end_before (find_line)
      3. units = detect_units(body, checklist, cfg.has_parts)
      4. chunks = make_chunks(cfg, units)
      5. long title / Schedule: slice `lines` with the block's start/end_before, make_block_chunk
         - long title goes FIRST in the output, Schedule LAST (document order)
         - the long title sits before body_start_after, the Schedule after body_end_before,
           so search `lines`, not `body`
    """
    lines = load_lines(cfg)
    body_start = find_line(lines, cfg.body_start_after) + 1
    body_end = find_line(lines, cfg.body_end_before, body_start)
    units = detect_units(lines[body_start:body_end], load_checklist(cfg.checklist_path), cfg.has_parts)
    chunks = make_chunks(cfg, units)

    if cfg.long_title:
        start = find_line(lines, cfg.long_title.start)
        end = find_line(lines, cfg.long_title.end_before, start + 1)
        chunks.insert(0, make_block_chunk(cfg, lines[start:end], cfg.long_title, "long_title"))
    if cfg.schedule:
        start = find_line(lines, cfg.schedule.start, body_end)
        end = find_line(lines, cfg.schedule.end_before, start + 1)
        chunks.append(make_block_chunk(cfg, lines[start:end], cfg.schedule, "Schedule"))
    return chunks


def sanity_check(chunks: list[Chunk]) -> None:
    """Cheap assertions from PRD 8.4 before anything touches the DB.

    - ids unique
    - every chunk has non-empty section, document_id, text
    - page_start <= page_end
    """
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--dump", type=Path, help="write all chunks as JSONL for spot checks")
    parser.add_argument("--no-db", action="store_true", help="skip the DB insert")
    args = parser.parse_args()

    configs = load_configs(args.corpus_dir)
    all_chunks: dict[str, list[Chunk]] = {}
    for cfg in configs:
        chunks = chunk_document(cfg)
        sanity_check(chunks)
        all_chunks[cfg.id] = chunks
        print(f"{cfg.id}: {len(chunks)} chunks")

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
        for cfg in configs:
            db.upsert_document(conn, cfg)
            db.replace_chunks(conn, cfg.id, all_chunks[cfg.id])
    print("inserted into the database; run app.ingest.run_embed next")


if __name__ == "__main__":
    main()

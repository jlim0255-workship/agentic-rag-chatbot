"""Step 2, outline: the document's table of contents, built from the PDF itself.

- A document with an arrangement page (Act 854, pages 3-6) -> parsed from that page.
- A document without one (the regulations) -> the heading line above each "N. " line of the body.

The outline is saved as data/corpus/<doc>.outline.md, reviewed against the PDF by a person, and
committed. On later runs the freshly built outline must equal the committed file, so the chunker
only ever works from an outline someone has looked at.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Line, OutlineEntry
from .structure import PART_LINE, OutlineMismatch, is_all_caps, title_case, unit_number

TOC_ENTRY = re.compile(r"^(\d+[A-Z]*)\. (.+)$")  # " 23. Duty to give notification on cyber security incident"
COLUMN_LABEL = "Section"                          # printed above the numbers on each arrangement page


def from_arrangement(lines: list[Line]) -> list[OutlineEntry]:
    """Parse arrangement-of-sections lines:

        Part IV                                      -> a new Part
        NATIONAL CRITICAL INFORMATION ... LEAD       -> its title (ALL CAPS, may wrap)
        AND NATIONAL CRITICAL INFORMATION ... ENTITY
        18. Designation of ... sector lead as        -> a row
        national critical information infrastructure entity  -> the row's heading, wrapped
    """
    entries: list[OutlineEntry] = []
    part = part_title = None
    title: list[str] | None = None  # collecting a Part's title lines
    for line in lines:
        text = line.text
        if text == COLUMN_LABEL:
            continue
        if m := PART_LINE.match(text):
            part, title = m.group(1), []
        elif title is not None and is_all_caps(text):
            title.append(text)
        elif m := TOC_ENTRY.match(text):
            if title is not None:
                part_title, title = title_case(" ".join(title)), None
            entries.append(OutlineEntry(part=part, part_title=part_title, unit=m.group(1), heading=m.group(2)))
        elif entries:
            entries[-1].heading += " " + text
        else:
            raise OutlineMismatch(f"arrangement: unexpected line {text!r} on page {line.page}")
    return entries


def from_body(body: list[Line]) -> list[OutlineEntry]:
    """For documents without an arrangement page: every line whose number is the next one in
    sequence (1, 2, 3, ...) starts a unit, and the line above it is the heading."""
    entries: list[OutlineEntry] = []
    for i, line in enumerate(body):
        if i and unit_number(line) == str(len(entries) + 1):
            entries.append(OutlineEntry(part=None, part_title=None, unit=unit_number(line), heading=body[i - 1].text))
    return entries


def to_markdown(title: str, source: str, entries: list[OutlineEntry]) -> str:
    rows = [f"| {e.part or ''} | {e.part_title or ''} | {e.unit} | {e.heading} |" for e in entries]
    return "\n".join([
        f"# Outline: {title}", "",
        f"Generated from {source} by `python -m app.ingest.run_chunk --write-outline`.",
        "Check every row against the PDF before committing: ingestion only runs when the outline",
        "built from the PDF equals this file.", "",
        "| Part | Part title | Unit | Heading |", "|---|---|---|---|", *rows, ""])


def read_markdown(path: Path) -> list[OutlineEntry]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| ")][1:]
    entries = []
    for row in rows:
        part, part_title, unit, heading = (cell.strip() for cell in row.split("|")[1:-1])
        entries.append(OutlineEntry(part=part or None, part_title=part_title or None, unit=unit, heading=heading))
    return entries

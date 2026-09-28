"""Data models for ingestion: the document card, outline entries, lines, units and chunks.

Field meanings follow PRD Sections 8.3 and 8.5. The pipeline, one step per module:
    extract.py  PDF -> Line list            (writes data/work/<doc>.lines.txt)
    outline.py  Lines -> OutlineEntry list  (writes/checks data/corpus/<doc>.outline.md)
    structure.py  body Lines + outline -> Unit list
    chunker.py  Units -> Chunk list
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    """A non-numbered block kept as one chunk (long title, Schedule). Lines are matched exactly."""

    heading: str      # heading stored on the chunk
    first_line: str
    last_line: str


class DocumentCard(BaseModel):
    """Facts about one PDF, hand-written in data/corpus/<id>.toml. See act854.toml for comments."""

    id: str                                   # the card's file name, e.g. "act854"; also the chunk id prefix
    title: str                                # used in the contextual header
    short_name: str                           # filter value, e.g. "ACT854"
    version_date: date
    source_url: str | None = None
    pdf: str                                  # relative to the card file; made absolute on load
    extraction: Literal["plain", "layout"]    # pypdf extraction mode
    page_offset: int = 0                      # printed page = PDF page + offset
    headers: list[str] = []                   # running headers, "{page}" = page number
    footers: list[str] = []
    toc_starts_after: str | None = None       # arrangement of sections; None = outline from the body
    toc_ends_before: str | None = None
    body_starts_after: str                    # exact lines that bound the operative text
    body_ends_before: str
    unit: Literal["Section", "Regulation"]
    definitions: list[str] = []               # units split one chunk per definition
    long_title: Block | None = None
    schedule: Block | None = None

    @property
    def id_prefix(self) -> str:
        """'s' for sections, 'r' for regulations (chunk ids: act854:s23:1, pua220:r2:1)."""
        return self.unit[0].lower()


class OutlineEntry(BaseModel):
    """One row of the outline: a unit the body must contain, in order."""

    part: str | None          # Roman numeral as printed, e.g. "IV"; None if the document has no Parts
    part_title: str | None    # title case, e.g. "Duties and Powers of Chief Executive"
    unit: str                 # "23", "12A"
    heading: str              # whitespace-normalised, exactly as printed


class Line(BaseModel):
    page: int       # printed page number
    text: str       # whitespace-normalised, never empty


class Unit(BaseModel):
    """One detected section or regulation, before it becomes chunk(s)."""

    unit: str
    heading: str
    part: str | None
    part_title: str | None
    lines: list[Line]         # starts with the "N. ..." line; heading and Part lines excluded


class Chunk(BaseModel):
    """Matches the `chunks` table (PRD 8.3), minus `embedding`."""

    id: str
    document_id: str
    part: str | None
    part_title: str | None
    section: str
    subsection: str | None
    heading: str
    page_start: int
    page_end: int
    text: str
    contextual_text: str

"""Data models for ingestion: the document card, lines, units and chunks.

The pipeline, one step per module:
    reader.py   data/llamaparsed/<doc>.md -> DocumentCard + Unit list (+ long title, Schedule)
    chunker.py  Units -> Chunk list
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    """A non-numbered block kept as one chunk (long title, Schedule)."""

    heading: str      # heading stored on the chunk


class DocumentCard(BaseModel):
    """The +++ TOML block at the top of data/llamaparsed/<id>.md. See act854.md."""

    id: str                                   # the file name, e.g. "act854"; also the chunk id prefix
    title: str                                # used in the contextual header
    short_name: str                           # filter value, e.g. "ACT854"
    version_date: date
    source_url: str | None = None
    source: str | None = None                 # where the Markdown came from (note only, not stored)
    unit: Literal["Section", "Regulation"]
    definitions: list[str] = []               # units split one chunk per definition

    @property
    def id_prefix(self) -> str:
        """'s' for sections, 'r' for regulations (chunk ids: act854:s23:1, pua220:r2:1)."""
        return self.unit[0].lower()


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
    """Matches the `chunks` table, minus `embedding`."""

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

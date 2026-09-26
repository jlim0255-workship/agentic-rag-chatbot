"""Data models for ingestion: per-document config, checklist entries, lines and chunks.

Complete: no blanks in this file. Field meanings follow PRD Sections 8.3 and 8.5.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class BlockConfig(BaseModel):
    """A non-numbered block captured as its own chunk (long title, Schedule)."""

    start: str        # regex matching the block's first line
    end_before: str   # regex matching the first line AFTER the block
    heading: str      # heading stored on the chunk


class DocumentConfig(BaseModel):
    """One per document, loaded from data/corpus/<id>.json."""

    id: str                                   # e.g. "act854"; also the chunk id prefix
    short_name: str                           # filter value, e.g. "ACT854"
    title: str                                # used in the contextual header
    source_url: str | None
    version_date: date
    pdf_path: str                             # relative to this config file; made absolute on load
    extraction_mode: Literal["plain", "layout"]  # pypdf mode; "layout" fixes split words in the P.U.(A) PDFs
    body_pages: tuple[int, int]               # physical pages, 1-based, inclusive
    page_offset: int                          # printed page = physical page + offset
    strip_top_patterns: list[str]             # running headers, matched against a page's first line(s)
    strip_bottom_patterns: list[str]          # running footers, matched against a page's last line(s)
    body_start_after: str                     # regex; body starts on the line after the match
    body_end_before: str                      # regex; body ends before the match
    unit_label: Literal["Section", "Regulation"]
    id_prefix: Literal["s", "r"]
    has_parts: bool
    definition_units: list[str]               # units split one chunk per definition
    long_title: BlockConfig | None
    schedule: BlockConfig | None
    checklist_path: str                       # relative to this config file; made absolute on load


class ChecklistEntry(BaseModel):
    unit: str       # "23", "12A"
    heading: str    # whitespace-normalised heading, exactly as printed


class Line(BaseModel):
    page: int       # printed page number
    text: str       # whitespace-normalised, never empty


class Unit(BaseModel):
    """One detected section or regulation, before it becomes chunk(s)."""

    unit: str
    heading: str
    part: str | None          # Roman numeral as printed, e.g. "IV"
    part_title: str | None    # title case, e.g. "Duties and Powers of Chief Executive"
    lines: list[Line]         # starts with the "N. ..." line; heading lines excluded


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

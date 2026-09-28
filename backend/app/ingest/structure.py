"""Step 3, structure: body lines + outline -> units (sections/regulations).

Printed legislation looks like this (Act 854, page 26):

    Part IV                                   <- Part block: only where a new Part starts
    NATIONAL CRITICAL INFORMATION ...            (title in ALL CAPS, may wrap)
    Duty to give notification on cyber        <- heading (may wrap over several lines)
    security incident
    23. (1) If it comes to the knowledge ...  <- unit line: "N. " starts the unit's text
    (2) ...

The outline says which units exist, in order, with their headings and Parts. For each outline row
this module finds the unit line, checks the lines just above it read exactly as the heading, and
cuts the text up to the next unit's heading. Anything the outline does not explain fails the run.
"""
from __future__ import annotations

import re

from .extract import normalise
from .models import Line, OutlineEntry, Unit

UNIT_START = re.compile(r"^(\d+[A-Z]*)\.\s")   # "23. (1) If it comes...", "4. In this Act..."
PART_LINE = re.compile(r"^Part ([IVXLC]+)$")    # "Part IV"; title on the next line(s), ALL CAPS
MAX_HEADING_LINES = 4                            # the longest heading in Act 854 (s18) wraps over 3

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


class OutlineMismatch(Exception):
    """The body does not match the outline. Fails the run."""


def title_case(text: str) -> str:
    """'DUTIES AND POWERS OF CHIEF EXECUTIVE' -> 'Duties and Powers of Chief Executive'."""
    words = text.lower().split()
    return " ".join(w if i and w in SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words))


def is_all_caps(text: str) -> bool:
    return any(c.isalpha() for c in text) and text == text.upper()


def unit_number(line: Line) -> str | None:
    m = UNIT_START.match(line.text)
    return m.group(1) if m else None


def heading_start(body: list[Line], unit_line: int, entry: OutlineEntry, floor: int) -> int:
    """Index of the heading's first line: the 1..MAX_HEADING_LINES lines above the unit line
    that, joined, read exactly as the outline heading (never reaching back to `floor`)."""
    for k in range(1, MAX_HEADING_LINES + 1):
        start = unit_line - k
        if start <= floor:
            break
        if normalise(" ".join(line.text for line in body[start:unit_line])) == entry.heading:
            return start
    above = body[unit_line - 1].text if unit_line > 0 else "<nothing>"
    raise OutlineMismatch(f"unit {entry.unit}: heading {entry.heading!r} not found above its first line "
                          f"(line above is {above!r}, page {body[unit_line].page})")


def cut_part_block(lines: list[Line], entry: OutlineEntry) -> list[Line]:
    """Remove the 'Part X' line and its ALL-CAPS title from the end of `lines`, checking both against
    the outline row that starts the Part. Returns the lines before the block."""
    i = next((i for i, line in enumerate(lines) if line.text == f"Part {entry.part}"), None)
    title = lines[i + 1:] if i is not None else []
    if i is None or not title or not all(is_all_caps(t.text) for t in title):
        raise OutlineMismatch(f"unit {entry.unit} starts Part {entry.part}, but no 'Part {entry.part}' line "
                              f"followed only by an ALL-CAPS title comes before its heading")
    printed = title_case(" ".join(t.text for t in title))
    if printed != entry.part_title:
        raise OutlineMismatch(f"Part {entry.part}: body title {printed!r} differs from outline {entry.part_title!r}")
    return lines[:i]


def detect_units(body: list[Line], outline: list[OutlineEntry]) -> list[Unit]:
    """Find every outline row in the body, in order, and return the units with their lines."""
    # 1. Locate each row's unit line and heading, in order.
    starts: list[tuple[int, int]] = []  # (heading_start, unit_line) per outline row
    prev = -1
    for entry in outline:
        unit_line = next((i for i in range(prev + 1, len(body)) if unit_number(body[i]) == entry.unit), None)
        if unit_line is None:
            raise OutlineMismatch(f"missing unit {entry.unit} ({entry.heading!r})")
        starts.append((heading_start(body, unit_line, entry, prev), unit_line))
        prev = unit_line

    # 2. Independence check: every "N. " line in the body must belong to an outline row.
    claimed = {unit_line for _, unit_line in starts}
    for i, line in enumerate(body):
        if unit_number(line) is not None and i not in claimed:
            raise OutlineMismatch(f"extra unit: line {line.text[:60]!r} on page {line.page} "
                                  "starts like a unit but is not in the outline")

    # 3. Nothing may come before the first heading except the first Part's block.
    lead_in = body[:starts[0][0]] if starts else body
    if outline and outline[0].part:
        lead_in = cut_part_block(lead_in, outline[0])
    if lead_in:
        raise OutlineMismatch(f"unexpected lines before the first unit: {[l.text for l in lead_in][:5]}")

    # 4. A unit's text runs to the next unit's heading, minus the next Part's block if a Part starts there.
    units: list[Unit] = []
    for k, (entry, (_, unit_line)) in enumerate(zip(outline, starts)):
        nxt = outline[k + 1] if k + 1 < len(outline) else None
        lines = body[unit_line:starts[k + 1][0] if nxt else len(body)]
        if nxt and nxt.part != entry.part:
            lines = cut_part_block(lines, nxt)
        stray = next((l for l in lines if PART_LINE.match(l.text)), None)
        if stray:
            raise OutlineMismatch(f"unit {entry.unit}: '{stray.text}' on page {stray.page} is not where the outline "
                                  "starts a Part")
        units.append(Unit(unit=entry.unit, heading=entry.heading, part=entry.part,
                          part_title=entry.part_title, lines=lines))
    return units

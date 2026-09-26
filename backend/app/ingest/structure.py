"""Body lines -> units (sections/regulations), located with the committed ToC checklist."""
from __future__ import annotations

import re

from .extract import normalise
from .models import ChecklistEntry, Line, Unit

UNIT_START = re.compile(r"^(\d+[A-Z]*)\.\s")   # "23. (1) If it comes...", "4. In this Act..."
PART_LINE = re.compile(r"^Part ([IVXLC]+)$")    # "Part IV"; title on the next line(s), ALL CAPS

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


class ChecklistMismatch(Exception):
    """Raised when the body does not match the checklist. Fails the run."""


def title_case(text: str) -> str:
    """'DUTIES AND POWERS OF CHIEF EXECUTIVE' -> 'Duties and Powers of Chief Executive'.

    str.title() would give 'Duties And Powers Of ...'; small words stay lowercase except first.
    """
    words = text.lower().split()
    return " ".join(w if i and w in SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words))


def is_all_caps(text: str) -> bool:
    return any(c.isalpha() for c in text) and text == text.upper()


def unit_number(line: Line) -> str | None:
    m = UNIT_START.match(line.text)
    return m.group(1) if m else None


def find_heading_start(body: list[Line], unit_line: int, heading: str, floor: int) -> int:
    """Walk back from the unit's "N." line, joining lines, until the join equals the heading.

    `floor` is the previous unit's "N." line; the heading can never reach back that far.
    """
    target = normalise(heading)
    for start in range(unit_line - 1, floor, -1):
        joined = normalise(" ".join(line.text for line in body[start:unit_line]))
        if joined == target:
            return start
        if len(joined) > len(target):
            break
    found = body[unit_line - 1].text if unit_line > 0 else "<nothing>"
    raise ChecklistMismatch(
        f"unit {unit_number(body[unit_line])}: heading {heading!r} not found above its first line "
        f"(line above is {found!r}, page {body[unit_line].page})"
    )


def split_part(region: list[Line]) -> tuple[list[Line], tuple[str, str] | None]:
    """Cut a 'Part X' line and its ALL-CAPS title line(s) off the end of a region.

    Returns (lines before the Part, (numeral, title in title case)) or (region, None).
    The Part block must be the last thing in the region: the next unit's heading follows it.
    """
    for i, line in enumerate(region):
        m = PART_LINE.match(line.text)
        if not m:
            continue
        title_lines = region[i + 1:]
        if not title_lines or not all(is_all_caps(t.text) for t in title_lines):
            raise ChecklistMismatch(
                f"'Part {m.group(1)}' on page {line.page} is not followed only by an ALL-CAPS title "
                f"before the next unit's heading: {[t.text for t in title_lines]}"
            )
        title = title_case(" ".join(t.text for t in title_lines))
        return region[:i], (m.group(1), title)
    return region, None


def detect_units(body: list[Line], checklist: list[ChecklistEntry], has_parts: bool) -> list[Unit]:
    """Locate every checklist unit in order and return them with their lines and Part.

    Why checklist-guided: a heading cannot be found by punctuation alone. Most headings have
    no full stop, but "List of seized objects, etc." (s41-45) and s58/s59 end with one, and
    long headings wrap over several lines (s18 spans 3). So for each entry, in order:
      1. find the next line (after the previous unit) where UNIT_START's number == entry.unit
      2. walk backwards from it, joining lines, until the normalised join == entry.heading;
         if you pass the previous unit or overshoot the heading length -> ChecklistMismatch
      3. the unit's lines run from its "N." line to the next unit's first heading line

    Then the independence checks (these are what make it a validation, not just a lookup):
      - any UNIT_START line not claimed by an entry -> ChecklistMismatch ("extra unit")
      - any entry not found -> ChecklistMismatch ("missing unit")

    Parts (only if has_parts):
      - a "Part X" line + ALL-CAPS title line(s) sits between the end of one unit's text and
        the next unit's heading. Cut it out of the previous unit's lines; it applies to the
        units that follow.
      - the first Part (Part I) appears before the first unit's heading, not inside a unit.
      - Part titles can wrap over several ALL-CAPS lines (Part IV spans 3); they are joined.
    """
    # 1-2: locate each entry's "N." line and heading start.
    located: list[tuple[ChecklistEntry, int, int]] = []  # (entry, heading_start, unit_line)
    prev_unit_line = -1
    for entry in checklist:
        unit_line = next((i for i in range(prev_unit_line + 1, len(body))
                          if unit_number(body[i]) == entry.unit), None)
        if unit_line is None:
            raise ChecklistMismatch(f"missing unit {entry.unit} ({entry.heading!r})")
        heading_start = find_heading_start(body, unit_line, entry.heading, prev_unit_line)
        located.append((entry, heading_start, unit_line))
        prev_unit_line = unit_line

    # Independence check: every "N." line in the body must belong to a checklist entry.
    claimed = {unit_line for _, _, unit_line in located}
    for i, line in enumerate(body):
        if unit_number(line) is not None and i not in claimed:
            raise ChecklistMismatch(f"extra unit: line {line.text[:60]!r} on page {line.page} "
                                    "starts like a unit but is not in the checklist")

    # 3: slice lines and assign Parts.
    part: tuple[str, str] | None = None
    lead_in = body[:located[0][1]] if located else body
    if has_parts:
        lead_in, part = split_part(lead_in)
    if lead_in:
        raise ChecklistMismatch(f"unexpected lines before the first unit: {[l.text for l in lead_in][:5]}")

    units: list[Unit] = []
    for k, (entry, _, unit_line) in enumerate(located):
        end = located[k + 1][1] if k + 1 < len(located) else len(body)
        lines, next_part = split_part(body[unit_line:end]) if has_parts else (body[unit_line:end], None)
        units.append(Unit(unit=entry.unit, heading=entry.heading,
                          part=part[0] if part else None, part_title=part[1] if part else None,
                          lines=lines))
        if next_part:
            part = next_part
    return units

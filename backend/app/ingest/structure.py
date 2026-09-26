"""Body lines -> units (sections/regulations), located with the committed ToC checklist."""
from __future__ import annotations

import re

from .models import ChecklistEntry, Line, Unit

UNIT_START = re.compile(r"^(\d+[A-Z]*)\.\s")   # "23. (1) If it comes...", "4. In this Act..."
PART_LINE = re.compile(r"^Part ([IVXLC]+)$")    # "Part IV"; title on the next line(s), ALL CAPS


class ChecklistMismatch(Exception):
    """Raised when the body does not match the checklist. Fails the run."""


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
      - Part IV's title wraps over two ALL-CAPS lines; join them.
    """
    # TODO(Jlim): implement. Verified: on Act 854 this finds sections 1-64 with every
    # heading matching act854.toc.json; on P.U.(A) 220 and 219 it finds regulations 1-3.
    ...

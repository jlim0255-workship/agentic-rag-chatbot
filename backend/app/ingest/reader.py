"""Read one document: data/llamaparsed/<doc>.md -> document card, units and blocks.

The Markdown is LlamaParse output corrected by hand into a fixed format (see act854.md):

    +++                                   document card (TOML), up to the next +++
    title = "..."
    +++
    <!-- page 7 -->                       printed page of the lines that follow
    ## Long title                         block: one chunk
    # Part I: Preliminary                 a Part starts (title in title case)
    ## 1. Short title and commencement    a unit starts: number and heading
    1. (1) This Act may be cited ...      text: one paragraph or (a)/(i) item per line
    ## Schedule: <heading>                block: one chunk

Blank lines are ignored. A paragraph that crosses a page break continues on the line after the
page marker. Anything else that looks like a heading fails the run, and so does unit numbering that
is not consecutive (1, 2, 3, ...; a letter suffix like 12A may follow 12), and so does a line inside
a unit that starts like another unit ("24. ..."): a deleted or mistyped "## N." line would otherwise
merge that unit's text silently into the one before it.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import Block, DocumentCard, Line, Unit

FRONT = re.compile(r"\A\+\+\+\n(.*?)\n\+\+\+\n", re.S)
PAGE = re.compile(r"^<!-- page (\d+) -->$")
PART = re.compile(r"^# Part ([IVXLC]+): (.+)$")
UNIT = re.compile(r"^## (\d+[A-Z]*)\. (.+)$")
UNIT_TEXT_START = re.compile(r"^\d+[A-Z]*\.\s")  # "23. (1) If it comes ..." (only the first line of a unit)
LONG_TITLE = "## Long title"
SCHEDULE = re.compile(r"^## Schedule: (.+)$")


class FormatError(Exception):
    """The Markdown does not follow the format above. Fails the run."""


@dataclass
class Document:
    card: DocumentCard
    units: list[Unit]
    blocks: dict[str, tuple[Block, list[Line]]] = field(default_factory=dict)  # "long_title" / "Schedule"


def read_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    front = FRONT.match(text)
    if not front:
        raise FormatError(f"{path.name}: must start with the document card between two '+++' lines")
    card = DocumentCard(id=path.stem, **tomllib.loads(front.group(1)))

    # Each heading opens a section of the file: (kind, what, lines), kind = "unit" or a block name.
    sections: list[tuple[str, Unit | Block, list[Line]]] = []
    page: int | None = None
    part = part_title = None
    first_line = text[:front.end()].count("\n") + 1
    for n, line in enumerate(text[front.end():].split("\n"), start=first_line):
        where = f"{path.name}:{n}"
        line = line.rstrip()
        if not line:
            continue
        if m := PAGE.match(line):
            page = int(m.group(1))
        elif m := PART.match(line):
            part, part_title = m.groups()
        elif m := UNIT.match(line):
            sections.append(("unit", Unit(unit=m.group(1), heading=m.group(2), part=part,
                                          part_title=part_title, lines=[]), []))
        elif line == LONG_TITLE:
            sections.append(("long_title", Block(heading="Long title"), []))
        elif m := SCHEDULE.match(line):
            sections.append(("Schedule", Block(heading=m.group(1)), []))
        elif line.startswith("#"):
            raise FormatError(f"{where}: unknown heading {line[:60]!r} (use '# Part X: Title', "
                              "'## N. Heading', '## Long title' or '## Schedule: Heading')")
        elif not sections:
            raise FormatError(f"{where}: text before the first heading: {line[:60]!r}")
        elif page is None:
            raise FormatError(f"{where}: text before the first '<!-- page N -->' marker")
        else:
            sections[-1][2].append(Line(page=page, text=line))

    doc = Document(card=card, units=[])
    for kind, what, lines in sections:
        name = f"unit {what.unit}" if kind == "unit" else kind
        if not lines:
            raise FormatError(f"{path.name}: {name} has no text")
        if kind == "unit":
            if not lines[0].text.startswith(f"{what.unit}. "):
                raise FormatError(f"{path.name}: {name} text must start with '{what.unit}. ', "
                                  f"found {lines[0].text[:40]!r}")
            stray = next((l for l in lines[1:] if UNIT_TEXT_START.match(l.text)), None)
            if stray:
                raise FormatError(f"{path.name}: inside {name}, {stray.text[:40]!r} starts like another unit "
                                  "(is its '## N. Heading' line missing?)")
            doc.units.append(what.model_copy(update={"lines": lines}))
        elif kind in doc.blocks:
            raise FormatError(f"{path.name}: more than one {kind} block")
        else:
            doc.blocks[kind] = (what, lines)
    check_numbering([u.unit for u in doc.units], path.name)
    return doc


def check_numbering(units: list[str], name: str) -> None:
    """Units must run 1, 2, 3, ...; '12A' may only follow 12 (or 12A...). Raises FormatError otherwise."""
    prev = 0
    for unit in units:
        n = int(re.match(r"\d+", unit).group())
        suffixed = unit != str(n)
        if not ((n == prev + 1 and not suffixed) or (n == prev and suffixed)):
            expected = f"{prev + 1}" + (f" or {prev}A" if prev else "")
            raise FormatError(f"{name}: unit {unit} follows unit {prev or 'none'}; expected {expected}. "
                              "Is a '## N. Heading' line missing, mistyped or out of order?")
        prev = n

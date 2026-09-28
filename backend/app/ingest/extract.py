"""Step 1, extract: PDF -> list[Line], page by page, running headers and footers stripped.

The whole PDF is extracted; the card's exact lines then say where the ToC, body and blocks are.
`write_lines` saves the result as data/work/<doc>.lines.txt so it can be read next to the PDF.
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentCard, Line

EDGE_LINES = 2  # headers are looked for among the first 2 lines of a page, footers among the last 2


def normalise(text: str) -> str:
    """Collapse all whitespace runs to one space and strip the ends."""
    # Layout mode pads lines with many spaces ("information  infrastructure  or  any").
    return " ".join(text.split())


def template(text: str) -> re.Pattern:
    """'Cyber Security {page}' -> a pattern matching 'Cyber Security 7'."""
    return re.compile(re.escape(text).replace(r"\{page\}", r"\d+"))


def strip_running_lines(lines: list[str], headers: list[str], footers: list[str]) -> list[str]:
    """Drop header lines among the first EDGE_LINES of a page and footer lines among the last."""
    top, bottom = [template(h) for h in headers], [template(f) for f in footers]
    last = len(lines) - EDGE_LINES
    return [text for i, text in enumerate(lines)
            if not ((i < EDGE_LINES and any(p.fullmatch(text) for p in top))
                    or (i >= last and any(p.fullmatch(text) for p in bottom)))]


def load_lines(card: DocumentCard) -> list[Line]:
    """Every non-empty line of the PDF, with printed page numbers.

    Per page, never on the joined document (joining first glues headers into the text,
    e.g. "CYBER SECURITY ACT 20242").
    """
    out: list[Line] = []
    for physical, page in enumerate(PdfReader(card.pdf).pages, start=1):
        texts = [t for t in (normalise(s) for s in page.extract_text(extraction_mode=card.extraction).split("\n")) if t]
        out += [Line(page=physical + card.page_offset, text=t)
                for t in strip_running_lines(texts, card.headers, card.footers)]
    return out


def find_line(lines: list[Line], text: str, start: int = 0) -> int:
    """Index of the first line at or after `start` that reads exactly `text`. Fails loudly otherwise."""
    for i in range(start, len(lines)):
        if lines[i].text == text:
            return i
    raise ValueError(f"no line reads exactly {text!r} (searched from line {start}); "
                     "copy it from data/work/<doc>.lines.txt")


def between(lines: list[Line], after: str, before: str) -> list[Line]:
    """The lines after the line `after` and before the next line `before`."""
    start = find_line(lines, after) + 1
    return lines[start:find_line(lines, before, start)]


def write_lines(lines: list[Line], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"p{line.page:<3}│ {line.text}\n" for line in lines), encoding="utf-8")

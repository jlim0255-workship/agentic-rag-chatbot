"""PDF -> list[Line]: page-by-page extraction with running headers/footers stripped."""
from __future__ import annotations

import re

from pypdf import PdfReader

from .models import DocumentConfig, Line

HEADER_SCAN_LINES = 2  # how many lines at the top/bottom of a page to test against strip patterns


def normalise(text: str) -> str:
    """Collapse all whitespace runs to one space and strip the ends."""
    # Layout mode pads lines with many spaces ("information  infrastructure  or  any"),
    # so runs are collapsed, not just trimmed.
    return " ".join(text.split())


def strip_running_lines(lines: list[str], top: list[str], bottom: list[str]) -> list[str]:
    """Drop header lines among the first HEADER_SCAN_LINES and footer lines among the last."""
    def matches(text: str, patterns: list[str]) -> bool:
        return any(re.fullmatch(p, text) for p in patterns)

    last = len(lines) - HEADER_SCAN_LINES
    return [
        text for i, text in enumerate(lines)
        if not ((i < HEADER_SCAN_LINES and matches(text, top)) or (i >= last and matches(text, bottom)))
    ]


def load_lines(cfg: DocumentConfig) -> list[Line]:
    """Return every non-empty line in cfg.body_pages, with printed page numbers.

    Steps per page (never on the joined document; joining first glues headers into text):
      1. text = reader.pages[i].extract_text(extraction_mode=cfg.extraction_mode)
      2. split into lines, normalise each, drop empty lines
      3. drop lines among the first HEADER_SCAN_LINES that fully match any strip_top_patterns,
         and among the last HEADER_SCAN_LINES that fully match any strip_bottom_patterns
      4. page = physical page (1-based) + cfg.page_offset
    """
    # Checked facts:
    #   - Act 854 (plain): header is line 1, e.g. "Cyber Security 7", "Laws of Malaysia8 Act 854".
    #   - P.U.(A) (layout): "P.U. (A) 220" is line 1; the page number moves to the LAST line.
    #   - Default (plain) mode on P.U.(A) 220 splits words ("nation al", "esti mated");
    #     layout mode does not. That is why extraction_mode is per document.
    reader = PdfReader(cfg.pdf_path)
    first, last = cfg.body_pages
    out: list[Line] = []
    for physical in range(first, last + 1):
        raw = reader.pages[physical - 1].extract_text(extraction_mode=cfg.extraction_mode)
        texts = [t for t in (normalise(s) for s in raw.split("\n")) if t]
        texts = strip_running_lines(texts, cfg.strip_top_patterns, cfg.strip_bottom_patterns)
        out.extend(Line(page=physical + cfg.page_offset, text=t) for t in texts)
    return out


def find_line(lines: list[Line], pattern: str, start: int = 0) -> int:
    """Index of the first line at or after `start` where re.search(pattern) matches.

    Raise ValueError naming the pattern if nothing matches, so a bad config fails loudly.
    """
    regex = re.compile(pattern)
    for i in range(start, len(lines)):
        if regex.search(lines[i].text):
            return i
    raise ValueError(f"no line matches {pattern!r} (searched from line {start})")

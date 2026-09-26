"""PDF -> list[Line]: page-by-page extraction with running headers/footers stripped."""
from __future__ import annotations

import re

from pypdf import PdfReader

from .models import DocumentConfig, Line

HEADER_SCAN_LINES = 2  # how many lines at the top/bottom of a page to test against strip patterns


def normalise(text: str) -> str:
    """Collapse all whitespace runs to one space and strip the ends."""
    # TODO(Jlim): one line. Note: layout mode pads lines with many spaces
    # ("information  infrastructure  or  any"), so this must collapse runs, not just trim.
    ...


def load_lines(cfg: DocumentConfig) -> list[Line]:
    """Return every non-empty line in cfg.body_pages, with printed page numbers.

    Steps per page (never on the joined document; joining first glues headers into text):
      1. text = reader.pages[i].extract_text(extraction_mode=cfg.extraction_mode)
      2. split into lines, normalise each, drop empty lines
      3. drop lines among the first HEADER_SCAN_LINES that fully match any strip_top_patterns,
         and among the last HEADER_SCAN_LINES that fully match any strip_bottom_patterns
      4. page = physical page (1-based) + cfg.page_offset
    """
    # TODO(Jlim): implement.
    # Checked facts:
    #   - Act 854 (plain): header is line 1, e.g. "Cyber Security 7", "Laws of Malaysia8 Act 854".
    #   - P.U.(A) (layout): "P.U. (A) 220" is line 1; the page number moves to the LAST line.
    #   - Default (plain) mode on P.U.(A) 220 splits words ("nation al", "esti mated");
    #     layout mode does not. That is why extraction_mode is per document.
    ...


def find_line(lines: list[Line], pattern: str, start: int = 0) -> int:
    """Index of the first line at or after `start` where re.search(pattern) matches.

    Raise ValueError naming the pattern if nothing matches, so a bad config fails loudly.
    """
    # TODO(Jlim): implement.
    ...

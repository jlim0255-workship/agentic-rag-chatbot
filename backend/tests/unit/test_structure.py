"""Section detection on small hand-built line lists (no PDF needed)."""
from __future__ import annotations

import pytest

from app.ingest.models import ChecklistEntry, Line
from app.ingest.structure import ChecklistMismatch, detect_units, title_case


def lines(*texts: str, page: int = 1) -> list[Line]:
    return [Line(page=page, text=t) for t in texts]


def toc(*pairs: tuple[str, str]) -> list[ChecklistEntry]:
    return [ChecklistEntry(unit=u, heading=h) for u, h in pairs]


BODY = lines(
    "Part I", "PRELIMINARY",
    "Short title",
    "1. (1) This Act may be cited as the Test Act.",
    "(2) It comes into operation on a date.",
    "Part II", "NATIONAL CYBER SECURITY", "COMMITTEE",
    "Designation of national critical information",
    "infrastructure entity",
    "2. The Chief Executive may designate under section",
    "17. of the Act.",  # a wrapped cross-reference that looks like a unit start
)


def test_headings_parts_and_wrapped_lines() -> None:
    body = BODY[:-1] + lines("an entity.")
    units = detect_units(body, toc(("1", "Short title"),
                                   ("2", "Designation of national critical information infrastructure entity")),
                         has_parts=True)
    assert [u.unit for u in units] == ["1", "2"]
    assert (units[0].part, units[0].part_title) == ("I", "Preliminary")
    # Part II block is cut out of unit 1's lines and applies to unit 2; its title wraps over 2 lines
    assert [l.text for l in units[0].lines] == ["1. (1) This Act may be cited as the Test Act.",
                                                "(2) It comes into operation on a date."]
    assert (units[1].part, units[1].part_title) == ("II", "National Cyber Security Committee")
    assert units[1].heading == "Designation of national critical information infrastructure entity"
    assert units[1].lines[0].text.startswith("2. ")


def test_unclaimed_unit_line_is_extra_unit() -> None:
    with pytest.raises(ChecklistMismatch, match="extra unit"):
        detect_units(BODY, toc(("1", "Short title"),
                               ("2", "Designation of national critical information infrastructure entity")),
                     has_parts=True)


def test_missing_unit() -> None:
    body = lines("Short title", "1. Text.")
    with pytest.raises(ChecklistMismatch, match="missing unit 2"):
        detect_units(body, toc(("1", "Short title"), ("2", "Other")), has_parts=False)


def test_heading_differs_from_checklist() -> None:
    body = lines("Short title and commencement", "1. Text.")
    with pytest.raises(ChecklistMismatch, match="heading"):
        detect_units(body, toc(("1", "Short title")), has_parts=False)


def test_out_of_order_is_caught() -> None:
    body = lines("Second", "2. Text.", "First", "1. Text.")
    with pytest.raises(ChecklistMismatch):
        detect_units(body, toc(("1", "First"), ("2", "Second")), has_parts=False)


def test_heading_ending_with_full_stop() -> None:
    body = lines("List of seized objects, etc.", "41. (1) Text.")
    units = detect_units(body, toc(("41", "List of seized objects, etc.")), has_parts=False)
    assert units[0].heading == "List of seized objects, etc."


def test_letter_suffix_unit_number() -> None:
    body = lines("First", "12. Text.", "Inserted", "12A. Text.")
    units = detect_units(body, toc(("12", "First"), ("12A", "Inserted")), has_parts=False)
    assert [u.unit for u in units] == ["12", "12A"]


def test_unexpected_lead_in_text() -> None:
    body = lines("Some stray line", "First", "1. Text.")
    with pytest.raises(ChecklistMismatch, match="before the first unit"):
        detect_units(body, toc(("1", "First")), has_parts=False)


def test_part_line_without_caps_title() -> None:
    body = lines("First", "1. Text.", "Part II", "Second", "2. Text.")
    with pytest.raises(ChecklistMismatch, match="Part II"):
        detect_units(body, toc(("1", "First"), ("2", "Second")), has_parts=True)


def test_title_case_keeps_small_words_lower() -> None:
    assert title_case("DUTIES AND POWERS OF CHIEF EXECUTIVE") == "Duties and Powers of Chief Executive"
    assert title_case("THE GENERAL") == "The General"

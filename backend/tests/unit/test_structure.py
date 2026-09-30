"""Unit detection against an outline, on small hand-built line lists (no PDF needed)."""
from __future__ import annotations

import pytest

from app.ingest.models import Line, OutlineEntry
from app.ingest.structure import OutlineMismatch, detect_units, title_case


def lines(*texts: str, page: int = 1) -> list[Line]:
    return [Line(page=page, text=t) for t in texts]


def row(unit: str, heading: str, part: str | None = None, part_title: str | None = None) -> OutlineEntry:
    return OutlineEntry(part=part, part_title=part_title, unit=unit, heading=heading)


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
OUTLINE = [row("1", "Short title", "I", "Preliminary"),
           row("2", "Designation of national critical information infrastructure entity", "II",
               "National Cyber Security Committee")]


def test_headings_parts_and_wrapped_lines() -> None:
    units = detect_units(BODY[:-1] + lines("an entity."), OUTLINE)
    assert [u.unit for u in units] == ["1", "2"]
    assert (units[0].part, units[0].part_title) == ("I", "Preliminary")
    # Part II block is cut out of unit 1's lines; its title wraps over 2 lines
    assert [l.text for l in units[0].lines] == ["1. (1) This Act may be cited as the Test Act.",
                                                "(2) It comes into operation on a date."]
    assert (units[1].part, units[1].part_title) == ("II", "National Cyber Security Committee")
    assert units[1].lines[0].text.startswith("2. ")


def test_unclaimed_unit_line_is_extra_unit() -> None:
    with pytest.raises(OutlineMismatch, match="extra unit"):
        detect_units(BODY, OUTLINE)


def test_part_title_must_match_outline() -> None:
    wrong = [OUTLINE[0], row(OUTLINE[1].unit, OUTLINE[1].heading, "II", "Some Other Title")]
    with pytest.raises(OutlineMismatch, match="differs from outline"):
        detect_units(BODY[:-1] + lines("an entity."), wrong)


def test_part_line_where_outline_has_no_new_part() -> None:
    body = lines("First", "1. Text.", "Part II", "SECOND PART", "Second", "2. Text.")
    with pytest.raises(OutlineMismatch, match="not where the outline starts a Part"):
        detect_units(body, [row("1", "First"), row("2", "Second")])


def test_missing_unit() -> None:
    with pytest.raises(OutlineMismatch, match="missing unit 2"):
        detect_units(lines("Short title", "1. Text."), [row("1", "Short title"), row("2", "Other")])


def test_heading_differs_from_outline() -> None:
    with pytest.raises(OutlineMismatch, match="heading"):
        detect_units(lines("Short title and commencement", "1. Text."), [row("1", "Short title")])


def test_out_of_order_is_caught() -> None:
    with pytest.raises(OutlineMismatch):
        detect_units(lines("Second", "2. Text.", "First", "1. Text."), [row("1", "First"), row("2", "Second")])


def test_heading_ending_with_full_stop() -> None:
    units = detect_units(lines("List of seized objects, etc.", "41. (1) Text."), [row("41", "List of seized objects, etc.")])
    assert units[0].heading == "List of seized objects, etc."


def test_letter_suffix_unit_number() -> None:
    units = detect_units(lines("First", "12. Text.", "Inserted", "12A. Text."), [row("12", "First"), row("12A", "Inserted")])
    assert [u.unit for u in units] == ["12", "12A"]


def test_unexpected_lead_in_text() -> None:
    with pytest.raises(OutlineMismatch, match="before the first unit"):
        detect_units(lines("Some stray line", "First", "1. Text."), [row("1", "First")])


def test_part_line_without_caps_title() -> None:
    body = lines("First", "1. Text.", "Part II", "Second", "2. Text.")
    with pytest.raises(OutlineMismatch, match="Part II"):
        detect_units(body, [row("1", "First"), row("2", "Second", "II", "Two")])


def test_title_case_keeps_small_words_lower() -> None:
    assert title_case("DUTIES AND POWERS OF CHIEF EXECUTIVE") == "Duties and Powers of Chief Executive"
    assert title_case("THE GENERAL") == "The General"

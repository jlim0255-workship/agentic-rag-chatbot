"""Outline building from an arrangement page or a body, and its Markdown round trip (no PDF needed)."""
from __future__ import annotations

import pytest

from app.ingest.models import Line, OutlineEntry
from app.ingest.outline import from_arrangement, from_body, read_markdown, to_markdown
from app.ingest.structure import OutlineMismatch


def lines(*texts: str) -> list[Line]:
    return [Line(page=3, text=t) for t in texts]


def test_arrangement_parts_wrapped_titles_and_headings() -> None:
    toc = from_arrangement(lines(
        "Part I", "PRELIMINARY", "Section", "1. Short title and commencement",
        "Part IV", "NATIONAL CRITICAL INFORMATION INFRASTRUCTURE SECTOR LEAD",
        "AND NATIONAL CRITICAL INFORMATION INFRASTRUCTURE ENTITY",
        "18. Designation of national critical information infrastructure sector lead as",
        "national critical information infrastructure entity",
        "Section",  # column label repeated on the next page
        "19. Revocation of designation",
    ))
    assert [(e.part, e.unit) for e in toc] == [("I", "1"), ("IV", "18"), ("IV", "19")]
    assert toc[0].part_title == "Preliminary"
    assert toc[1].part_title == ("National Critical Information Infrastructure Sector Lead "
                                 "and National Critical Information Infrastructure Entity")
    assert toc[1].heading == ("Designation of national critical information infrastructure sector lead as "
                              "national critical information infrastructure entity")


def test_arrangement_rejects_text_before_first_entry() -> None:
    with pytest.raises(OutlineMismatch, match="unexpected line"):
        from_arrangement(lines("Stray text", "1. Heading"))


def test_body_outline_follows_numbering_not_cross_references() -> None:
    body = lines("Citation", "1. (1) These regulations may be cited.", "(2) see regulation",
                 "3. of the Act.",  # wrapped cross-reference: 3 is not the next number, so not a unit
                 "Interpretation", "2. In these Regulations, “x” means y.")
    assert [(e.unit, e.heading) for e in from_body(body)] == [("1", "Citation"), ("2", "Interpretation")]


def test_markdown_round_trip_keeps_empty_parts(tmp_path) -> None:
    entries = [OutlineEntry(part="IV", part_title="Code of Practice", unit="23", heading="Duty to notify"),
               OutlineEntry(part=None, part_title=None, unit="2", heading="Interpretation")]
    path = tmp_path / "x.outline.md"
    path.write_text(to_markdown("Title", "a test", entries), encoding="utf-8")
    assert read_markdown(path) == entries

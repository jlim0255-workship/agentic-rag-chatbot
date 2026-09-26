"""Text joining, definition splitting, headers and IDs (no PDF needed)."""
from __future__ import annotations

from datetime import date

import pytest

from app.ingest.chunker import chunk_id, contextual_header, make_chunks, split_definitions, to_text
from app.ingest.extract import find_line, normalise, strip_running_lines
from app.ingest.models import DocumentConfig, Line, Unit


def make_cfg(**overrides) -> DocumentConfig:
    base = dict(
        id="act854", short_name="ACT854", title="Cyber Security Act 2024 (Act 854)", source_url=None,
        version_date=date(2024, 6, 26), pdf_path="x.pdf", extraction_mode="plain", body_pages=(7, 50),
        page_offset=0, strip_top_patterns=[], strip_bottom_patterns=[], body_start_after="x",
        body_end_before="y", unit_label="Section", id_prefix="s", has_parts=True,
        definition_units=["4"], long_title=None, schedule=None, checklist_path="t.json",
    )
    return DocumentConfig(**{**base, **overrides})


def L(text: str, page: int = 8) -> Line:
    return Line(page=page, text=text)


# --- extract helpers ---

def test_normalise_collapses_layout_padding() -> None:
    assert normalise("  information  infrastructure      or  any ") == "information infrastructure or any"


def test_strip_running_lines_only_at_page_edges() -> None:
    page = ["P.U. (A) 220", "Citation", "3", "(a) text", "5"]
    assert strip_running_lines(page, [r"^P\.U\. \(A\) 220$"], [r"^\d+$"]) == ["Citation", "3", "(a) text"]


def test_find_line_fails_loudly() -> None:
    with pytest.raises(ValueError, match="nothing"):
        find_line([L("a")], "nothing")


# --- to_text ---

def test_to_text_rejoins_wraps_and_breaks_before_markers() -> None:
    lines = [L("23. (1) If it comes to the"), L("knowledge of an entity—"), L("(a) first;"),
             L("(iv) roman;"), L("(A) capital;"), L("(2) Any entity which"), L("contravenes.")]
    assert to_text(lines) == ("23. (1) If it comes to the knowledge of an entity—\n(a) first;\n(iv) roman;\n"
                              "(A) capital;\n(2) Any entity which contravenes.")


def test_to_text_keeps_source_verbatim() -> None:
    assert to_text([L("the date and time of the occurence")]) == "the date and time of the occurence"


# --- definitions ---

def test_split_definitions_multiline_drops_intro() -> None:
    unit = Unit(unit="4", heading="Interpretation", part="I", part_title="Preliminary", lines=[
        L("4. In this Act, unless the context otherwise requires—"),
        L("“this Act” includes any subsidiary legislation made under"),
        L("this Act;"),
        L("“Government Entity” means—", page=9),
        L("(a) any ministry; and", page=9),
        L("(b) any local authority;", page=9),
    ])
    defs = split_definitions(unit)
    assert [t for t, _ in defs] == ["this Act", "Government Entity"]
    assert [l.text for l in defs[1][1]] == ["“Government Entity” means—", "(a) any ministry; and",
                                           "(b) any local authority;"]


def test_split_definitions_inline_first_definition() -> None:
    unit = Unit(unit="2", heading="Interpretation", part=None, part_title=None, lines=[
        L("2. In these Regulations, “cyber security risks” means the risks", page=3),
        L("that a vulnerability may be exploited.", page=3),
    ])
    [(term, lines)] = split_definitions(unit)
    assert term == "cyber security risks"
    assert to_text(lines) == "“cyber security risks” means the risks that a vulnerability may be exploited."


# --- header, ids, chunks ---

def test_contextual_header_formats() -> None:
    cfg = make_cfg()
    assert contextual_header(cfg, "Section", "23", "Duty to give notification", "IV", "Code Stuff") == (
        "Cyber Security Act 2024 (Act 854) | Part IV: Code Stuff | Section 23: Duty to give notification")
    assert contextual_header(cfg, "Section", "4", "Interpretation", "I", "Preliminary", term="computer").endswith(
        "| Section 4: Interpretation — “computer”")
    assert contextual_header(cfg, None, None, "Long title", None, None) == "Cyber Security Act 2024 (Act 854) | Long title"
    reg = make_cfg(id="pua220", title="Regs [P.U.(A) 220/2024]", unit_label="Regulation", id_prefix="r")
    assert contextual_header(reg, "Regulation", "2", "Period", None, None) == "Regs [P.U.(A) 220/2024] | Regulation 2: Period"


def test_chunk_id_format() -> None:
    assert chunk_id(make_cfg(), "s23", 1) == "act854:s23:1"
    assert chunk_id(make_cfg(), "long_title", 1) == "act854:long_title:1"


def test_make_chunks_ids_pages_and_definitions() -> None:
    cfg = make_cfg()
    units = [
        Unit(unit="4", heading="Interpretation", part="I", part_title="Preliminary", lines=[
            L("4. In this Act, unless the context otherwise requires—"),
            L("“a” means a;"), L("“b” means", page=8), L("b;", page=9)]),
        Unit(unit="23", heading="Duty", part="IV", part_title="Entities", lines=[
            L("23. (1) Text", page=26), L("(2) More", page=27)]),
    ]
    chunks = make_chunks(cfg, units)
    assert [c.id for c in chunks] == ["act854:s4:1", "act854:s4:2", "act854:s23:1"]
    assert (chunks[1].subsection, chunks[1].page_start, chunks[1].page_end) == ("b", 8, 9)
    assert chunks[2].subsection is None and (chunks[2].page_start, chunks[2].page_end) == (26, 27)
    assert chunks[2].contextual_text.startswith("Cyber Security Act 2024 (Act 854) | Part IV: Entities | Section 23: Duty\n23. (1)")
    assert chunks[2].text == "23. (1) Text\n(2) More"


# --- embed step ---

def test_check_dims_rejects_wrong_size() -> None:
    from app.ingest.run_embed import check_dims
    check_dims([[0.0] * 4, [1.0] * 4], 4)
    with pytest.raises(ValueError, match="EMBED_DIM=4"):
        check_dims([[0.0] * 4, [0.0] * 3], 4)

"""The Markdown reader (data/llamaparsed/<doc>.md format) and its checks."""
from __future__ import annotations

import pytest

from app.ingest.reader import FormatError, check_numbering, read_document

CARD = '''+++
title = "Test Act"
short_name = "TEST"
version_date = 2024-06-26
unit = "Section"
+++
'''

GOOD = CARD + '''
<!-- page 7 -->

## Long title

An Act to test.

# Part I: Preliminary

## 1. Short title

1. (1) This Act may be cited as the Test Act.

(2) It starts
<!-- page 8 -->
on a date.

# Part II: Other Matters

## 2. Interpretation

2. In this Act—

“a” means a;

## Schedule: List of things

[Section 2]

1. Thing
'''


def write(tmp_path, text: str, name: str = "test.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_card_parts_units_pages_and_blocks(tmp_path) -> None:
    doc = read_document(write(tmp_path, GOOD))
    assert (doc.card.id, doc.card.short_name, doc.card.unit) == ("test", "TEST", "Section")

    s1, s2 = doc.units
    assert (s1.unit, s1.heading, s1.part, s1.part_title) == ("1", "Short title", "I", "Preliminary")
    assert [(l.page, l.text) for l in s1.lines] == [
        (7, "1. (1) This Act may be cited as the Test Act."), (7, "(2) It starts"), (8, "on a date.")]
    assert (s2.part, s2.part_title, [l.text for l in s2.lines]) == ("II", "Other Matters", ["2. In this Act—", "“a” means a;"])

    title, lines = doc.blocks["long_title"]
    assert (title.heading, [l.text for l in lines]) == ("Long title", ["An Act to test."])
    schedule, lines = doc.blocks["Schedule"]
    assert (schedule.heading, lines[0].page, [l.text for l in lines]) == ("List of things", 8, ["[Section 2]", "1. Thing"])


def test_windows_line_endings(tmp_path) -> None:
    path = tmp_path / "crlf.md"
    path.write_bytes(GOOD.replace("\n", "\r\n").encode("utf-8"))
    assert len(read_document(path).units) == 2


@pytest.mark.parametrize("text, message", [
    ("## 1. Short title\n", "must start with the document card"),
    (CARD + "<!-- page 7 -->\nstray text\n## 1. Short title\n1. Text\n", "text before the first heading"),
    (CARD + "## 1. Short title\n1. Text\n", "before the first '<!-- page N -->'"),
    (CARD + "<!-- page 7 -->\n# PART I\n## 1. Short title\n1. Text\n", "unknown heading"),
    (CARD + "<!-- page 7 -->\n### 1. Short title\n1. Text\n", "unknown heading"),
    (CARD + "<!-- page 7 -->\n## 1. Short title\n2. Text\n", "must start with '1. '"),
    (CARD + "<!-- page 7 -->\n## 1. Short title\n## 2. Next\n2. Text\n", "unit 1 has no text"),
    (CARD + "<!-- page 7 -->\n## Long title\nA.\n## Long title\nB.\n", "more than one long_title"),
    (CARD + "<!-- page 7 -->\n## 1. A\n1. Text\n## 3. C\n3. Text\n", "unit 3 follows unit 1"),
])
def test_format_errors(tmp_path, text: str, message: str) -> None:
    with pytest.raises(FormatError, match=message):
        read_document(write(tmp_path, text))


def test_numbering_must_be_consecutive() -> None:
    check_numbering(["1", "2", "3"], "doc")
    check_numbering(["1", "2", "2A", "2B", "3"], "doc")  # amended documents insert 2A, 2B after 2
    for bad, message in [(["2", "3"], "unit 2 follows unit none"),          # first unit missing
                         (["1", "3"], "unit 3 follows unit 1"),             # "## 2." deleted: s2 merged into s1
                         (["1", "2", "2"], "unit 2 follows unit 2"),        # number typed twice
                         (["1", "3", "2"], "unit 3 follows unit 1"),        # out of order
                         (["1A", "2"], "unit 1A follows unit none")]:
        with pytest.raises(FormatError, match=message):
            check_numbering(bad, "doc")


def test_missing_heading_line_fails_the_read(tmp_path) -> None:
    text = GOOD.replace("## 2. Interpretation\n", "")  # s2's text would silently join s1
    with pytest.raises(FormatError, match="inside unit 1, '2. In this Act—' starts like another unit"):
        read_document(write(tmp_path, text))

"""Units -> chunks: paragraph text, definition splitting, contextual header, IDs."""
from __future__ import annotations

import re

from .models import Block, Chunk, DocumentCard, Line, Unit

# A new display line starts at a subsection "(2)", paragraph "(a)", subparagraph "(iv)",
# the s14 "(A)"-"(F)" list, a unit start "23.", or a definition opening quote.
BREAK_BEFORE = re.compile(r"^(\((\d+[A-Z]*|[a-z]{1,4}|[A-Z])\)\s|\d+[A-Z]*\.\s|“)")

# s4 intro line: "4. In this Act, unless the context otherwise requires—"
# "drop": the contextual header already says "Section 4: Interpretation — “term”", so repeating
# the intro in all 20 definition chunks adds nothing for retrieval.
DEFINITIONS_INTRO = "drop"

TERM = re.compile(r"“([^”]+)”")


def to_text(lines: list[Line]) -> str:
    """Rejoin wrapped lines into paragraphs; newline only before BREAK_BEFORE lines.

    Everything else is kept verbatim (typos, curly quotes). Known cosmetic edge case:
    a wrapped line that happens to begin with a cross-reference like "(1) accordingly"
    gets its own line. Harmless for retrieval; ignore unless it shows up in spot checks.
    """
    parts: list[str] = []
    for i, line in enumerate(lines):
        if i:
            parts.append("\n" if BREAK_BEFORE.match(line.text) else " ")
        parts.append(line.text)
    return "".join(parts)


def split_definitions(unit: Unit) -> list[tuple[str, list[Line]]]:
    """Split a definitions unit into (defined_term, lines) per definition.

    - Act 854 s4: each definition starts on a line beginning with “ (curly quote).
      20 definitions expected. Nested (a)/(b) lines stay with their definition.
    - P.U.(A) 219 reg 2: the only definition is INLINE on the unit line
      ('2. In these Regulations, “cyber security risks” means ...'), so the first term
      must be found anywhere in the first line, not only at line start.
    - Term = text between the first “ and ” of the definition, e.g. national critical
      information infrastructure.
    - Apply DEFINITIONS_INTRO to the intro sentence.
    """
    first, rest = unit.lines[0], unit.lines[1:]
    quote = first.text.find("“")
    intro: list[Line]
    groups: list[list[Line]] = []
    if quote > 0:  # inline: split the first line into intro + start of the first definition
        intro = [Line(page=first.page, text=first.text[:quote].rstrip())]
        groups.append([Line(page=first.page, text=first.text[quote:])])
    else:
        intro = [first]
    for line in rest:
        if line.text.startswith("“"):
            groups.append([line])
        elif groups:
            groups[-1].append(line)
        else:
            intro.append(line)  # intro sentence wrapped onto a second line

    if not groups:
        raise ValueError(f"unit {unit.unit}: no definitions found")

    out: list[tuple[str, list[Line]]] = []
    for g in groups:
        m = TERM.match(g[0].text)
        if not m:
            raise ValueError(f"unit {unit.unit}: definition without a quoted term: {g[0].text[:60]!r}")
        out.append((m.group(1), intro + g if DEFINITIONS_INTRO == "repeat" else g))
    return out


def contextual_header(card: DocumentCard, unit_label: str | None, unit_no: str | None,
                      heading: str, part: str | None, part_title: str | None,
                      term: str | None = None) -> str:
    """Build the header prepended to text for embedding and BM25 (PRD 8.5).

    Format: "<title> | Part <X>: <Part title, title case> | <Section|Regulation> <n>: <heading>"
      - omit the Part segment when part is None (regulations, long title, Schedule)
      - long title / Schedule: last segment is just the block heading
      - definition chunks: append ' — “<term>”' to the last segment
    Part titles arrive in title case, as written in data/llamaparsed/<doc>.md.
    """
    segments = [card.title]
    if part is not None:
        segments.append(f"Part {part}: {part_title}" if part_title else f"Part {part}")
    last = f"{unit_label} {unit_no}: {heading}" if unit_label and unit_no else heading
    if term:
        last += f" — “{term}”"
    segments.append(last)
    return " | ".join(segments)


def chunk_id(card: DocumentCard, unit_key: str, seq: int) -> str:
    """'<doc>:<prefix><unit>:<n>', e.g. act854:s23:1, pua220:r2:1, act854:s4:7.

    Blocks use their own key without prefix: act854:long_title:1, act854:schedule:1.
    """
    return f"{card.id}:{unit_key}:{seq}"


def _make_chunk(card: DocumentCard, id_: str, lines: list[Line], section: str, subsection: str | None,
                heading: str, part: str | None, part_title: str | None, header: str) -> Chunk:
    text = to_text(lines)
    return Chunk(id=id_, document_id=card.id, part=part, part_title=part_title, section=section,
                 subsection=subsection, heading=heading, page_start=lines[0].page,
                 page_end=lines[-1].page, text=text, contextual_text=f"{header}\n{text}")


def make_block_chunk(card: DocumentCard, lines: list[Line], block: Block,
                     section: str) -> Chunk:
    """One chunk for the long title (section='long_title') or Schedule (section='Schedule')."""
    header = contextual_header(card, None, None, block.heading, None, None)
    return _make_chunk(card, chunk_id(card, section.lower(), 1), lines, section, None,
                       block.heading, None, None, header)


def make_chunks(card: DocumentCard, units: list[Unit]) -> list[Chunk]:
    """One chunk per unit; units in card.definitions become one chunk per definition.

    page_start/page_end come from the first/last Line of each chunk (already printed pages).
    subsection: None for whole units, the defined term for definition chunks.
    """
    chunks: list[Chunk] = []
    for unit in units:
        key = f"{card.id_prefix}{unit.unit}"
        if unit.unit in card.definitions:
            for seq, (term, lines) in enumerate(split_definitions(unit), start=1):
                header = contextual_header(card, card.unit, unit.unit, unit.heading,
                                           unit.part, unit.part_title, term)
                chunks.append(_make_chunk(card, chunk_id(card, key, seq), lines, unit.unit, term,
                                          unit.heading, unit.part, unit.part_title, header))
        else:
            header = contextual_header(card, card.unit, unit.unit, unit.heading,
                                       unit.part, unit.part_title)
            chunks.append(_make_chunk(card, chunk_id(card, key, 1), unit.lines, unit.unit, None,
                                      unit.heading, unit.part, unit.part_title, header))
    return chunks

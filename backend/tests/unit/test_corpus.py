"""The real corpus against its committed outlines (PRD 8.4). Skipped if the PDFs are not in data/raw/."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config import REPO_ROOT
from app.ingest.outline import read_markdown
from app.ingest.run_chunk import chunk_document, load_cards, outline_path, sanity_check

CORPUS = REPO_ROOT / "data" / "corpus"
CARDS = {c.id: c for c in load_cards(CORPUS)}

# sha256 of the chunks as JSON lines (LF endings; all documents, in id order). The chunks the embeddings and the
# evaluation were built on; if it changes, re-embed and re-run the evaluation, then update it here.
CHUNKS_SHA256 = "27b1f16f41ae745644f149927d8fbea89475947931a360fec3761572e7a32118"

pytestmark = pytest.mark.skipif(
    not all(Path(c.pdf).exists() for c in CARDS.values()),
    reason="source PDFs not in data/raw/ (they are not committed)",
)


@pytest.fixture(scope="module")
def chunks() -> dict[str, list]:
    return {doc_id: chunk_document(card, CORPUS) for doc_id, card in CARDS.items()}


def test_every_outline_unit_found_in_order(chunks) -> None:
    for doc_id, card in CARDS.items():
        expected = [e.unit for e in read_markdown(outline_path(CORPUS, card))]
        found = list(dict.fromkeys(c.section for c in chunks[doc_id] if c.section not in ("long_title", "Schedule")))
        assert found == expected, doc_id


def test_chunks_unchanged_snapshot(chunks) -> None:
    dump = "".join(json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for doc in sorted(chunks) for c in chunks[doc])
    assert hashlib.sha256(dump.encode("utf-8")).hexdigest() == CHUNKS_SHA256


def test_chunk_counts(chunks) -> None:
    # Act 854: long title + 63 sections + 20 definitions (s4) + Schedule
    assert {k: len(v) for k, v in chunks.items()} == {"act854": 85, "pua219": 3, "pua220": 3}


def test_sanity_and_non_empty_fields(chunks) -> None:
    for doc_chunks in chunks.values():
        sanity_check(doc_chunks)


def test_ids_are_deterministic() -> None:
    first = [c.model_dump() for c in chunk_document(CARDS["pua220"], CORPUS)]
    assert first == [c.model_dump() for c in chunk_document(CARDS["pua220"], CORPUS)]


def test_act854_key_chunks(chunks) -> None:
    by_id = {c.id: c for c in chunks["act854"]}
    assert list(by_id)[0] == "act854:long_title:1" and list(by_id)[-1] == "act854:schedule:1"
    s23 = by_id["act854:s23:1"]
    assert (s23.part, s23.page_start) == ("IV", 26)
    assert "five hundred thousand ringgit" in s23.text
    assert by_id["act854:s4:20"].subsection == "code of practice"
    assert by_id["act854:s18:1"].heading.startswith("Designation of national critical information infrastructure sector lead")
    assert by_id["act854:s35:1"].part_title == "Cyber Security Incident"
    assert "11. Science, technology and innovation" in by_id["act854:schedule:1"].text
    assert "PERCETAKAN" not in by_id["act854:schedule:1"].text  # printer's text excluded


def test_regulations(chunks) -> None:
    r2 = {c.id: c for c in chunks["pua220"]}["pua220:r2:1"]
    assert (r2.page_start, r2.page_end) == (5, 7)  # printed pages, not physical
    assert "Within six hours" in r2.text and "fourteen days" in r2.text
    assert "P.U. (A) 220" not in r2.text  # running header stripped
    d = {c.id: c for c in chunks["pua219"]}["pua219:r2:1"]
    assert d.subsection == "cyber security risks" and d.text.startswith("“cyber security risks” means")

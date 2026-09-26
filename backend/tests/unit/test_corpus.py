"""The real corpus against its checklists (PRD 8.4). Skipped if the PDFs are not in data/raw/."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import REPO_ROOT
from app.ingest.run_chunk import chunk_document, load_checklist, load_configs, sanity_check

CORPUS = REPO_ROOT / "data" / "corpus"
CONFIGS = {c.id: c for c in load_configs(CORPUS)}

pytestmark = pytest.mark.skipif(
    not all(Path(c.pdf_path).exists() for c in CONFIGS.values()),
    reason="source PDFs not in data/raw/ (they are not committed)",
)


@pytest.fixture(scope="module")
def chunks() -> dict[str, list]:
    return {doc_id: chunk_document(cfg) for doc_id, cfg in CONFIGS.items()}


def test_every_checklist_unit_found_in_order(chunks) -> None:
    for doc_id, cfg in CONFIGS.items():
        expected = [e.unit for e in load_checklist(cfg.checklist_path)]
        found = list(dict.fromkeys(c.section for c in chunks[doc_id] if c.section not in ("long_title", "Schedule")))
        assert found == expected, doc_id


def test_chunk_counts(chunks) -> None:
    # Act 854: long title + 63 sections + 20 definitions (s4) + Schedule
    assert {k: len(v) for k, v in chunks.items()} == {"act854": 85, "pua219": 3, "pua220": 3}


def test_sanity_and_non_empty_fields(chunks) -> None:
    for doc_chunks in chunks.values():
        sanity_check(doc_chunks)


def test_ids_are_deterministic() -> None:
    first = [c.model_dump() for c in chunk_document(CONFIGS["pua220"])]
    assert first == [c.model_dump() for c in chunk_document(CONFIGS["pua220"])]


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

"""replace_chunks against the real database, using a throwaway document id. Run with: pytest -m integration"""
from __future__ import annotations

from datetime import date

import pytest

from app.config import get_settings
from app.ingest import db
from app.ingest.models import Chunk, DocumentConfig

pytestmark = pytest.mark.integration

DOC = "zz_test_doc"


def chunk(n: int, text: str = "text") -> Chunk:
    return Chunk(id=f"{DOC}:s{n}:1", document_id=DOC, part=None, part_title=None, section=str(n),
                 subsection=None, heading="h", page_start=1, page_end=1, text=text,
                 contextual_text=f"header\n{text}")


@pytest.fixture
def conn():
    cfg = DocumentConfig(id=DOC, short_name="ZZTEST", title="Test", source_url=None, version_date=date(2024, 1, 1),
                         pdf_path="x", extraction_mode="plain", body_pages=(1, 1), page_offset=0,
                         strip_top_patterns=[], strip_bottom_patterns=[], body_start_after="x",
                         body_end_before="y", unit_label="Section", id_prefix="s", has_parts=False,
                         definition_units=[], long_title=None, schedule=None, checklist_path="x")
    with db.get_connection() as c:
        db.upsert_document(c, cfg)
        yield c
        c.rollback()  # nothing from the test is kept


def rows(conn):
    return conn.execute("SELECT id, embedding IS NOT NULL FROM chunks WHERE document_id = %s ORDER BY id",
                        [DOC]).fetchall()


def test_replace_chunks_is_idempotent_and_keeps_unchanged_embeddings(conn) -> None:
    dim = get_settings().embed_dim
    db.replace_chunks(conn, DOC, [chunk(1), chunk(2)])
    db.write_embeddings(conn, [(f"{DOC}:s1:1", [0.1] * dim), (f"{DOC}:s2:1", [0.2] * dim)])

    db.replace_chunks(conn, DOC, [chunk(1), chunk(2, "changed")])  # re-run; chunk 2's text changed
    assert rows(conn) == [(f"{DOC}:s1:1", True), (f"{DOC}:s2:1", False)]

    db.replace_chunks(conn, DOC, [chunk(1)])  # chunk 2 no longer produced -> deleted
    assert rows(conn) == [(f"{DOC}:s1:1", True)]

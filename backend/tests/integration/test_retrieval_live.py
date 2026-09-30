"""Hybrid retrieval and one real question end to end. Needs ingested DB + Ollama. Run: pytest -m integration"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import get_settings
from app.db.connection import connect
from app.retrieval.search import Filters, retrieve

pytestmark = pytest.mark.integration


def test_filter_to_one_document_returns_only_that_document() -> None:
    s = get_settings()
    with connect(s.database_url, s.health_timeout_s) as conn:
        results = retrieve(conn, s, "notification of a cyber security incident", Filters(documents=["PUA220"]))
        unfiltered = retrieve(conn, s, "notification of a cyber security incident")
    assert results and {r.short_name for r in results} == {"PUA220"}
    assert len(unfiltered) <= s.top_k_final and len({r.id for r in unfiltered}) == len(unfiltered)


def test_part_filter() -> None:
    s = get_settings()
    with connect(s.database_url, s.health_timeout_s) as conn:
        results = retrieve(conn, s, "penalty", Filters(parts=["IV"]))
    assert results and {r.part for r in results} == {"IV"}


def test_one_real_question_end_to_end() -> None:
    with TestClient(app) as client:
        r = client.post("/chat", json={"question": "What is the penalty if an NCII entity fails to notify "
                                                   "a cyber security incident?", "mode": "traditional"})
    events = [block.split("\n", 1) for block in r.text.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[-4:] == ["answer", "citations", "metrics", "done"], r.text
    citations = json.loads(events[-3][1].removeprefix("data: "))
    assert "act854:s23:1" in {c["chunk_id"] for c in citations}


def test_single_search_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.llm import ollama
    s = get_settings()
    with connect(s.database_url, s.health_timeout_s) as conn:
        vector = retrieve(conn, s, "penalty for not notifying an incident", method="vector")
        wide = retrieve(conn, s, "penalty for not notifying an incident", top_k=10)
        monkeypatch.setattr(ollama, "embed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no embed")))
        bm25 = retrieve(conn, s, "penalty for not notifying an incident", method="bm25")
    assert len(vector) == len(bm25) == s.top_k_final and len(wide) == 10
    assert vector[0].score > vector[-1].score

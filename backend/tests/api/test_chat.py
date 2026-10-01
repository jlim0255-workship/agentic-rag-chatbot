"""/chat SSE stream and /chunks, with Postgres and Ollama replaced by fakes."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.config import Settings, get_settings
from app.llm import ollama
from app.pipeline import core, flow
from app.retrieval import search
from tests.fakes import FakeToolModel, fake_connection

CHUNKS = [search.Retrieved(id="act854:s23:1", short_name="ACT854", section="23", subsection=None,
                           heading="Duty to give notification", part="IV", page_start=26, page_end=26,
                           text="(2) ... fine not exceeding five hundred thousand ringgit ...", score=0.03)]


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.pipeline.nodes.retrieve.connection.connect", fake_connection)
    monkeypatch.setattr(search, "retrieve", lambda conn, s, q, filters=None, timeout_s=None: list(CHUNKS))
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: "The fine is up to RM500,000 [act854:s23:1(2)].")
    main_module.app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(main_module.app)
    main_module.app.dependency_overrides.clear()


def ask(client: TestClient, question: str = "What is the penalty?", mode: str = "traditional"):
    r = client.post("/chat", json={"question": question, "mode": mode})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return parse_sse(r.text)


def test_event_order_and_exactly_one_answer(client: TestClient) -> None:
    events = ask(client)
    names = [n for n, _ in events]
    assert names == ["trace"] * 4 + ["answer", "citations", "metrics", "done"]
    answer = dict(events)["answer"]
    assert "[act854:s23:1(2)]" in answer["text"]
    assert answer["label"] == flow.UNCHECKED_LABEL  # never "verified"
    assert dict(events)["citations"] == [{"chunk_id": "act854:s23:1", "document": "ACT854",
                                         "section": "ACT854 s23", "page": "26"}]
    metrics = dict(events)["metrics"]
    assert metrics["llm_calls"] == 1 and metrics["retries"] == 0
    assert events[-1] == ("done", {"status": "answered"})


def test_retrieve_trace_carries_chunk_ids_and_scores(client: TestClient) -> None:
    retrieve_end = [d for n, d in ask(client) if n == "trace" and d["node"] == "retrieve" and d["status"] == "end"][0]
    assert retrieve_end["data"]["queries"] == [{"query": "What is the penalty?",
                                                "chunks": [{"id": "act854:s23:1", "score": 0.03}]}]


def test_cited_id_not_in_retrieved_is_not_listed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: "Claim [act854:s23:1]. Invented [act854:s99:1].")
    assert [c["chunk_id"] for c in dict(ask(client))["citations"]] == ["act854:s23:1"]


@pytest.mark.parametrize("reply", ["NOT_FOUND", "The sources do not say.\nNOT_FOUND", ""])
def test_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: reply)
    events = dict(ask(client, "What is the capital of France?"))
    assert events["answer"]["text"] == core.NOT_FOUND_TEXT
    assert events["citations"] == []
    assert events["done"] == {"status": "not_found"}


def test_ollama_down_gives_error_event(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_a, **_k):
        raise ollama.OllamaError("Ollama unreachable at http://ollama.test:11434: connection refused")
    monkeypatch.setattr(search, "retrieve", down)
    events = ask(client)
    assert events[-1][0] == "error" and "unreachable" in events[-1][1]["message"]
    assert "answer" not in dict(events)


def test_llm_timeout_ends_with_timeout_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*_a, **_k):
        raise ollama.OllamaTimeout("Ollama call /api/chat took longer than 30s")
    monkeypatch.setattr(ollama, "chat", slow)
    events = ask(client)
    assert "answer" not in dict(events)
    assert [n for n, _ in events][-2:] == ["metrics", "done"]
    assert events[-1] == ("done", {"status": "timeout"})


def test_question_budget_used_up_before_generation(client: TestClient, settings: Settings,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # Retrieval (50 ms) outlasts the whole budget (10 ms). Real sleeps, because Windows'
    # monotonic clock ticks every ~15 ms, so a near-zero budget can look unused.
    def slow_retrieve(*_a, **_k):
        time.sleep(0.05)
        return list(CHUNKS)
    monkeypatch.setattr(search, "retrieve", slow_retrieve)
    main_module.app.dependency_overrides[get_settings] = lambda: settings.model_copy(update={"question_budget_s": 0.01})
    events = ask(client)
    assert "answer" not in dict(events)
    assert events[-1] == ("done", {"status": "timeout"})


def test_agentic_mode_streams_in_prd_order(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.pipeline.nodes.planner.connection.connect", fake_connection)
    monkeypatch.setattr("app.pipeline.nodes.planner.allowed_filters", lambda conn: ([("ACT854", "Act")], []))
    monkeypatch.setattr("app.pipeline.nodes.tool_agent.ChatOllama", lambda **_k: FakeToolModel())
    plan = json.dumps({"sub_queries": [{"query": "penalty for not notifying"}]})
    grade = json.dumps({"sufficient": True, "missing": ""})
    monkeypatch.setattr(ollama, "chat", lambda *a, format=None, **k:
                        ((plan if "sub_queries" in format["properties"] else grade) if format
                         else "The fine is up to RM500,000 [act854:s23:1]."))
    events = ask(client, mode="agentic")
    names = [n for n, _ in events]
    assert set(names[:-4]) == {"trace"} and names[-4:] == ["answer", "citations", "metrics", "done"]
    assert dict(events)["done"] == {"status": "answered"}
    assert dict(events)["metrics"]["llm_calls"] == 4  # plan, tool choice, grade, generate
    assert dict(events)["answer"]["label"] == flow.UNCHECKED_LABEL


def test_empty_question_rejected(client: TestClient) -> None:
    assert client.post("/chat", json={"question": "", "mode": "traditional"}).status_code == 422


# --- /chunks/{id} ---

class FakeCursor:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *_a):
        return self

    def fetchone(self):
        return self.row


def fake_db_returning(row):
    class Conn:
        def cursor(self, **_k):
            return FakeCursor(row)

    @contextmanager
    def connect(*_a, **_k):
        yield Conn()
    return connect


def test_get_chunk(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"id": "act854:s23:1", "document_title": "Cyber Security Act 2024 (Act 854)", "text": "23. (1) ..."}
    monkeypatch.setattr(main_module.connection, "connect", fake_db_returning(row))
    r = client.get("/chunks/act854:s23:1")
    assert r.status_code == 200 and r.json() == row


def test_get_chunk_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.connection, "connect", fake_db_returning(None))
    assert client.get("/chunks/act854:s99:1").status_code == 404


def test_no_chunks_means_not_found_without_llm_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "retrieve", lambda *a, **k: [])
    def must_not_call(*_a, **_k):
        raise AssertionError("the model must not be called without sources")
    monkeypatch.setattr(ollama, "chat", must_not_call)
    events = dict(ask(client))
    assert events["answer"]["text"] == core.NOT_FOUND_TEXT
    assert events["done"] == {"status": "not_found"}
    assert events["metrics"]["llm_calls"] == 0


def test_every_event_is_logged_with_one_request_id(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="app.api.main"):
        ask(client)
    lines = [r.getMessage() for r in caplog.records if r.name == "app.api.main"]
    rid = lines[0].split("]")[0]
    assert all(line.startswith(rid) for line in lines)
    assert "question: What is the penalty?" in lines[0]
    assert any("retrieve end: 1 chunks" in line for line in lines)
    assert any("done: {'status': 'answered'}" in line for line in lines)

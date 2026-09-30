from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import ollama
from app.retrieval import search
from tests.fakes import FakeLLM, FakeToolModel, chunk, fake_connection


@pytest.fixture
def settings() -> Settings:
    """Settings built only from explicit values: no .env file, no real services."""
    return Settings(
        _env_file=None,
        embed_model="qwen3-embedding:0.6b",
        embed_dim=1024,
        database_url="postgresql://test:test@localhost:1/test",
        staff_database_url="postgresql://test:test@localhost:1/staff",
        ollama_base_url="http://ollama.test:11434",
        health_timeout_s=1,
        top_k_vector=20, top_k_bm25=20, top_k_final=5,
        llm_call_timeout_s=30, question_budget_s=60,
    )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch):
    """Fake database, retrieval and LLM for pipeline tests. Retrieval returns chunks by query text
    (`world.by_query`, default: one irrelevant chunk); `world.llm` is a FakeLLM tests can replace."""
    class World:
        by_query: dict[str, list] = {}
        queries: list[str] = []  # every query searched, in order
        llm = FakeLLM()
        vectors: dict[str, list[float]] = {}  # text -> embedding; default [1, 0] (all texts identical)
        tool_calls: list[dict] = []  # what the fake tool_agent model "calls"; default: no tool

    w = World()
    monkeypatch.setattr("app.pipeline.nodes.tool_agent.ChatOllama", lambda **_k: FakeToolModel(w.tool_calls))

    def fake_retrieve(conn, s, query, filters=None, timeout_s=None, **_k):
        w.queries.append(query)
        return list(w.by_query.get(query, [chunk("act854:s99:1", "99")]))

    monkeypatch.setattr("app.pipeline.nodes.retrieve.connection.connect", fake_connection)
    monkeypatch.setattr("app.pipeline.nodes.planner.connection.connect", fake_connection)
    monkeypatch.setattr("app.pipeline.nodes.planner.allowed_filters",
                        lambda conn: ([("ACT854", "Act"), ("PUA220", "Regs")], [("IV", "Entities")]))
    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: w.llm(*a, **k))
    monkeypatch.setattr(ollama, "embed", lambda url, model, texts, timeout: [w.vectors.get(t, [1.0, 0.0]) for t in texts])
    return w

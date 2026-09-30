"""Staff-directory tool (new_prd.md): tool_agent → tools_condition → ToolNode → contact list appended by code."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from tests.fakes import FakeLLM, chunk, final, grade, run, trace_nodes

ROWS = [("Aisha Tan", "Authorised person", "aisha.tan@example.com", "+60 3-0000 0010", "24/7")]


@pytest.fixture
def staff_db(monkeypatch, settings):
    """lookup_staff's SQL, faked: records the topics asked for and returns ROWS."""
    asked: list = []

    class Conn:
        def execute(self, sql, params):
            asked.append(params[0])
            return type("R", (), {"fetchall": lambda _self: ROWS})()

    @contextmanager
    def connect(*_a, **_k):
        yield Conn()

    monkeypatch.setattr("app.pipeline.tools.connect", connect)
    monkeypatch.setattr("app.pipeline.tools.get_settings", lambda: settings)
    return asked


def setup(world, tool_calls) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    world.llm = FakeLLM(grades=[grade(["act854:s23:1"])])
    world.tool_calls = tool_calls


def test_tool_call_appends_contacts_but_writer_never_sees_them(settings, world, staff_db) -> None:
    prompts = []
    setup(world, [{"name": "lookup_staff", "args": {"topics": ["ncii_notification"]}, "id": "c1"}])
    llm = world.llm
    world.llm = lambda *a, **k: prompts.append(a[2]) or llm(*a, **k)
    events = run(settings)
    text = final(events)["answer"]["text"]
    assert text.startswith("The fine is RM500,000") and "**Who to contact**" in text and "Aisha Tan" in text
    assert staff_db == [["ncii_notification"]]
    assert not any("Aisha" in p for p in prompts)  # option B: code appends, the writer never sees staff data
    assert "tools:end" in trace_nodes(events)


def test_no_tool_call_means_no_lookup(settings, world, staff_db) -> None:
    setup(world, [])
    text = final(run(settings))["answer"]["text"]
    assert staff_db == [] and "Who to contact" not in text


def test_malformed_tool_call_still_answers(settings, world, staff_db) -> None:
    setup(world, [{"name": "lookup_staff", "args": {"topic": "legal"}, "id": "c1"}])  # wrong argument name
    out = final(run(settings))
    assert out["done"] == {"status": "answered"} and "Who to contact" not in out["answer"]["text"]
    assert staff_db == []


def test_traditional_mode_has_no_tools(settings, world, staff_db) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    events = run(settings, mode="traditional")
    assert not any(n.startswith(("tool_agent", "tools")) for n in trace_nodes(events))

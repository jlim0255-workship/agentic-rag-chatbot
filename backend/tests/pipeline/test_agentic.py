"""Agentic routing with the LLM and retrieval faked: graph routing and node behaviour."""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm import ollama
from app.llm.structured import call_structured, strip_fence
from app.pipeline import flow
from app.pipeline.core import Ctx, SubQuery
from app.pipeline.nodes import generate
from app.pipeline.nodes.grade import GradeOutput
from app.pipeline.nodes.not_found import not_found_node
from tests.fakes import FakeLLM, chunk, final, grade, run, trace_nodes


# --- routing ---

def test_enough_evidence_goes_straight_to_generate(settings: Settings, world) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1"), chunk("act854:s99:1", "99")]}
    world.llm = FakeLLM(grades=[grade(["act854:s23:1"], ["act854:s99:1"])])
    events = run(settings)
    assert trace_nodes(events) == ["planner:start", "planner:end", "retrieve:start", "retrieve:end", "grade:start", "grade:end",
                                   "generate:start", "generate:end"]
    assert [n for n, _ in events if n != "trace"] == ["answer", "citations", "metrics", "done"]
    out = final(events)
    assert out["done"] == {"status": "answered"}
    assert out["metrics"]["llm_calls"] == 3 and out["metrics"]["retries"] == 0
    gen_start = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    # steer, don't delete: verified chunk first, the other retrieved chunk still reaches the writer
    assert gen_start["data"]["chunks"] == ["act854:s23:1", "act854:s99:1"]
    assert out["answer"]["label"] == flow.UNCHECKED_LABEL  # no answer check runs; never "verified"


def test_insufficient_grade_triggers_rewrite_then_retrieve_with_new_query(settings: Settings, world) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s99:1", "99")],
                      "offence of contravening the notification duty": [chunk("act854:s23:1"), chunk("act854:s24:1", "24")]}
    world.llm = FakeLLM(grades=[grade([], ["act854:s99:1"], missing="the penalty section"),
                                grade(["act854:s23:1"])],
                        rewrites=[{"expanded": "offence of contravening the notification duty"}])
    events = run(settings)
    # retry 1 = expansion: one new query, searched instead of the old one
    assert world.queries == ["What is the penalty?", "offence of contravening the notification duty"]
    retrieve_ends = [d for n, d in events if n == "trace" and d["node"] == "retrieve" and d["status"] == "end"]
    assert [c["id"] for c in retrieve_ends[1]["data"]["queries"][0]["chunks"]] == ["act854:s23:1", "act854:s24:1"]
    assert world.llm.calls == ["plan", "grade", "rewrite", "grade", "generate"]
    out = final(events)
    assert out["done"] == {"status": "answered"} and out["metrics"]["retries"] == 1
    rewrite_end = [d for n, d in events if n == "trace" and d["node"] == "rewrite" and d["status"] == "end"][0]
    assert rewrite_end["data"]["rewrites"] == [{
        "technique": "expansion", "from": "What is the penalty?",
        "to": "offence of contravening the notification duty",
        "gave_up": False, "filters_removed": {"documents": [], "parts": []}}]


def test_loop_stops_at_max_retries_and_never_calls_generate(settings: Settings, world) -> None:
    assert settings.max_retries == 2
    world.llm = FakeLLM(grades=[grade([], ["act854:s99:1"])] * 3,
                        rewrites=[{"expanded": "second formal try"}, {"step_back": "broader question"}])
    events = run(settings, "Who won the 2022 World Cup?")
    assert world.llm.calls == ["plan", "grade", "rewrite", "grade", "rewrite", "grade"]  # no generate
    out = final(events)
    assert out["answer"]["text"] == "Not found in the provided documents."
    assert out["done"] == {"status": "not_found"}
    assert out["metrics"]["retries"] == 2 and out["metrics"]["llm_calls"] == 6
    techniques = [d["data"]["rewrites"][0]["technique"] for n, d in events
                  if n == "trace" and d["node"] == "rewrite" and d["status"] == "end"]
    assert techniques == ["expansion", "step-back"]  # escalates one technique per retry


def test_repeated_rewrite_is_rejected_once_then_given_up(settings: Settings, world) -> None:
    world.llm = FakeLLM(grades=[grade([], ["act854:s99:1"])],
                        rewrites=[{"expanded": "What is the  PENALTY?"}, {"expanded": "what is the penalty?"}])
    events = run(settings)
    assert world.llm.calls == ["plan", "grade", "rewrite", "rewrite"]
    assert world.queries == ["What is the penalty?"]  # nothing new to retrieve
    assert final(events)["done"] == {"status": "not_found"}


def test_grade_invalid_twice_falls_back_to_all_relevant(settings: Settings, world) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    world.llm = FakeLLM(grades=["not json", '{"needs": "wrong"}'])
    events = run(settings)
    assert world.llm.calls == ["plan", "grade", "grade", "generate"]
    grade_end = [d for n, d in events if n == "trace" and d["node"] == "grade" and d["status"] == "end"][0]
    assert grade_end["data"]["sub_queries"][0]["fallback"] is True
    assert final(events)["done"] == {"status": "answered"}


def test_llm_timeout_ends_with_timeout(settings: Settings, world, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*_a, **_k):
        raise ollama.OllamaTimeout("took too long")
    monkeypatch.setattr(ollama, "chat", slow)
    events = run(settings)
    assert "answer" not in final(events)
    assert events[-1] == ("done", {"status": "timeout"})


# --- partial evidence (two sub-queries, as the planner creates them) ---

def test_partial_evidence_asks_only_the_answered_part_and_code_names_the_rest(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(Ctx, "emit", staticmethod(lambda *a, **k: None))
    subs = [SubQuery(original="reporting deadline", query="q1", answered=True, partly=True),
            SubQuery(original="licence fee amount", query="q2", answered=False)]
    assert not_found_node({"sub_queries": subs}, Ctx(settings)) == {"status": "partial"}
    assert flow.after_not_found({"status": "partial"}) == "generate"
    prompts = []
    monkeypatch.setattr(ollama, "chat", lambda url, model, prompt, *a, **k: prompts.append(prompt) or
                        "Within six hours [pua220:r2:1].")
    s = chunk("pua220:r2:1", "2")
    out = generate.generate_node({"question": "deadline and fee?", "sub_queries": subs, "status": "partial",
                                  "relevant": {s.id: s}, "supporting": {s.id: s}, "llm_calls": 0}, Ctx(settings))
    assert "Question: reporting deadline\n" in prompts[0] and "fee" not in prompts[0].split("Question:")[1]
    assert out["answer"] == ("Within six hours [pua220:r2:1].\n\n"
                             "Not covered by the provided documents: licence fee amount")


def test_retrieved_evidence_still_reaches_the_writer_on_the_partial_path(settings: Settings, world) -> None:
    # Two sub-queries: the first is judged sufficient, the second never is; s23 still reaches the writer.
    s35, s23 = chunk("act854:s35:1", "35"), chunk("act854:s23:1")
    no = {"sufficient": False, "missing": "penalty"}
    world.by_query = {"q": [s35], "q2": [s35, s23], "e1": [s23], "s1": [s23]}
    world.llm = FakeLLM(plans=[{"sub_queries": [{"query": "q"}, {"query": "q2"}]}],
                        grades=[{"sufficient": True}, no, no, no],
                        rewrites=[{"expanded": "e1"}, {"step_back": "s1"}])
    events = run(settings, "two part question?")
    gen = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    assert gen["data"]["chunks"][0] == "act854:s35:1" and "act854:s23:1" in gen["data"]["chunks"]
    assert final(events)["done"] == {"status": "partial"}


def test_known_tradeoff_a_single_question_never_judged_sufficient_is_not_found(settings: Settings, world) -> None:
    # A "no" keeps nothing: after the retries, no partial answer from s23.
    world.by_query = {q: [chunk("act854:s23:1")] for q in ("What is the penalty?", "e", "s")}
    no = {"sufficient": False, "missing": "penalty for ignoring a directive"}
    world.llm = FakeLLM(grades=[no] * 3, rewrites=[{"expanded": "e"}, {"step_back": "s"}])
    events = run(settings)
    assert "generate" not in world.llm.calls
    assert final(events)["done"] == {"status": "not_found"}


def test_no_evidence_at_all_is_not_found(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(Ctx, "emit", staticmethod(lambda *a, **k: None))
    subs = [SubQuery(original="a", query="a"), SubQuery(original="b", query="b")]
    assert not_found_node({"sub_queries": subs}, Ctx(settings)) == {"status": "not_found"}
    assert flow.after_not_found({"status": "not_found"}) == "__end__"


# --- structured output helper ---

def test_structured_retries_once_then_succeeds(settings: Settings, monkeypatch) -> None:
    replies = iter(["{broken", json.dumps(grade(["a:s1:1"]))])
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: next(replies))
    out, calls = call_structured("p", GradeOutput, settings, 5)
    assert calls == 2 and out.sufficient is True


def test_structured_gives_none_after_two_invalid(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: "nope")
    assert call_structured("p", GradeOutput, settings, 5) == (None, 2)


def test_structured_prompt_spells_out_the_schema_sent_as_format(settings: Settings, monkeypatch) -> None:
    """Ollama Cloud ignores `format`, so the prompt must carry the same schema (incl. an override)."""
    seen = {}
    def chat(base_url, model, prompt, timeout_s, num_ctx, format=None):
        seen.update(prompt=prompt, format=format)
        return json.dumps(grade(["a:s1:1"]))
    monkeypatch.setattr(ollama, "chat", chat)
    override = {**GradeOutput.model_json_schema(), "title": "Override"}
    call_structured("p", GradeOutput, settings, 5, json_schema=override)
    assert seen["format"] == override
    assert seen["prompt"].startswith("p\n\n") and seen["prompt"].endswith(json.dumps(override))


def test_strip_fence_unwraps_json_only() -> None:
    assert strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fence('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fence('{"a": 1}') == '{"a": 1}'
    assert strip_fence("Yes, see ```x```.") == "Yes, see ```x```."  # prose stays prose (-> invalid -> retry)

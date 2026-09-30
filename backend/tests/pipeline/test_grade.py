"""Yes/no sufficiency grader: yes -> generate, no -> rewrite aimed at `missing`."""
from __future__ import annotations

from tests.fakes import FakeLLM, chunk, final, run

S23 = chunk("act854:s23:1")


def test_sufficient_goes_to_generate_with_retrieved_chunks_first(settings, world) -> None:
    world.by_query = {"What is the penalty?": [S23]}
    world.llm = FakeLLM(grades=[{"sufficient": True, "missing": ""}])
    events = run(settings)
    assert world.llm.calls == ["plan", "grade", "generate", "check"]
    gen = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    assert gen["data"]["chunks"] == ["act854:s23:1"]
    assert final(events)["done"] == {"status": "answered"}


def test_insufficient_triggers_rewrite_aimed_at_missing(settings, world) -> None:
    world.by_query = {"What is the penalty?": [S23], "who is notified": [S23], "notify the Chief Executive": [S23]}
    world.llm = FakeLLM(grades=[{"sufficient": False, "missing": "who must be notified"},
                                {"sufficient": True, "missing": ""}],
                        rewrites=[{"paraphrase": "who is notified", "expanded": "notify the Chief Executive"}])
    events = run(settings)
    grade_end = [d for n, d in events if n == "trace" and d["node"] == "grade" and d["status"] == "end"][0]
    assert grade_end["data"]["sub_queries"][0]["missing"] == "who must be notified"
    assert final(events)["metrics"]["retries"] == 1 and final(events)["done"] == {"status": "answered"}


def test_insufficient_without_a_reason_still_rewrites(settings, world) -> None:
    world.by_query = {"What is the penalty?": [S23], "p": [S23], "e": [S23]}
    world.llm = FakeLLM(grades=[{"sufficient": False}, {"sufficient": True}],
                        rewrites=[{"paraphrase": "p", "expanded": "e"}])
    events = run(settings)
    grade_end = [d for n, d in events if n == "trace" and d["node"] == "grade" and d["status"] == "end"][0]
    assert grade_end["data"]["sub_queries"][0]["missing"] == "not stated"
    assert final(events)["done"] == {"status": "answered"}


def test_retrieved_evidence_still_reaches_the_writer_on_the_partial_path(settings, world) -> None:
    # Two sub-queries: the first is judged sufficient, the second never is; s23 still reaches the writer.
    s35 = chunk("act854:s35:1", "35")
    no = {"sufficient": False, "missing": "penalty"}
    world.by_query = {"q": [s35], "q2": [s35, S23], "p1": [S23], "e1": [S23], "s1": [S23]}
    world.llm = FakeLLM(plans=[{"sub_queries": [{"query": "q"}, {"query": "q2"}]}],
                        grades=[{"sufficient": True}, no, no, no],
                        rewrites=[{"paraphrase": "p1", "expanded": "e1"}, {"step_back": "s1"}])
    events = run(settings, "two part question?")
    gen = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    assert gen["data"]["chunks"][0] == "act854:s35:1" and "act854:s23:1" in gen["data"]["chunks"]
    assert final(events)["done"] == {"status": "partial"}


def test_known_tradeoff_a_single_question_never_judged_sufficient_is_not_found(settings, world) -> None:
    # Unlike the quote grader, a "no" keeps nothing: after the retries, no partial answer from s23.
    world.by_query = {q: [S23] for q in ("What is the penalty?", "p", "e", "s")}
    no = {"sufficient": False, "missing": "penalty for ignoring a directive"}
    world.llm = FakeLLM(grades=[no] * 3, rewrites=[{"paraphrase": "p", "expanded": "e"}, {"step_back": "s"}])
    events = run(settings)
    assert "generate" not in world.llm.calls
    assert final(events)["done"] == {"status": "not_found"}

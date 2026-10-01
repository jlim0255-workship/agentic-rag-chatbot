"""Planner node: splitting, filter validation, fallback (PRD 10.2), and multi-sub-query graph paths."""
from __future__ import annotations

import pytest

from app.pipeline.nodes.planner import PlanItem, PlanOutput, clean_plan
from tests.fakes import FakeLLM, chunk, final, grade, run

DOCS, PARTS = {"ACT854", "PUA220", "PUA219"}, {"I", "IV", "IX"}


def plan(*items: PlanItem) -> PlanOutput:
    return PlanOutput(sub_queries=list(items))


def test_multi_part_question_keeps_each_sub_query_and_valid_filters() -> None:
    subs, dropped = clean_plan(plan(PlanItem(query="penalty for not notifying", documents=["ACT854"], parts=["Part IV", "9"]),
                                    PlanItem(query="deadline for initial particulars", documents=["pua220"])),
                               "q", DOCS, PARTS, 3)
    assert [s.query for s in subs] == ["penalty for not notifying", "deadline for initial particulars"]
    assert subs[0].filters.documents == ["ACT854"] and subs[0].filters.parts == ["IV", "IX"]  # "Part IV" and "9" normalised
    assert subs[1].filters.documents == ["PUA220"]
    assert dropped == []


def test_invented_filter_values_are_dropped() -> None:
    subs, dropped = clean_plan(plan(PlanItem(query="x", documents=["ACT854", "PDPA2010"], parts=["XII", "IV"])),
                               "q", DOCS, PARTS, 3)
    assert subs[0].filters.documents == ["ACT854"] and subs[0].filters.parts == ["IV"]
    assert dropped == ["PDPA2010", "XII"]


def test_more_than_max_sub_queries_is_truncated() -> None:
    subs, _ = clean_plan(plan(*[PlanItem(query=f"q{i}") for i in range(5)]), "q", DOCS, PARTS, 3)
    assert [s.query for s in subs] == ["q0", "q1", "q2"]


@pytest.mark.parametrize("bad", [None, plan(), plan(PlanItem(query="   "))])
def test_invalid_or_empty_plan_falls_back_to_the_question(bad) -> None:
    subs, dropped = clean_plan(bad, "What is the penalty?", DOCS, PARTS, 3)
    assert [(s.query, s.filters.documents, s.filters.parts) for s in subs] == [("What is the penalty?", [], [])]


# --- graph paths with several sub-queries ---

def test_each_sub_query_is_retrieved_with_its_filters(settings, world, monkeypatch) -> None:
    seen = []
    import app.retrieval.search as search
    monkeypatch.setattr(search, "retrieve", lambda conn, s, q, filters=None, **k:
                        seen.append((q, filters.documents)) or [chunk("act854:s23:1")])
    world.llm = FakeLLM(plans=[{"sub_queries": [{"query": "a", "documents": ["ACT854"]},
                                                {"query": "b", "documents": ["MADEUP"]}]}],
                        grades=[grade(["act854:s23:1"]), grade(["act854:s23:1"])])
    events = run(settings, "a and b?")
    assert seen == [("a", ["ACT854"]), ("b", [])]  # invented document dropped
    planner_end = [d for n, d in events if n == "trace" and d["node"] == "planner" and d["status"] == "end"][0]
    assert planner_end["data"]["dropped_filters"] == ["MADEUP"]
    assert final(events)["done"] == {"status": "answered"}


def test_one_sub_query_found_other_not_gives_partial_answer(settings, world) -> None:
    world.by_query = {"reporting deadline": [chunk("pua220:r2:1", "2")], "licence fee amount": [chunk("act854:s30:1", "30")]}
    world.llm = FakeLLM(
        plans=[{"sub_queries": [{"query": "reporting deadline"}, {"query": "licence fee amount"}]}],
        grades=[grade(["pua220:r2:1"]), grade([], ["act854:s30:1"], "the fee amount"),   # attempt 1
                grade([], ["act854:s30:1"]), grade([], ["act854:s30:1"])],               # retries 1 and 2
        rewrites=[{"paraphrase": "prescribed licence fee", "expanded": "fee payable for licence"},
                  {"step_back": "fees under the Act"}],
        answer="Initial particulars are due within six hours [pua220:r2:1]. The fee is not covered.")
    events = run(settings, "What is the deadline and what is the fee?")
    gen = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    assert gen["data"]["unanswered"] == ["licence fee amount"]
    assert gen["data"]["chunks"][0] == "pua220:r2:1" and "act854:s30:1" in gen["data"]["chunks"]  # verified first
    out = final(events)
    assert out["done"] == {"status": "partial"} and out["metrics"]["retries"] == 2
    # only the unanswered sub-query was rewritten and re-retrieved
    assert world.queries == ["reporting deadline", "licence fee amount", "prescribed licence fee",
                             "fee payable for licence", "fees under the Act"]


def test_rewrite_removes_the_sub_query_filters(settings, world, monkeypatch) -> None:
    seen = []
    import app.retrieval.search as search
    monkeypatch.setattr(search, "retrieve", lambda conn, s, q, filters=None, **k:
                        seen.append(filters.parts) or [chunk("act854:s23:1")])
    world.llm = FakeLLM(plans=[{"sub_queries": [{"query": "a", "parts": ["IV"]}]}],
                        grades=[grade([], ["act854:s23:1"]), grade(["act854:s23:1"])],
                        rewrites=[{"paraphrase": "a better", "expanded": "a formal"}])
    run(settings, "a?")
    assert seen == [["IV"], [], []]  # both rewritten variants searched without the filter


# --- routing (planner output decides the path) ---

def test_out_of_scope_gets_fixed_reply_without_retrieval_or_writer(settings, world) -> None:
    world.llm = FakeLLM(plans=[{"route": "out_of_scope", "sub_queries": []}])
    events = run(settings, "What is 1+1?")
    out = final(events)
    assert out["done"] == {"status": "out_of_scope"}
    assert out["answer"]["label"] == "Out of scope: not answered" and "only answer questions" in out["answer"]["text"]
    assert world.llm.calls == ["plan"] and world.queries == []


def test_writer_not_found_gets_the_not_found_label_in_agentic_mode(settings, world) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    world.llm = FakeLLM(grades=[grade(["act854:s23:1"])], answer="NOT_FOUND")
    out = final(run(settings))
    assert out["done"] == {"status": "not_found"}
    assert out["answer"]["label"] == "Nothing to check: no answer was found in the documents"


def test_invalid_plan_falls_back_to_complex_route(settings, world) -> None:
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    world.llm = FakeLLM(plans=["not json", "still not json"], grades=[grade(["act854:s23:1"])])
    run(settings)
    assert world.llm.calls == ["plan", "plan", "grade", "generate"]

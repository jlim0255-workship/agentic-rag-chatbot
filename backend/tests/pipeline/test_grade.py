"""Needs-coverage grader: code verifies every quote; the grader steers but never removes evidence."""
from __future__ import annotations

from app.pipeline.nodes.grade import Need, schema_for, verify
from tests.fakes import FakeLLM, chunk, final, run

S23 = chunk("act854:s23:1").model_copy(update={"text": "(2) ... liable to a fine not exceeding five hundred thousand ringgit ..."})


def test_verbatim_quote_is_covered_ignoring_case_and_spacing() -> None:
    [v] = verify([Need(need="penalty", chunk_id="act854:s23:1", quote="Fine  not exceeding\nfive hundred")], [S23])
    assert v["covered"] is True


def test_invented_quote_is_rejected() -> None:
    [v] = verify([Need(need="penalty", chunk_id="act854:s23:1", quote="fine of one million ringgit")], [S23])
    assert v["covered"] is False


def test_quote_from_a_chunk_that_was_not_retrieved_is_rejected() -> None:
    [v] = verify([Need(need="penalty", chunk_id="act854:s99:1", quote="five hundred thousand ringgit")], [S23])
    assert v["covered"] is False


def test_trivially_short_quote_is_rejected() -> None:
    [v] = verify([Need(need="penalty", chunk_id="act854:s23:1", quote="fine")], [S23])
    assert v["covered"] is False


def test_schema_restricts_chunk_id_to_retrieved_ids() -> None:
    prop = schema_for(["act854:s23:1", "pua220:r2:1"])["$defs"]["Need"]["properties"]["chunk_id"]
    assert prop == {"anyOf": [{"enum": ["act854:s23:1", "pua220:r2:1"]}, {"type": "null"}]}


def test_uncovered_need_triggers_rewrite_aimed_at_it(settings, world) -> None:
    world.by_query = {"What is the penalty?": [S23], "who is notified": [S23], "notify the Chief Executive": [S23]}
    world.llm = FakeLLM(
        grades=[{"needs": [{"need": "penalty amount", "chunk_id": "act854:s23:1", "quote": "five hundred thousand ringgit"},
                           {"need": "who must be notified", "chunk_id": None, "quote": None}]},
                {"needs": [{"need": "who", "chunk_id": "act854:s23:1", "quote": "five hundred thousand ringgit"}]}],
        rewrites=[{"paraphrase": "who is notified", "expanded": "notify the Chief Executive"}])
    events = run(settings)
    rewrite_prompt_missing = [d for n, d in events if n == "trace" and d["node"] == "grade" and d["status"] == "end"][0]
    assert rewrite_prompt_missing["data"]["sub_queries"][0]["missing"] == "who must be notified"
    assert final(events)["metrics"]["retries"] == 1 and final(events)["done"] == {"status": "answered"}


def test_grader_mistakes_cannot_hide_evidence_from_the_writer(settings, world) -> None:
    # Grader quotes s35 but never s23 across all attempts; s23 is still given to the writer (partial path).
    s35 = chunk("act854:s35:1", "35")
    world.by_query = {"q": [s35, S23]}
    miss = {"needs": [{"need": "a", "chunk_id": "act854:s35:1", "quote": "text of act854:s35:1"},
                      {"need": "penalty", "chunk_id": None, "quote": None}]}
    world.llm = FakeLLM(plans=[{"sub_queries": [{"query": "q"}, {"query": "q2"}]}],
                        grades=[{"needs": [{"need": "x", "chunk_id": "act854:s35:1", "quote": "text of act854:s35:1"}]},
                                miss, miss, miss],
                        rewrites=[{"paraphrase": "p1", "expanded": "e1"}, {"step_back": "s1"}])
    world.by_query.update({"q2": [s35, S23], "p1": [S23], "e1": [S23], "s1": [S23]})
    events = run(settings, "two part question?")
    gen = [d for n, d in events if n == "trace" and d["node"] == "generate" and d["status"] == "start"][0]
    assert "act854:s23:1" in gen["data"]["chunks"]
    assert final(events)["done"] == {"status": "partial"}


def test_schema_caps_the_number_of_needs() -> None:
    needs = schema_for(["a:s1:1"])["properties"]["needs"]
    assert (needs["minItems"], needs["maxItems"]) == (1, 4)

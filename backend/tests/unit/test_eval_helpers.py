from __future__ import annotations

from eval.report import load_golden, outcome
from eval.run_e2e import collect, parse_sse

EXPECTED = [("ACT854 s23", {"act854:s23:1"}), ("PUA220 reg 2", {"pua220:r2:1"}), ("ACT854 s4", {"act854:s4:1", "act854:s4:2"})]


def test_outcome_levels_rank_and_missing() -> None:
    full = outcome(EXPECTED, ["x", "act854:s23:1", "act854:s4:2", "pua220:r2:1"])
    assert (full["level"], full["found"], full["rank"]) == ("full", 3, 2)
    partial = outcome(EXPECTED, ["act854:s23:1"])
    assert (partial["level"], partial["missing"]) == ("partial", ["PUA220 reg 2", "ACT854 s4"])
    assert outcome(EXPECTED, ["x"]) == {"level": "miss", "found": 0, "total": 3, "missing": [l for l, _ in EXPECTED],
                                        "rank": None}


def test_parse_sse() -> None:
    lines = ["event: trace", 'data: {"node": "retrieve"}', "", "event: done", 'data: {"status": "answered"}']
    assert parse_sse(lines) == [("trace", {"node": "retrieve"}), ("done", {"status": "answered"})]


def test_collect_picks_report_fields() -> None:
    events = [("trace", {"node": "generate", "status": "start", "data": {"chunks": ["a", "b"]}}),
              ("answer", {"text": "Within 6 hours.", "label": "Not checked"}),
              ("citations", [{"chunk_id": "a"}]), ("metrics", {"latency_ms": 5, "llm_calls": 1}),
              ("done", {"status": "answered"})]
    res = collect(events)
    assert res == {"status": "answered", "answer": "Within 6 hours.", "citations": ["a"], "passed": ["a", "b"]}
    # An error before `done` leaves status "error" (in-process runner turns exceptions into this event).
    assert collect([("error", {"message": "rate limited"})])["status"] == "error"


def test_golden_set_shape() -> None:
    qs = load_golden()
    assert len(qs) == 30 and len({q["id"] for q in qs}) == 30
    counts = {c: sum(q["category"] == c for q in qs) for c in ("single_hop", "multi_hop", "not_in_corpus", "ambiguous")}
    assert counts == {"single_hop": 10, "multi_hop": 10, "not_in_corpus": 5, "ambiguous": 5}
    for q in qs:
        assert q["question"] and q["answer_notes"]
        assert (q["category"] == "not_in_corpus") <= (q["expected_sections"] == [])

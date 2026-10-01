from __future__ import annotations

import math

from eval.report import cell, e2e_report, outcome, percentile, rate, retrieval_report, section_label
from eval.run import load_golden, parse_sse

EXPECTED = [("ACT854 s23", {"act854:s23:1"}), ("PUA220 reg 2", {"pua220:r2:1"}), ("ACT854 s4", {"act854:s4:1", "act854:s4:2"})]


def test_rate_shows_count_and_percent() -> None:
    assert rate(7, 10) == "7/10 (70%)"
    assert rate(0, 0) == "n/a (0 questions)"


def test_outcome_levels_rank_and_missing() -> None:
    full = outcome(EXPECTED, ["x", "act854:s23:1", "act854:s4:2", "pua220:r2:1"])
    assert (full["level"], full["found"], full["rank"]) == ("full", 3, 2)
    partial = outcome(EXPECTED, ["act854:s23:1"])
    assert (partial["level"], partial["missing"]) == ("partial", ["PUA220 reg 2", "ACT854 s4"])
    assert outcome(EXPECTED, ["x"]) == {"level": "miss", "found": 0, "total": 3, "missing": [l for l, _ in EXPECTED],
                                        "rank": None}


def test_cells() -> None:
    assert cell(outcome(EXPECTED[:1], ["y", "act854:s23:1"])) == "FULL #2"
    assert cell(outcome(EXPECTED[:2], ["act854:s23:1"])) == "PART 1/2 (missing PUA220 reg 2)"
    assert cell(outcome(EXPECTED[:1], [])) == "MISS"


def test_section_labels() -> None:
    assert section_label({"document": "PUA220", "section": "2"}) == "PUA220 reg 2"
    assert section_label({"document": "ACT854", "section": "4", "term": "computer"}) == "ACT854 s4 “computer”"
    assert section_label({"document": "ACT854", "section": "Schedule"}) == "ACT854 Schedule"


def test_percentile_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert percentile(list(range(1, 21)), 0.95) == 19
    assert math.isnan(percentile([], 0.5))


def test_parse_sse() -> None:
    lines = ["event: trace", 'data: {"node": "retrieve"}', "", "event: done", 'data: {"status": "answered"}']
    assert parse_sse(lines) == [("trace", {"node": "retrieve"}), ("done", {"status": "answered"})]


def test_reports_render_tables_without_emoji() -> None:
    qs = [{"id": "S01", "category": "single_hop", "question": "How fast?", "expected_sections": [{"document": "PUA220", "section": "2"}]},
          {"id": "N01", "category": "not_in_corpus", "question": "PDPA?", "expected_sections": []}]
    exp = [("PUA220 reg 2", {"pua220:r2:1"})]
    rows = [{"id": "S01", "category": "single_hop", "method": m, "outcome": outcome(exp, got), "ms": 10}
            for m, got in [("Semantic", ["pua220:r2:1"]), ("BM25", [])]]
    md = retrieval_report(rows, ["Semantic", "BM25"], qs, ["# A"], {"S01": 0.71, "N01": 0.52})
    assert "| Semantic | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 1.0 | 10 ms |" in md
    assert "| single_hop (1) | 0.710 | 0.710 | 0.710 |" in md and "| not_in_corpus (1) | 0.520 |" in md
    assert "| S01 | How fast? | PUA220 reg 2 | FULL #1 | MISS | 0.710 |" in md
    assert "| N01 | PDPA? | – | – | – | 0.520 |" in md

    def e2e_row(q, mode, status, got, label):
        return {"id": q["id"], "category": q["category"], "mode": mode, "status": status, "label": label,
                "outcome": outcome(exp, got) if q["expected_sections"] else None, "cited": 1, "cited_expected": 1,
                "latency_ms": 1000, "llm_calls": 1, "retries": 0}
    rows = [e2e_row(qs[0], "traditional", "answered", ["pua220:r2:1"], "Not checked: traditional mode has no source check"),
            e2e_row(qs[0], "agentic", "answered", ["pua220:r2:1"], "Passed automated source check"),
            e2e_row(qs[1], "traditional", "answered", [], "Not checked: traditional mode has no source check"),
            e2e_row(qs[1], "agentic", "not_found", [], "Nothing to check: no answer was found in the documents")]
    md = e2e_report(rows, ["traditional", "agentic"], qs, ["# B"])
    assert "| Not-in-docs questions declined (OK) | 0/1 (0%) | 1/1 (100%) |" in md
    assert "source check" not in md  # no answer check runs in either mode any more
    assert "| not_in_corpus (1) | OK 0/1 (0%) | OK 1/1 (100%) |" in md
    # same evidence and status: no difference between the modes; the N01 status is
    assert "| S01 | How fast? | PUA220 reg 2 | FULL #1 · answered | FULL #1 · answered |  |" in md
    assert "| N01 | PDPA? | – | WRONG · answered | OK · said not found | * |" in md
    assert not any(ord(ch) > 0x2600 for ch in md)  # no emoji


def test_golden_set_shape() -> None:
    qs = load_golden()
    assert len(qs) == 30 and len({q["id"] for q in qs}) == 30
    counts = {c: sum(q["category"] == c for q in qs) for c in ("single_hop", "multi_hop", "not_in_corpus", "ambiguous")}
    assert counts == {"single_hop": 10, "multi_hop": 10, "not_in_corpus": 5, "ambiguous": 5}
    for q in qs:
        assert q["question"] and q["answer_notes"]
        assert (q["category"] == "not_in_corpus") <= (q["expected_sections"] == [])

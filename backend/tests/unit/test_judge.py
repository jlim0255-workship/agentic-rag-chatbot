"""LLM-as-judge scoring (eval/judge.py) with the judge model faked."""
from __future__ import annotations

import json

from app.llm import ollama
from eval.judge import attribute, judge_row
from eval.report import judge_report

ANSWERABLE = {"id": "S01", "category": "single_hop", "question": "Within how many hours?",
              "answer_notes": ["Notify immediately", "Initial particulars within six hours"]}
OFF = {"id": "N01", "category": "not_in_corpus", "question": "PDPA?", "answer_notes": ["Not found"]}
CHUNKS = {"pua220:r2:1": "[pua220:r2:1] Period\nwithin six hours"}


def row(status="answered", answer="Within six hours [pua220:r2:1].", level="full", **extra) -> dict:
    return {"id": "S01", "category": "single_hop", "mode": "agentic", "model": "m", "status": status,
            "answer": answer, "passed": "pua220:r2:1", "outcome": {"level": level} if level else None,
            "latency_ms": 1000, "llm_calls": 3, **extra}


def judge_replies(monkeypatch, *replies) -> list[str]:
    """Fake judge: returns the replies in order and records each prompt."""
    prompts, queue = [], [r if isinstance(r, str) else json.dumps(r) for r in replies]
    monkeypatch.setattr(ollama, "chat", lambda url, model, prompt, *a, **k: prompts.append(prompt) or queue.pop(0))
    return prompts


def test_answer_is_scored_on_all_three(settings, monkeypatch) -> None:
    prompts = judge_replies(monkeypatch,
                            {"claims": [{"claim": "six hours", "supported": True},
                                        {"claim": "fine of RM1", "supported": False}]},
                            {"addresses": "yes"},
                            {"verdicts": ["missing", "covered"]})
    res = judge_row(row(answer="Within six hours.\n\n**Who to contact** Alice"), ANSWERABLE, CHUNKS, settings)
    assert (res["faithfulness"], res["relevance"], res["correctness"]) == (0.5, 1.0, 0.5)
    assert "within six hours" in prompts[0] and "Alice" not in prompts[0]  # chunks given; contact list not judged
    assert "1. Notify immediately" in prompts[2] and res["unjudged"] == []


def test_declines_are_scored_without_the_judge(settings, monkeypatch) -> None:
    prompts = judge_replies(monkeypatch)
    assert judge_row(row(status="not_found", category="not_in_corpus"), OFF, CHUNKS, settings)["correctness"] == 1.0
    wrongly = judge_row(row(status="not_found"), ANSWERABLE, CHUNKS, settings)
    assert (wrongly["correctness"], wrongly["faithfulness"], wrongly["relevance"]) == (0.0, None, None)
    assert prompts == []


def test_answered_not_in_docs_question_is_wrong_but_still_checked_for_hallucination(settings, monkeypatch) -> None:
    judge_replies(monkeypatch, {"claims": [{"claim": "x", "supported": False}]}, {"addresses": "yes"})
    res = judge_row(row(), OFF, CHUNKS, settings)
    assert (res["correctness"], res["faithfulness"]) == (0.0, 0.0)


def test_invalid_twice_or_wrong_fact_count_is_unjudged(settings, monkeypatch) -> None:
    judge_replies(monkeypatch, "nope", "still nope", {"addresses": "partly"}, {"verdicts": ["covered"]})
    res = judge_row(row(), ANSWERABLE, CHUNKS, settings)
    assert res["unjudged"] == ["faithfulness", "correctness"]
    assert (res["faithfulness"], res["relevance"], res["correctness"]) == (None, 0.5, None)


def test_attribution_order() -> None:
    def judged(correctness, faithfulness, level="full"):
        return {"correctness": correctness, "faithfulness": faithfulness, "outcome": {"level": level}}
    assert attribute(judged(1.0, 1.0)) == "Success"
    assert attribute(judged(1.0, 0.5)) == "Correct but not grounded"
    assert attribute(judged(0.0, 0.5)) == "Hallucination"
    assert attribute(judged(0.5, 1.0)) == "Writer missed facts it had"
    assert attribute(judged(0.0, None, "miss")) == "Retrieval failure"  # declined, nothing retrieved


def test_report_has_one_column_per_configuration() -> None:
    base = {"faithfulness": 1.0, "relevance": 1.0, "correctness": 1.0, "verdicts": ["covered"], "unjudged": [],
            "attribution": "Success"}
    rows = [{**row(mode=m), **base, "config": f"m {m}"} for m in ("traditional", "agentic")]
    md = judge_report(rows, [ANSWERABLE], ["# J"], 0.75)
    assert "|  | m traditional | m agentic |" in md
    assert "| Correct (score >= 0.75), all questions | 1/1 (100%) | 1/1 (100%) |" in md
    assert "| Success | 1/1 (100%) | 1/1 (100%) |" in md

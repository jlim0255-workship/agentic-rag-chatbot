"""LLM-as-judge scoring (eval/judge.py) with the judge model faked."""
from __future__ import annotations

import json

from app.llm import ollama
from eval.judge import judge_row
from eval.report import judge_report

QUESTION = {"id": "S01", "category": "single_hop", "question": "Within how many hours?",
            "answer_notes": ["Notify immediately", "Initial particulars within six hours"]}
CHUNKS = {"pua220:r2:1": "[pua220:r2:1] Period\nwithin six hours", "act854:s99:1": "[act854:s99:1] Other\nunrelated"}


def row(status="answered", answer="Within six hours [1].", passed="pua220:r2:1 act854:s99:1", **extra) -> dict:
    return {"id": "S01", "category": "single_hop", "mode": "agentic", "model": "m", "status": status,
            "answer": answer, "passed": passed, "citations": "pua220:r2:1", **extra}


def judge_replies(monkeypatch, *replies) -> list[str]:
    """Fake judge: returns the replies in order and records each prompt."""
    prompts, queue = [], [r if isinstance(r, str) else json.dumps(r) for r in replies]
    monkeypatch.setattr(ollama, "chat", lambda url, model, prompt, *a, **k: prompts.append(prompt) or queue.pop(0))
    return prompts


def test_answer_is_scored_on_the_triad(settings, monkeypatch) -> None:
    prompts = judge_replies(monkeypatch,
                            {"relevant": [True, False]},
                            {"claims": [{"claim": "six hours", "supported": True},
                                        {"claim": "fine of RM1", "supported": False}]},
                            {"addresses": "yes"})
    answer = "Within six hours [1].\n\nNot covered by the provided documents: the fee"
    res = judge_row(row(answer=answer), QUESTION, CHUNKS, settings)
    assert (res["context_relevance"], res["faithfulness"], res["relevance"]) == (0.5, 0.5, 1.0)
    assert "PASSAGE 2:\n[act854:s99:1]" in prompts[0]  # every chunk given to the writer is judged, in order
    # the line appended by code is not the writer's claim
    assert "within six hours" in prompts[1] and "Not covered" not in prompts[1]
    assert res["unjudged"] == []


def test_declined_with_chunks_gets_only_context_relevance(settings, monkeypatch) -> None:
    prompts = judge_replies(monkeypatch, {"relevant": [True, True]})
    res = judge_row(row(status="not_found", answer="Not found."), QUESTION, CHUNKS, settings)
    assert (res["context_relevance"], res["faithfulness"], res["relevance"]) == (1.0, None, None)
    assert len(prompts) == 1


def test_declined_without_chunks_is_not_judged(settings, monkeypatch) -> None:
    prompts = judge_replies(monkeypatch)
    res = judge_row(row(status="out_of_scope", passed=""), QUESTION, CHUNKS, settings)
    assert (res["context_relevance"], res["faithfulness"], res["relevance"]) == (None, None, None)
    assert prompts == []


def test_invalid_twice_or_wrong_verdict_count_is_unjudged(settings, monkeypatch) -> None:
    judge_replies(monkeypatch, {"relevant": [True]}, {"relevant": [True]},  # 1 verdict for 2 chunks, twice
                  "nope", "still nope", {"addresses": "partly"})
    res = judge_row(row(), QUESTION, CHUNKS, settings)
    assert res["unjudged"] == ["context_relevance", "faithfulness"]
    assert (res["context_relevance"], res["faithfulness"], res["relevance"]) == (None, None, 0.5)


def test_report_has_one_column_per_configuration_and_the_triad() -> None:
    base = {"context_relevance": 0.5, "faithfulness": 1.0, "relevance": 1.0, "unjudged": []}
    rows = [{**row(mode=m), **base, "config": f"m {m}"} for m in ("traditional", "agentic")]
    md = judge_report(rows, [QUESTION], ["# J"])
    assert "|  | m traditional | m agentic |" in md
    assert "| Context relevance, mean | 0.50 (n=1) | 0.50 (n=1) |" in md
    assert "| Faithfulness, mean | 1.00 (n=1) | 1.00 (n=1) |" in md
    assert "| Answer relevance, mean | 1.00 (n=1) | 1.00 (n=1) |" in md
    assert "CR 0.50 · F 1.00 · AR 1.00" in md

"""Plumbing shared by both modes: the state a question carries, its per-question context, the SSE
events sent to the client, and citation parsing. No LLM calls and no decisions live here.

Sections: 1. State   2. Context (time budget, live trace)   3. Citations   4. Final SSE events
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.config import Settings
from app.llm.ollama import OllamaTimeout
from app.retrieval.search import Filters, Retrieved

# ---------------------------------------------------------------------------------------------------
# 1. State (PRD 10.1): what a question carries through the graph. Nodes return partial updates.
# ---------------------------------------------------------------------------------------------------


class SubQuery(BaseModel):
    original: str             # as planned (planner); the question itself in traditional mode
    query: str                # current wording; changed by Rewrite
    variants: list[str] = []  # extra queries from Rewrite, all searched and fused; empty = just `query`
    filters: Filters = Filters()
    answered: bool = False    # every need the grader listed is covered by a verified quote
    partly: bool = False      # at least one need is covered by a verified quote
    missing: str = ""         # grader's note on what evidence is missing
    exhausted: bool = False   # Rewrite gave up (only repeats); no more attempts


class State(TypedDict, total=False):
    question: str                       # never modified
    sub_queries: list[SubQuery]
    current: list[int]                  # indexes of sub-queries used in the latest retrieval attempt
    query_history: list[str]            # every query tried, to detect repeats
    retrieved: dict[int, list[Retrieved]]
    relevant: dict[str, Retrieved]      # chunks with a verified quote, accumulated, keyed by chunk id
    supporting: dict[str, Retrieved]    # every chunk retrieved at any attempt (writer context after `relevant`)
    passed: list[Retrieved]             # chunks actually given to the writer
    retry_count: int
    llm_calls: int
    answer: str                         # the writer's text, with [chunk id] citation markers
    status: str                         # answered | partial | not_found
    route: str                          # agentic planner: out_of_scope | complex
    messages: Annotated[list, add_messages]  # tool_agent's AIMessage + ToolNode's ToolMessage (new_prd.md)


def new_state(question: str) -> State:
    """Starting state. Traditional mode starts with the question as its only sub-query; agentic mode's
    planner replaces it."""
    return {"question": question, "sub_queries": [SubQuery(original=question, query=question)],
            "current": [0], "query_history": [question], "relevant": {}, "supporting": {},
            "retry_count": 0, "llm_calls": 0, "status": "answered"}


# ---------------------------------------------------------------------------------------------------
# 2. Context: settings, time budget and live trace output for one question.
# ---------------------------------------------------------------------------------------------------

Event = tuple[str, Any]


def trace(node: str, status: str, summary: str, data: dict | None = None) -> Event:
    return "trace", {"node": node, "status": status, "summary": summary, "data": data or {}}


@dataclass
class Ctx:
    settings: Settings
    started: float = field(default_factory=time.monotonic)

    def timeout(self) -> float:
        """Timeout for the next LLM call: the per-call limit, capped by what is left of the question budget."""
        remaining = self.settings.question_budget_s - (time.monotonic() - self.started)
        if remaining <= 0:
            raise OllamaTimeout("question budget used up")
        return min(self.settings.llm_call_timeout_s, remaining)

    @staticmethod
    def emit(node: str, status: str, summary: str, data: dict | None = None) -> None:
        """Send a trace event to the client immediately (not when the node returns)."""
        get_stream_writer()(trace(node, status, summary, data))


# ---------------------------------------------------------------------------------------------------
# 3. Citations: "[act854:s23:1]" markers in the writer's answer.
# ---------------------------------------------------------------------------------------------------

NOT_FOUND = "NOT_FOUND"  # what the writer replies when its sources do not answer the question
BRACKETS = re.compile(r"\[([^\]]+)\]")
CHUNK_ID = re.compile(r"[a-z0-9_]+:[a-z0-9_]+:\d+")


def cited_ids(answer: str, given=None) -> list[str]:
    """Chunk IDs cited in the answer, in order of first use.

    Tolerates what the small model actually writes: '[a:s1:1, a:s2:1]' and '[act854:s22:1(1)(a)]'.
    With `given` (the source IDs), misread IDs are resolved to their source (see `resolve`).
    """
    ids = (i for group in BRACKETS.findall(answer) for i in CHUNK_ID.findall(group))
    return list(dict.fromkeys(resolve(i, given) if given else i for i in ids))


def resolve(id_: str, given) -> str:
    """The model reads our chunk counter as a subsection ('pua220:r2:1' cited as 'pua220:r2:2' for reg 2(2)).
    If exactly one given source has the same document and section, it is that source; otherwise unchanged."""
    if id_ in given:
        return id_
    section = id_.rsplit(":", 1)[0]
    same = [g for g in given if g.rsplit(":", 1)[0] == section]
    return same[0] if len(same) == 1 else id_


# ---------------------------------------------------------------------------------------------------
# 4. Final SSE events (PRD 11.2 order: trace* -> answer -> citations -> metrics -> done).
# ---------------------------------------------------------------------------------------------------

NOT_FOUND_TEXT = "Not found in the provided documents."
NOT_FOUND_LABEL = "Nothing to check: no answer was found in the documents"


def finish(status: str, started: float, llm_calls: int, retries: int = 0) -> Iterator[Event]:
    yield "metrics", {"latency_ms": round((time.monotonic() - started) * 1000), "llm_calls": llm_calls,
                      "retries": retries, "regenerations": 0}
    yield "done", {"status": status}


def fixed_reply(text: str, label: str, status: str, started: float, llm_calls: int,
                retries: int = 0) -> Iterator[Event]:
    """A reply written by code, not the model: no citations."""
    yield "answer", {"text": text, "label": label}
    yield "citations", []
    yield from finish(status, started, llm_calls, retries)


def not_found(label: str, started: float, llm_calls: int, retries: int = 0, extra: str = "") -> Iterator[Event]:
    yield from fixed_reply(NOT_FOUND_TEXT + extra, label, "not_found", started, llm_calls, retries)


def answer(text: str, chunks: list[Retrieved], label: str, started: float, llm_calls: int,
           retries: int = 0, status: str = "answered", extra: str = "") -> Iterator[Event]:
    """Final events for a written answer. Only chunks given to the writer can be cited.
    `extra` is text written by code (the staff contact list), appended after the writer's answer."""
    by_id = {c.id: c for c in chunks}
    citations = [{"chunk_id": i, "document": by_id[i].short_name, "section": by_id[i].label,
                  "page": by_id[i].pages} for i in cited_ids(text, by_id) if i in by_id]
    # The model sometimes writes a sentence and then NOT_FOUND; without citations that is a not-found.
    if not text or (NOT_FOUND in text and not citations):
        yield from not_found(NOT_FOUND_LABEL, started, llm_calls, retries, extra)
        return
    yield "answer", {"text": text.replace(NOT_FOUND, "").strip() + extra, "label": label}
    yield "citations", citations
    yield from finish(status, started, llm_calls, retries)

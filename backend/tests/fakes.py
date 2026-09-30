"""Shared fakes for pipeline tests: a scripted LLM, fake chunks, and helpers to run a question and read
its events. The `world` fixture that wires them in lives in conftest.py."""
from __future__ import annotations

import json
from contextlib import contextmanager

from app.config import Settings
from langchain_core.messages import AIMessage

from app.pipeline.flow import run as run_flow
from app.retrieval.search import Retrieved


def chunk(id_: str, section: str = "23") -> Retrieved:
    return Retrieved(id=id_, short_name="ACT854", section=section, subsection=None, heading="H",
                     part="IV", page_start=26, page_end=26, text=f"text of {id_}", score=0.03)


class FakeLLM:
    """Scripted replies per call type, told apart by the output schema: plan ('sub_queries'),
    grade ('needs'), grounding check ('supported'), rewrite (other schema), generate (no schema).
    Defaults: plan = one sub-query equal to the question; check = every sentence supported."""

    def __init__(self, grades=(), rewrites=(), answer="The fine is RM500,000 [act854:s23:1].", plans=None,
                 checks=()):
        self.grades, self.rewrites, self.answer, self.checks = list(grades), list(rewrites), answer, list(checks)
        self.plans = list(plans) if plans is not None else None
        self.calls: list[str] = []

    def __call__(self, base_url, model, prompt, timeout_s, num_ctx, format=None):
        if format is None:
            kind = "generate"
        else:
            props = format["properties"]
            kind = ("plan" if "sub_queries" in props else "grade" if "sufficient" in props
                    else "check" if "supported" in props else "rewrite")
        self.calls.append(kind)
        if kind == "generate":
            return self.answer
        if kind == "check":
            n = format["properties"]["supported"]["minItems"]
            return json.dumps({"supported": self.checks.pop(0) if self.checks else [True] * n})
        if kind == "plan":
            if self.plans is None:
                question = prompt.split("Question: ", 1)[1].splitlines()[0]
                return json.dumps({"sub_queries": [{"query": question}]})
            reply = self.plans.pop(0)
        else:
            reply = (self.grades if kind == "grade" else self.rewrites).pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def grade(relevant: list[str], not_relevant: list[str] = (), missing: str = "") -> dict:
    """Yes/no grader reply: sufficient when any chunk is relevant and nothing is missing.
    The signature is kept from the needs grader so the routing tests read the same."""
    sufficient = bool(relevant) and not missing
    return {"sufficient": sufficient, "missing": "" if sufficient else (missing or "the answer")}


class FakeToolModel:
    """Stands in for ChatOllama(...).bind_tools([...]) in tool_agent: replies with the given tool calls."""

    def __init__(self, tool_calls: list[dict] | None = None):
        self.tool_calls = tool_calls or []

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return AIMessage(content="", tool_calls=self.tool_calls)


@contextmanager
def fake_connection(*_a, **_k):
    yield object()


def run(settings: Settings, question: str = "What is the penalty?", mode: str = "agentic") -> list:
    return list(run_flow(question, mode, settings))


def final(events: list) -> dict:
    return {name: data for name, data in events if name != "trace"}


def trace_nodes(events: list) -> list[str]:
    return [f"{d['node']}:{d['status']}" for n, d in events if n == "trace"]

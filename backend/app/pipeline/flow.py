"""START HERE. How a question flows, in both modes. Every box is one file in app/pipeline/nodes/.

TRADITIONAL: one search with the question, then write. 1 LLM call.

    retrieve ─► any chunks? ─ yes ─► generate ─► end
                            └ no ──► end ("not found")

AGENTIC: the LLM's judgements choose the route; code enforces the limits.

    planner ─┬─ out_of_scope ─► end (fixed reply written by code, no writer call)
     (LLM)   └─ complex ▼
               retrieve ─► grade ─┬─ sufficient ─────────────────────► generate ─► end
               (no LLM)   (LLM)   ├─ not sufficient, retries left ─► rewrite ─► back to retrieve
                                  │                                  (LLM)
                                  └─ not sufficient, no retries left ─► not_found (no LLM)
                                                                       ├─ some parts found ─► generate
                                                                       └─ nothing found ────► end ("not found")

No answer check runs after generate in either mode (the grounding check was removed; its small-model
judge mostly flagged correct sentences). Citations are still filtered by code in core.answer.

Where things live:
    flow.py        this map, the decisions between steps (after_*), and the runner (run)
    core.py        state, time budget, trace/SSE events, citation parsing (shared plumbing)
    nodes/*.py     one step each: planner, retrieve, grade, rewrite, not_found, generate
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import partial

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm.ollama import OllamaTimeout
from app.pipeline import core
from app.pipeline.core import Ctx, Event, State, new_state, trace
from app.pipeline.nodes.generate import generate_node
from app.pipeline.nodes.grade import grade_node
from app.pipeline.nodes.not_found import not_found_node
from app.pipeline.nodes.planner import planner_node
from app.pipeline.nodes.retrieve import retrieve_node
from app.pipeline.nodes.rewrite import rewrite_node

UNCHECKED_LABEL = "Not checked: no automated source check"
FIXED_REPLIES = {  # route -> (text, label); written by code so the model never answers from its own knowledge
    "out_of_scope": ("I can only answer questions about Malaysia's Cyber Security Act 2024 and its two "
                     "regulations (P.U.(A) 219/2024 and P.U.(A) 220/2024), citing the exact sections. For example: "
                     "\"How quickly must a cyber security incident be reported?\"", "Out of scope: not answered"),
}


# --- decisions between steps (code, not LLM) ---

def after_planner(state: State) -> str:
    return END if state["route"] in FIXED_REPLIES else "retrieve"


def after_retrieve(state: State) -> str:
    """Traditional: never ask the writer without sources (it would answer from its own knowledge).
    Agentic ('complex'): grade the evidence."""
    if state.get("route") == "complex":
        return "grade"
    return "generate" if state["supporting"] else END


def after_grade(state: State, settings: Settings) -> str:
    subs = state["sub_queries"]
    if all(s.answered for s in subs):
        return "generate"
    can_retry = any(not s.answered and not s.exhausted for s in subs)
    return "rewrite" if can_retry and state["retry_count"] < settings.max_retries else "not_found"


def after_not_found(state: State) -> str:
    return "generate" if state["status"] == "partial" else END


# --- the two graphs ---

def traditional_graph(ctx: Ctx):
    g = StateGraph(State)
    g.add_node("retrieve", partial(retrieve_node, ctx=ctx))
    g.add_node("generate", partial(generate_node, ctx=ctx))
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", END])
    g.add_edge("generate", END)
    return g.compile()


def agentic_graph(ctx: Ctx):
    g = StateGraph(State)
    # for name, node in [("planner", planner_node), ("retrieve", retrieve_node), ("grade", grade_node), ("rewrite", rewrite_node), ("not_found", not_found_node), ("generate", generate_node)]:
    #     g.add_node(name, partial(node, ctx=ctx))
    g.add_node("planner", partial(planner_node, ctx=ctx))
    g.add_node("retrieve", partial(retrieve_node, ctx=ctx))
    g.add_node("grade", partial(grade_node, ctx=ctx))
    g.add_node("rewrite", partial(rewrite_node, ctx=ctx))
    g.add_node("not_found", partial(not_found_node, ctx=ctx))
    g.add_node("generate", partial(generate_node, ctx=ctx))

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", after_planner, ["retrieve", END])
    g.add_conditional_edges("retrieve", after_retrieve, ["grade", "generate", END])
    g.add_conditional_edges("grade", partial(after_grade, settings=ctx.settings), ["generate", "rewrite", "not_found"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("not_found", after_not_found, ["generate", END])
    g.add_edge("generate", END)
    return g.compile()


def recursion_limit(settings: Settings) -> int:
    """Longest agentic path: planner, (retrieve, grade, rewrite) per retry, final retrieve, grade, not_found,
    generate. LangGraph raises if a run ever exceeds it."""
    return 1 + 3 * settings.max_retries + 4 + 1


# --- the runner: one graph run -> SSE events ---

def run(question: str, mode: str, settings: Settings) -> Iterator[Event]:
    """Yield (event, payload) pairs in SSE order (trace* -> answer -> citations -> metrics -> done). Trace events stream live from inside the nodes.
    Timeouts end with status 'timeout'; other errors propagate (the API turns them into an `error` event)."""
    ctx = Ctx(settings)
    graph = agentic_graph(ctx) if mode == "agentic" else traditional_graph(ctx)
    final = new_state(question)
    try:
        # custom items = trace events like logs, if ctx.emit(...) will send it to browser
        # values items = full state after each step, we keep overwriting final ( the state )
        for kind, item in graph.stream(final, {"recursion_limit": recursion_limit(settings)},
                                       stream_mode=["custom", "values"]):
            if kind == "custom":
                yield item        # a trace event emitted by a node
            else:
                final = item      # the state after each step; the last one is the result
    except OllamaTimeout as e:
        yield trace("timeout", "end", str(e))
        yield from core.finish("timeout", ctx.started, final["llm_calls"], final["retry_count"])
        return

    # turn final into closing events
    calls, retries = final["llm_calls"], final["retry_count"]
    if final.get("route") in FIXED_REPLIES:
        text, label = FIXED_REPLIES[final["route"]]
        yield from core.fixed_reply(text, label, final["route"], ctx.started, calls, retries)
        return
    if final["status"] == "not_found" or "answer" not in final:
        yield from core.not_found(core.NOT_FOUND_LABEL, ctx.started, calls, retries)
        return
    yield from core.answer(final["answer"], final["passed"], UNCHECKED_LABEL, ctx.started, calls, retries,
                           status=final["status"])

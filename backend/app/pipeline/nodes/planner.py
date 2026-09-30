"""Planner node (PRD 10.2 planner + M7 router in one call): picks a route, and for questions about the law
splits the question into 1..MAX_SUB_QUERIES sub-queries, each with optional filters.

Routes (the JSON schema only allows these): out_of_scope gets a fixed reply (no retrieval, no writer call);
complex runs the full agent. Invalid output -> complex (safest). Two routes, not four: with "direct" and
"simple" the small model sent answerable questions to the greeting reply and never used "simple".

Allowed filter values come from the database, never from the prompt or code; anything else the model
writes is dropped in code.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from app.db import connection
from app.llm.structured import call_structured
from app.pipeline.core import Ctx, State, SubQuery
from app.retrieval.search import Filters

PROMPT = """You plan searches over Malaysian cyber security law to answer a user's question.

Documents (use these exact codes in "documents"):
{documents}

Parts of the Act (use these exact numerals in "parts"):
{parts}

Question: {question}

First choose "route":
- "out_of_scope": not a question about Malaysian cyber security law (greetings, other topics, other laws,
  general knowledge). Return an empty "sub_queries" list.
- "complex": any question about Malaysian cyber security law, including who to contact or who is
  responsible when a cyber security incident happens.

Rules for "sub_queries":
- If the question asks about several different things, write one search query per thing (at most {max_n}).
  Otherwise write exactly one query.
- Each query must be self-contained and use the formal terms a statute would use.
- Add a document or part filter ONLY if you are certain the answer is in it. When unsure, leave the list empty."""


class PlanItem(BaseModel):
    query: str
    documents: list[str] = []
    parts: list[str] = []


Route = Literal["out_of_scope", "complex"]


class PlanOutput(BaseModel):
    route: Route = "complex"
    sub_queries: list[PlanItem] = []


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}


def normalise_part(value: str) -> str:
    """'Part IV', 'iv', '4' -> 'IV' (the form stored in chunks.part)."""
    v = re.sub(r"(?i)^part\s+", "", value.strip()).upper()
    return ROMAN.get(int(v), v) if v.isdigit() else v


def allowed_filters(conn) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """([(short_name, title)], [(part, part_title)]) currently in the database."""
    docs = conn.execute("SELECT short_name, title FROM documents ORDER BY short_name").fetchall()
    parts = conn.execute("SELECT DISTINCT part, part_title FROM chunks WHERE part IS NOT NULL").fetchall()
    order = {v: k for k, v in ROMAN.items()}
    return [tuple(d) for d in docs], sorted((tuple(p) for p in parts), key=lambda p: order.get(p[0], 99))


def clean_plan(plan: PlanOutput | None, question: str, docs: set[str], parts: set[str],
               max_n: int) -> tuple[list[SubQuery], list[str]]:
    """Validated sub-queries plus a list of dropped filter values."""
    items = [i for i in (plan.sub_queries if plan else []) if i.query.strip()][:max_n]
    if not items:
        return [SubQuery(original=question, query=question)], []
    subs, dropped = [], []
    for item in items:
        d = [x.strip().upper() for x in item.documents]
        p = [normalise_part(x) for x in item.parts]
        dropped += [x for x in d if x not in docs] + [x for x in p if x not in parts]
        f = Filters(documents=[x for x in dict.fromkeys(d) if x in docs],
                    parts=[x for x in dict.fromkeys(p) if x in parts])
        subs.append(SubQuery(original=item.query.strip(), query=item.query.strip(), filters=f))
    return subs, dropped


def planner_node(state: State, ctx: Ctx) -> dict:
    ctx.emit("planner", "start", "Planning the search…")
    with connection.connect(ctx.settings.database_url, ctx.settings.health_timeout_s) as conn:
        # connect db and fetch allowed filters (documents and parts) from the database
        docs, parts = allowed_filters(conn)

    # fill in the prompt
    prompt = PROMPT.format(documents="\n".join(f"- {s}: {t}" for s, t in docs),
                           parts="\n".join(f"- {p}: {t}" for p, t in parts),
                           question=state["question"], max_n=ctx.settings.max_sub_queries)

    # pass in the structured prompt to the LLM, get back a structured plan
    plan, calls = call_structured(prompt, PlanOutput, ctx.settings, ctx.timeout())
    route = plan.route if plan else "complex"
    
    subs, dropped = clean_plan(plan, state["question"], {d for d, _ in docs}, {p for p, _ in parts},
                               ctx.settings.max_sub_queries)
    summary = (f"route: {route}" if route == "out_of_scope"
               else f"route: {route} · {len(subs)} sub-quer{'y' if len(subs) == 1 else 'ies'}")
    ctx.emit("planner", "end", summary,
             {"route": route, "sub_queries": [{"query": s.query, "filters": s.filters.model_dump()} for s in subs],
              "dropped_filters": dropped, "fallback": plan is None})
    return {"route": route, "sub_queries": subs, "current": list(range(len(subs))),
            "query_history": [state["question"], *(s.query for s in subs)],
            "llm_calls": state["llm_calls"] + calls}

"""Retrieve node: hybrid retrieval for each sub-query in `current`. No LLM."""
from __future__ import annotations

from app.db import connection
from app.pipeline.core import Ctx, State
from app.retrieval import search


def retrieve_node(state: State, ctx: Ctx) -> dict:
    subs = state["sub_queries"]
    ctx.emit("retrieve", "start", "Retrieving…", {"queries": [subs[i].query for i in state["current"]]})
    with connection.connect(ctx.settings.database_url, ctx.settings.health_timeout_s) as conn:
        retrieved = {i: search.retrieve(conn, ctx.settings, subs[i].query, subs[i].filters, timeout_s=ctx.timeout())
                     for i in state["current"]}
    ctx.emit("retrieve", "end", f"{sum(map(len, retrieved.values()))} chunks",
             {"queries": [{"query": subs[i].query,
                           "chunks": [{"id": c.id, "score": round(c.score, 4)} for c in retrieved[i]]}
                          for i in state["current"]]})
    supporting = dict(state.get("supporting", {}))
    for chunks in retrieved.values():
        for c in chunks:
            supporting.setdefault(c.id, c)
    return {"retrieved": retrieved, "supporting": supporting}

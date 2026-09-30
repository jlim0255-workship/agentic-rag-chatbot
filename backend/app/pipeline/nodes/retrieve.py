"""Retrieve node (PRD 10.2): hybrid retrieval for each sub-query in `current`. No LLM.

A sub-query with several variants (from Rewrite) searches each and fuses them with RRF (multi-query).
"""
from __future__ import annotations

from app.db import connection
from app.pipeline.core import Ctx, State
from app.retrieval import search


def retrieve_node(state: State, ctx: Ctx) -> dict:
    subs = state["sub_queries"]
    queries = {i: subs[i].variants or [subs[i].query] for i in state["current"]}
    ctx.emit("retrieve", "start", "Retrieving…", {"queries": [q for i in state["current"] for q in queries[i]]})
    retrieved = {}
    with connection.connect(ctx.settings.database_url, ctx.settings.health_timeout_s) as conn:
        for i in state["current"]:
            lists = [search.retrieve(conn, ctx.settings, q, subs[i].filters, timeout_s=ctx.timeout())
                     for q in queries[i]]
            if len(lists) == 1:  # one query: keep its hybrid scores as they are
                retrieved[i] = lists[0]
                continue
            by_id = {c.id: c for results in lists for c in results}
            fused = search.rrf([[c.id for c in results] for results in lists], ctx.settings.rrf_k)
            retrieved[i] = [by_id[id_].model_copy(update={"score": score})
                            for id_, score in fused[:ctx.settings.top_k_final]]
    ctx.emit("retrieve", "end", f"{sum(map(len, retrieved.values()))} chunks",
             {"queries": [{"query": " | ".join(queries[i]),
                           "chunks": [{"id": c.id, "score": round(c.score, 4)} for c in retrieved[i]]}
                          for i in state["current"]]})
    supporting = dict(state.get("supporting", {}))
    for chunks in retrieved.values():
        for c in chunks:
            supporting.setdefault(c.id, c)
    return {"retrieved": retrieved, "supporting": supporting}

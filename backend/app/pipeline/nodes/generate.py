"""Generate node: the writer. Answers from the given chunks only, citing chunk IDs.
Shared by both modes. Uses the answer model; the only node that writes text for the user.

Partial path: the writer is asked only the sub-queries judged sufficient, and code appends which parts are
not covered. Asked the full question with a note about the missing part, the small model replied NOT_FOUND
to all of it, discarding the evidence it held (replayed 3/3 on gemma4:e2b, 2026-10-03)."""
from __future__ import annotations

from app.llm import ollama
from app.pipeline.core import NOT_FOUND, Ctx, State
from app.retrieval.search import Retrieved

PROMPT = """You answer questions about Malaysia's Cyber Security Act 2024 and its regulations.
Use ONLY the sources below. Never use outside knowledge.

Rules:
- Start with one sentence that directly answers the question, ending with the ID of its source in square
  brackets, e.g. [act854:s23:1].
- Then give the supporting details as a short Markdown bullet list, one point per line, each ending with the
  ID of its source in square brackets.
- Only use IDs that appear in the sources.
- If the sources do not answer the question, reply with exactly: {not_found}
- Be concise and plain. Quote numbers and time limits exactly as the source states them.

Sources:
{sources}

Question: {question}
Answer:"""

NOT_COVERED = "Not covered by the provided documents: {parts}"  # appended by code on the partial path


def build_prompt(question: str, chunks: list[Retrieved]) -> str:
    sources = "\n\n".join(f"[{c.id}] {c.label} — {c.heading}\n{c.text}" for c in chunks)
    return PROMPT.format(not_found=NOT_FOUND, sources=sources, question=question)


def generate_node(state: State, ctx: Ctx) -> dict:
    # Steer, don't delete: chunks the grader judged sufficient first, then everything else retrieved, capped.
    # (Traditional mode has no grader, so `relevant` is empty and the writer gets what was retrieved.)
    verified = list(state["relevant"].values())
    others = [c for id_, c in state["supporting"].items() if id_ not in state["relevant"]]
    chunks = (verified + others)[:max(ctx.settings.writer_max_chunks, len(verified))]
    partial = state["status"] == "partial"
    # Partial: the writer answers only what was judged sufficient; code names the rest (see module docstring).
    # The writer still replies NOT_FOUND if its sources don't answer, and core.answer then reports not found.
    unanswered = [s.original for s in state["sub_queries"] if not s.answered and not s.partly] if partial else []
    answered = [s.original for s in state["sub_queries"] if s.answered]
    question = "\n".join(answered) if partial and answered else state["question"]
    ctx.emit("generate", "start", "Writing the answer…",
             {"chunks": [c.id for c in chunks], "question": question, "unanswered": unanswered})
    text = ollama.chat(ctx.settings.ollama_base_url, ctx.settings.answer_model, build_prompt(question, chunks),
                       ctx.timeout(), ctx.settings.llm_num_ctx).strip()
    if unanswered:
        text += "\n\n" + NOT_COVERED.format(parts="; ".join(unanswered))
    ctx.emit("generate", "end", "Answer written")
    return {"answer": text, "passed": chunks, "llm_calls": state["llm_calls"] + 1}

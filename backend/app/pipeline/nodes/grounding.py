"""Grounding check node (PRD 10.2, M6): is every sentence backed by the source it cites?

Code checks first (free and exact), then one batched fast-model call for the sentences that pass them.
Failing sentences are FLAGGED in the answer, not removed. No regeneration (agreed deviation from PRD:
the same small model given the same evidence tends to repeat its mistake; flags keep it transparent).
Grounded is not the same as correct: an answer can match its sources and still miss the question.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from app.llm.structured import call_structured
from app.pipeline.core import Ctx, NOT_FOUND, State, cited_ids

PASSED = "Passed automated source check"
INCOMPLETE = "Source check incomplete (checker output invalid); citation checks only"

# ponytail: phrase list for the model's own "this part is not covered" sentences, which need no citation.
NOT_COVERED = ("not covered", "not found", "not stated", "not specified", "does not specify",
               "do not specify", "not provided", "does not state", "do not state", "not mention")

# A sentence ends at . ! ? + space, or after a trailing citation ("... years. [act854:s23:1] Next ...").
# ponytail: abbreviations like "s. 23" split early; harmless (both halves are checked).
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\[\s])|(?<=\])\s+(?=[A-Z(])")

PROMPT = """For each numbered sentence, decide whether the passages it cites support it: true only if every
fact in the sentence is stated in those passages. Give one true/false per sentence, in order.

Passages:
{passages}

Sentences:
{sentences}"""


class CheckOutput(BaseModel):
    supported: list[bool]


def flagged_label(n: int) -> str:
    return f"Automated source check flagged {n} sentence{'s' if n > 1 else ''}"


def split_sentences(answer: str) -> list[list[str]]:
    """Sentences per line (lines keep list items like '(a) ...' separate)."""
    return [[s for s in SPLIT.split(line.strip()) if s] for line in answer.splitlines() if line.strip()]


def code_problems(sentence: str, given: set[str]) -> list[str]:
    ids = cited_ids(sentence, given)
    unknown = [i for i in ids if i not in given]
    if unknown:
        return [f"cites {', '.join(unknown)}, which was not among the sources"]
    lower = sentence.lower()
    if not ids and not sentence.rstrip().endswith(":") and not any(p in lower for p in NOT_COVERED):
        return ["no citation"]
    return []


def grounding_node(state: State, ctx: Ctx) -> dict:
    answer = state["answer"]
    if not answer or (NOT_FOUND in answer and not cited_ids(answer)):
        return {}  # a not-found answer has no claims to check
    ctx.emit("grounding", "start", "Checking the answer against its sources…")
    by_id = {c.id: c for c in state["passed"]}
    lines = split_sentences(answer)
    checks = [{"sentence": s, "ids": cited_ids(s, by_id), "problems": code_problems(s, set(by_id))}
              for line in lines for s in line]

    to_llm = [c for c in checks if c["ids"] and not c["problems"]]
    calls, incomplete = 0, False
    if to_llm:
        used = dict.fromkeys(i for c in to_llm for i in c["ids"])
        prompt = PROMPT.format(passages="\n\n".join(f"[{i}]\n{by_id[i].text}" for i in used),
                               sentences="\n".join(f"{n}. {c['sentence']}" for n, c in enumerate(to_llm, 1)))
        schema = CheckOutput.model_json_schema()
        schema["properties"]["supported"].update(minItems=len(to_llm), maxItems=len(to_llm))
        out, calls = call_structured(prompt, CheckOutput, ctx.settings, ctx.timeout(), json_schema=schema)
        if out is None or len(out.supported) != len(to_llm):
            incomplete = True  # fallback: keep the code checks, say the check was incomplete
        else:
            for c, ok in zip(to_llm, out.supported):
                if not ok:
                    c["problems"].append("not supported by the cited source")

    # Rebuild the answer line by line, appending a visible flag to each failing sentence.
    out_lines, k = [], 0
    for line in lines:
        parts = []
        for s in line:
            problems = checks[k]["problems"]
            parts.append(f"{s} ⚠ [{'; '.join(problems)}]" if problems else s)
            k += 1
        out_lines.append(" ".join(parts))
    text = "\n".join(out_lines)
    flagged = sum(bool(c["problems"]) for c in checks)
    label = INCOMPLETE if incomplete else (flagged_label(flagged) if flagged else PASSED)
    ctx.emit("grounding", "end", label, {"sentences": checks})
    return {"answer": text, "label": label, "llm_calls": state["llm_calls"] + calls}

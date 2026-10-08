"""Experiment B: traditional vs agentic, end to end, on the golden questions.

Answers are generated once and saved (`-rows.jsonl`, the input of `eval.judge`, which writes the report),
plus one `-<mode>-answers.md` per mode for reading each answer next to its key facts.

  default       asks the running API (POST /chat), i.e. the deployed stack with the models in .env
  --model X     runs the pipeline in-process with X as both FAST and ANSWER model (e.g. an Ollama Cloud
                model, to see how the model alone changes the results); the API need not be running.
                Eval only: a `-cloud` model sends questions and chunks to Ollama Cloud.

Usage (from backend/; DB ingested, Ollama running):
    python -m eval.run_e2e                                   # both modes via the API (~25 min)
    python -m eval.run_e2e --mode agentic                    #   one mode only
    python -m eval.run_e2e --model gemma4:31b-cloud          # in-process, other model
One-time setup for a cloud model:  ollama signin  then  ollama pull gemma4:31b-cloud
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import httpx

from app.config import Settings, get_settings
from app.llm import ollama
from app.pipeline.flow import run
from eval.report import EVAL_DIR, answers_file, load_golden, stem


def parse_sse(lines) -> list[tuple[str, object]]:
    events, name = [], None
    for line in lines:
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def collect(events) -> dict:
    """One question's (event, data) pairs -> the fields the judge needs."""
    result = {"status": "error", "answer": "", "citations": [], "passed": []}
    for name, data in events:
        # Evidence = the chunks given to the writer. None if the writer was never called.
        if name == "trace" and data["node"] == "generate" and data["status"] == "start":
            result["passed"] = data["data"]["chunks"]
        elif name == "answer":
            result["answer"] = data["text"]
        elif name == "citations":
            result["citations"] = [c["chunk_id"] for c in data]
        elif name == "done":
            result["status"] = data["status"]
        elif name == "error":
            result["answer"] = f"ERROR: {data['message']}"
    return result


def ask(api: str, question: str, mode: str, timeout_s: float) -> dict:
    with httpx.stream("POST", f"{api}/chat", json={"question": question, "mode": mode},
                      timeout=httpx.Timeout(10, read=timeout_s)) as r:
        r.raise_for_status()
        return collect(parse_sse(r.iter_lines()))


def ask_local(question: str, mode: str, settings: Settings) -> dict:
    """Like the API: any failure becomes one `error` event instead of stopping the whole run."""
    events = []
    try:
        for event in run(question, mode, settings):
            events.append(event)
    except Exception as e:  # noqa: BLE001 - one bad question (e.g. a cloud rate limit) must not end the run
        events.append(("error", {"message": str(e) or type(e).__name__}))
    return collect(events)


def preflight(settings: Settings) -> None:
    """One tiny call, so a missing model or sign-in fails here with a clear message, not on question 1."""
    try:
        ollama.chat(settings.ollama_base_url, settings.answer_model, "Reply with OK.",
                    settings.llm_call_timeout_s, settings.llm_num_ctx)
    except ollama.OllamaError as e:
        raise SystemExit(f"Model {settings.answer_model!r} is not usable: {e}\n"
                         f"For a cloud model run `ollama signin` and `ollama pull {settings.answer_model}`.")


def run_e2e(modes: list[str], ask_fn, settings: Settings, name: str = "e2e") -> Path:
    """ask_fn(question, mode) -> collect()-style dict. Returns the path of the -rows.jsonl file.
    `settings` must hold the model ask_fn really uses (saved per row)."""
    questions = load_golden()
    rows = []
    for mode in modes:
        ask_fn("warm-up: what is the short title of the Act?", mode)  # load models; not counted
        for i, q in enumerate(questions, 1):
            print(f"[{mode} {i}/{len(questions)}] {q['id']}", flush=True)
            res = ask_fn(q["question"], mode)
            rows.append({"id": q["id"], "category": q["category"], "mode": mode, "model": settings.answer_model,
                         "status": res["status"], "passed": " ".join(res["passed"]),
                         "citations": " ".join(res["citations"]), "answer": res["answer"]})

    when = f"{datetime.now():%Y-%m-%d %H:%M}"
    base = stem(name)
    rows_path = base.with_name(f"{base.name}-rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for mode in modes:
        text = answers_file(mode, [r for r in rows if r["mode"] == mode], questions, when)
        base.with_name(f"{base.name}-{mode}-answers.md").write_text(text, encoding="utf-8")
    return rows_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["traditional", "agentic"], help="run one mode only (default: both)")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API to ask (ignored with --model)")
    parser.add_argument("--model", help="run in-process with this model as FAST and ANSWER model")
    args = parser.parse_args()
    modes = [args.mode] if args.mode else ["traditional", "agentic"]

    if args.model:
        settings = get_settings().model_copy(update={"fast_model": args.model, "answer_model": args.model})
        preflight(settings)
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
        rows = run_e2e(modes, lambda question, mode: ask_local(question, mode, settings), settings,
                       name=f"e2e-{slug}")
    else:
        settings = get_settings()
        timeout = settings.question_budget_s + 30
        rows = run_e2e(modes, lambda question, mode: ask(args.api, question, mode, timeout), settings)
    print(f"rows: {rows}\nnext: python -m eval.judge {rows.relative_to(EVAL_DIR.parent).as_posix()}")


if __name__ == "__main__":
    main()

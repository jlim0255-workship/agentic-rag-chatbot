"""End-to-end evaluation (Experiment B) on a different model, e.g. an Ollama Cloud one, to see how the
model alone changes the results. Same golden set, prompts, retrieval and limits as `eval.run e2e`; only
FAST_MODEL and ANSWER_MODEL are swapped. Runs the pipeline in-process, so the API need not be running.

Eval only: a `-cloud` model sends questions and retrieved chunks to Ollama Cloud (through the local Ollama
daemon). The shipped system stays local.

One-time setup:  ollama signin  then  ollama pull gemma4:31b-cloud
Usage (from backend/; DB ingested, Ollama running):
    python -m eval.run_cloud                                  # gemma4:31b-cloud, both modes
    python -m eval.run_cloud --model gemma4:31b-cloud --mode agentic
"""
from __future__ import annotations

import argparse
import re

from app.config import Settings, get_settings
from app.llm import ollama
from app.pipeline.flow import run
from eval.run import collect, run_e2e

DEFAULT_MODEL = "gemma4:31b-cloud"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"used as both FAST and ANSWER model "
                                                                f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--mode", choices=["traditional", "agentic"], help="run one mode only (default: both)")
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"fast_model": args.model, "answer_model": args.model})
    preflight(settings)
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
    stem = run_e2e([args.mode] if args.mode else ["traditional", "agentic"],
                   lambda question, mode: ask_local(question, mode, settings), settings, name=f"e2e-{slug}")
    print(f"report: {stem}.md")


if __name__ == "__main__":
    main()

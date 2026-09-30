"""Grounding check (M6 done-when): deliberately corrupted answers are caught and flagged, never 'verified'."""
from __future__ import annotations

from app.pipeline.nodes.grounding import INCOMPLETE, PASSED, code_problems, split_sentences
from tests.fakes import FakeLLM, chunk, final, grade, run


def ask(settings, world, answer: str, checks=()):
    world.by_query = {"What is the penalty?": [chunk("act854:s23:1")]}
    world.llm = FakeLLM(grades=[grade(["act854:s23:1"])], answer=answer, checks=list(checks))
    events = run(settings)
    grounding = [d for n, d in events if n == "trace" and d["node"] == "grounding" and d["status"] == "end"]
    return final(events), (grounding[0]["data"]["sentences"] if grounding else None)


def test_clean_answer_passes(settings, world) -> None:
    out, _ = ask(settings, world, "The fine is RM500,000 [act854:s23:1].")
    assert out["answer"]["label"] == PASSED and "⚠" not in out["answer"]["text"]


def test_fake_citation_id_is_flagged(settings, world) -> None:
    out, sentences = ask(settings, world, "The fine is RM500,000 [act854:s23:1]. The fee is RM50 [act854:s29:2].")
    assert out["answer"]["label"] == "Automated source check flagged 1 sentence"
    assert "The fee is RM50 [act854:s29:2]. ⚠ [cites act854:s29:2, which was not among the sources]" in out["answer"]["text"]
    assert [bool(s["problems"]) for s in sentences] == [False, True]


def test_unsupported_claim_is_flagged(settings, world) -> None:
    out, _ = ask(settings, world, "The fine is RM500,000 [act854:s23:1]. Jail is 20 years [act854:s23:1].",
                 checks=[[True, False]])
    assert "Jail is 20 years [act854:s23:1]. ⚠ [not supported by the cited source]" in out["answer"]["text"]
    assert out["answer"]["label"] == "Automated source check flagged 1 sentence"


def test_uncited_factual_sentence_is_flagged(settings, world) -> None:
    out, _ = ask(settings, world, "The fine is RM500,000 [act854:s23:1]. Directors are also liable.")
    assert "Directors are also liable. ⚠ [no citation]" in out["answer"]["text"]


def test_checker_output_invalid_says_incomplete_not_passed(settings, world) -> None:
    out, _ = ask(settings, world, "The fine is RM500,000 [act854:s23:1].", checks=[[True, True], [True, True]])
    assert out["answer"]["label"] == INCOMPLETE  # wrong length twice (schema asks for exactly 1)


def test_not_covered_statements_need_no_citation() -> None:
    assert code_problems("The licence fee amount is not covered by the provided documents.", set()) == []
    assert code_problems("The penalties are:", set()) == []
    assert code_problems("Directors are liable.", set()) == ["no citation"]


def test_sentence_split_keeps_citation_after_full_stop_and_list_lines() -> None:
    assert split_sentences("First fact. [a:s1:1] Second fact [a:s2:1].\n(a) item one [a:s3:1];") == [
        ["First fact. [a:s1:1]", "Second fact [a:s2:1]."], ["(a) item one [a:s3:1];"]]


def test_uncited_sentence_cannot_hide_behind_previous_citation() -> None:
    [[first, second]] = split_sentences("The fine is RM500,000. [a:s1:1] Directors are liable too.")
    assert code_problems(first, {"a:s1:1"}) == [] and code_problems(second, {"a:s1:1"}) == ["no citation"]


def test_subsection_style_citation_resolves_to_its_single_chunk() -> None:
    from app.pipeline.core import cited_ids
    assert cited_ids("Six hours [pua220:r2:2].", {"pua220:r2:1", "act854:s23:1"}) == ["pua220:r2:1"]
    assert code_problems("Six hours [pua220:r2:2].", {"pua220:r2:1"}) == []


def test_ambiguous_section_is_not_remapped() -> None:
    # s4 has one chunk per definition: 's4:9' must not be guessed into s4:7 or s4:8
    assert code_problems("Defined term [act854:s4:9].", {"act854:s4:7", "act854:s4:8"}) == [
        "cites act854:s4:9, which was not among the sources"]

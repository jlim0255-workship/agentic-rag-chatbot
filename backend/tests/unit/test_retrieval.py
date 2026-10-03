from __future__ import annotations

from app.pipeline.core import cited_ids, number_markers
from app.pipeline.nodes.generate import build_prompt
from app.retrieval.bm25 import bm25_scores, tokenize
from app.retrieval.search import Retrieved, rrf


def retrieved(id_: str, section: str, subsection: str | None = None, pages=(26, 26)) -> Retrieved:
    return Retrieved(id=id_, short_name=id_.split(":")[0].upper(), section=section, subsection=subsection,
                     heading="Heading", part=None, page_start=pages[0], page_end=pages[1], text="text", score=0.1)


# --- BM25 ---

def test_tokenize() -> None:
    assert tokenize("Section 23: Duty to NOTIFY—(1)") == ["section", "23", "duty", "to", "notify", "1"]


def test_bm25_ranks_matching_doc_first_and_ignores_unmatched() -> None:
    docs = ["duty to notify cyber security incident", "licensing of service provider", "cyber security committee"]
    scores = bm25_scores("notify incident", docs)
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0


def test_bm25_rare_term_outweighs_common_term() -> None:
    docs = ["cyber security penalty", "cyber security", "cyber security", "cyber security"]
    common, rare = bm25_scores("cyber", docs)[0], bm25_scores("penalty", docs)[0]
    assert rare > common


def test_bm25_empty_corpus() -> None:
    assert bm25_scores("anything", []) == []


# --- RRF ---

def test_rrf_known_order() -> None:
    # b is 2nd and 1st -> 1/62 + 1/61; a is 1st and absent -> 1/61; c is 3rd and 2nd -> 1/63 + 1/62
    fused = rrf([["a", "b", "c"], ["b", "c"]], k=60)
    assert [id_ for id_, _ in fused] == ["b", "c", "a"]
    assert abs(dict(fused)["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_rrf_unique_ids() -> None:
    assert len(rrf([["a", "b"], ["b", "a"], ["a"]], k=60)) == 2


# --- citations ---

def test_cited_ids_variants_seen_from_the_model() -> None:
    answer = ("Report within six hours [pua220:r2:1]. Fine up to RM500,000 [act854:s23:1, act854:s23:1]. "
              "Audit every two years [pua219:r3:1(b)]. See [act854:schedule:1; act854:s4:18]. Not an id [1].")
    assert cited_ids(answer) == ["pua220:r2:1", "act854:s23:1", "pua219:r3:1", "act854:schedule:1", "act854:s4:18"]


def test_markers_numbered_as_the_citations_list() -> None:
    given = {"pua220:r2:1", "act854:s23:1"}
    answer = ("Report within six hours [pua220:r2:1].\n* Fine up to RM500,000 [act854:s23:1, pua220:r2:1].\n"
              "* Particulars [pua220:r2:2(c)(i)].\n* Made up [act854:s99:1].\n* Not an id [sic].")
    order = [i for i in cited_ids(answer, given) if i in given]
    assert order == ["pua220:r2:1", "act854:s23:1"]
    assert number_markers(answer, order, given) == (
        "Report within six hours [1].\n* Fine up to RM500,000 [2, 1].\n"
        "* Particulars [1].\n* Made up.\n* Not an id [sic].")  # misread id resolved; unknown id dropped


def test_labels_and_pages() -> None:
    assert retrieved("act854:s23:1", "23").label == "ACT854 s23"
    assert retrieved("pua220:r2:1", "2", pages=(5, 7)).label == "PUA220 reg 2"
    assert retrieved("pua220:r2:1", "2", pages=(5, 7)).pages == "5–7"
    assert retrieved("act854:s4:13", "4", "computer").label == "ACT854 s4 “computer”"
    assert retrieved("act854:schedule:1", "Schedule").label == "ACT854 Schedule"


def test_prompt_contains_ids_and_question() -> None:
    prompt = build_prompt("How fast?", [retrieved("pua220:r2:1", "2")])
    assert "[pua220:r2:1] PUA220 reg 2 — Heading\ntext" in prompt
    assert prompt.rstrip().endswith("Question: How fast?\nAnswer:")

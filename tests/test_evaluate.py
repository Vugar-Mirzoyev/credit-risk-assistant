"""Tests for retrieval scoring.

The metric decides which retriever the report recommends, so it is checked
against hand-worked cases. The rule admitting a section's official
interpretation as an answer gets the most attention: it is the one piece of
generosity in the scoring, and generosity that is slightly too wide would
flatter every retriever at once and go unnoticed.
"""

from __future__ import annotations

import pytest

from src.rag.chunk import Chunk
from src.rag.evaluate import (
    Question,
    answers,
    comparison_markdown,
    evaluate_retriever,
    load_questions,
)
from src.rag.retrieve import Hit

QUESTIONS = (
    Question("q1", "first?", ("12 CFR 1002.9",), "lexical", "notice"),
    Question("q2", "second?", ("12 CFR 1002.2",), "paraphrase", "definition"),
)


def _chunk(citation: str, chunk_id: str = "c") -> Chunk:
    return Chunk(
        chunk_id=f"{citation}#{chunk_id}",
        part="1002",
        citation=citation,
        heading="h",
        text="t",
        position=0,
    )


class _Scripted:
    """Returns a fixed list of citations, so scoring can be tested alone."""

    name = "scripted"

    def __init__(self, per_query):
        self.per_query = per_query

    def search(self, query, k):
        citations = self.per_query.get(query, [])[:k]
        return [
            Hit(chunk=_chunk(c, str(i)), score=1.0, rank=i)
            for i, c in enumerate(citations, start=1)
        ]


def test_a_section_answers_a_question_about_itself():
    assert answers(["12 CFR 1002.9"], "12 CFR 1002.9")


def test_the_commentary_on_a_section_answers_a_question_about_it():
    # Asked what a finance charge is, the Bureau's interpretation of the
    # provision is a real answer, and scoring it wrong would understate every
    # retriever equally.
    assert answers(["12 CFR 1026.4"], "12 CFR 1026 Supp. I 4(a)")
    assert answers(["12 CFR 1002.9"], "12 CFR 1002 Supp. I 1002.9")
    assert answers(["12 CFR 1002.2"], "12 CFR 1002 Supp. I 2(p)")


def test_the_commentary_on_a_different_section_does_not():
    # The generous rule has to stop somewhere, and the obvious place it could
    # leak is a numeric prefix: 4 must not admit 40, and 1 must not admit 104.
    assert not answers(["12 CFR 1026.4"], "12 CFR 1026 Supp. I 40(f)")
    assert not answers(["12 CFR 1002.1"], "12 CFR 1002 Supp. I 104(b)(9)")
    assert not answers(["12 CFR 1002.9"], "12 CFR 1002 Supp. I 108(d)")


def test_commentary_from_another_part_does_not_count():
    assert not answers(["12 CFR 1002.9"], "12 CFR 1022 Supp. I 9(a)")


def test_any_of_several_accepted_citations_is_enough():
    # A question a person could answer from either section is answered by
    # either; requiring both would score a helpful result as a failure.
    assert answers(["12 CFR 1026.17", "12 CFR 1026.18"], "12 CFR 1026.18")


def test_recall_counts_a_question_once_however_many_chunks_matched():
    retriever = _Scripted(
        {
            "first?": ["12 CFR 1002.9", "12 CFR 1002.9", "12 CFR 1002.9"],
            "second?": ["12 CFR 9999.9"],
        }
    )

    scores = evaluate_retriever(retriever, QUESTIONS, k=5)

    assert scores.recall == pytest.approx(0.5)
    assert scores.questions == 2


def test_mrr_reflects_how_far_down_the_answer_was():
    retriever = _Scripted(
        {
            "first?": ["12 CFR 0000.0", "12 CFR 1002.9"],  # rank 2
            "second?": ["12 CFR 1002.2"],  # rank 1
        }
    )

    scores = evaluate_retriever(retriever, QUESTIONS, k=5)

    assert scores.mrr == pytest.approx((0.5 + 1.0) / 2)


def test_scoring_is_by_section_not_by_passage():
    # Five chunks of the right section is one answer, and the distinct-section
    # count is what tells the reader how much they actually got.
    retriever = _Scripted({"first?": ["12 CFR 1002.9"] * 5, "second?": []})

    scores = evaluate_retriever(retriever, QUESTIONS[:1], k=5)

    assert scores.recall == pytest.approx(1.0)
    assert scores.sections_per_query == pytest.approx(1.0)


def test_failures_are_listed_with_what_came_back_instead():
    retriever = _Scripted({"first?": ["12 CFR 5555.5"], "second?": ["12 CFR 1002.2"]})

    scores = evaluate_retriever(retriever, QUESTIONS, k=5)

    assert len(scores.failures) == 1
    assert scores.failures[0].question.id == "q1"
    table = scores.failure_markdown()
    assert "12 CFR 5555.5" in table
    assert "12 CFR 1002.9" in table


def test_results_split_by_kind():
    retriever = _Scripted({"first?": ["12 CFR 1002.9"], "second?": ["12 CFR 0.0"]})

    scores = evaluate_retriever(retriever, QUESTIONS, k=5)

    assert scores.by_kind("lexical").recall == pytest.approx(1.0)
    assert scores.by_kind("paraphrase").recall == pytest.approx(0.0)


def test_a_comparison_table_carries_the_split():
    retriever = _Scripted({"first?": ["12 CFR 1002.9"], "second?": ["12 CFR 1002.2"]})
    scores = evaluate_retriever(retriever, QUESTIONS, k=5)

    table = comparison_markdown([scores])

    assert "lexical recall" in table
    assert "paraphrase recall" in table


def test_an_empty_question_set_raises():
    with pytest.raises(ValueError, match="question set is empty"):
        evaluate_retriever(_Scripted({}), [], k=5)


def _write(tmp_path, body: str):
    path = tmp_path / "questions.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_question_without_a_citation_is_refused(tmp_path):
    body = """
questions:
  - id: q1
    question: what?
    citations: []
    kind: lexical
    evidence: something
"""
    with pytest.raises(ValueError, match="missing"):
        load_questions(_write(tmp_path, body))


def test_a_question_without_evidence_is_refused(tmp_path):
    # Evidence is what proves the citation is the right one. Without it a
    # question could point anywhere and every number would be wrong quietly.
    body = """
questions:
  - id: q1
    question: what?
    citations: ["12 CFR 1002.9"]
    kind: lexical
"""
    with pytest.raises(ValueError, match="missing"):
        load_questions(_write(tmp_path, body))


def test_an_unknown_kind_is_refused(tmp_path):
    body = """
questions:
  - id: q1
    question: what?
    citations: ["12 CFR 1002.9"]
    kind: tricky
    evidence: something
"""
    with pytest.raises(ValueError, match="lexical"):
        load_questions(_write(tmp_path, body))


def test_duplicate_ids_are_refused(tmp_path):
    body = """
questions:
  - id: q1
    question: a?
    citations: ["12 CFR 1002.9"]
    kind: lexical
    evidence: x
  - id: q1
    question: b?
    citations: ["12 CFR 1002.2"]
    kind: lexical
    evidence: y
"""
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(_write(tmp_path, body))


def test_a_missing_question_file_says_what_is_needed(tmp_path):
    with pytest.raises(FileNotFoundError, match="written by\\s+hand"):
        load_questions(tmp_path / "absent.yaml")


def test_the_committed_question_set_is_answerable_from_the_corpus():
    # The ground truth polices itself: every cited section must exist, and must
    # contain the evidence phrase that proves it answers the question. Without
    # this the retrieval numbers could be measured against a key that is simply
    # wrong, and nothing in the output would show it.
    from src.rag.corpus import load_corpus

    corpus = load_corpus("data/policies")
    questions = load_questions("config/retrieval_questions.yaml")

    assert len(questions) >= 40
    for question in questions:
        sections = [corpus.by_citation(c) for c in question.citations]
        assert all(s is not None for s in sections), question.id
        haystack = " ".join(s.heading + " " + s.text for s in sections).lower()
        assert question.evidence.lower() in haystack, question.id


def test_the_committed_set_is_not_all_one_kind():
    # If every question used the section's own vocabulary, BM25 would win by
    # construction and the D7 comparison would measure nothing.
    questions = load_questions("config/retrieval_questions.yaml")
    kinds = {q.kind for q in questions}

    assert kinds == {"lexical", "paraphrase"}
    paraphrase = sum(1 for q in questions if q.kind == "paraphrase")
    assert 0.4 < paraphrase / len(questions) < 0.8

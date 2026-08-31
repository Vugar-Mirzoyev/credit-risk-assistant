"""Tests for retrieval.

BM25 is the number every other retriever is judged against, so it is checked
against its own formula on counts small enough to work out by hand rather than
against remembered output. The dense retriever is exercised with supplied
vectors, which keeps the suite free of a model download and lets the ranking
logic be tested for what it is: cosine similarity and an ordering.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.rag.chunk import Chunk
from src.rag.retrieve import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    retrieved_citations,
    tokenize,
)


def _chunk(chunk_id: str, text: str, citation: str = "12 CFR 1002.1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        part="1002",
        citation=citation,
        heading="heading",
        text=text,
        position=0,
    )


CHUNKS = [
    _chunk("a", "adverse action notice must state specific reasons", "12 CFR 1002.9"),
    _chunk("b", "credit scoring system empirically derived and sound", "12 CFR 1002.2"),
    _chunk("c", "the creditor shall retain records for months", "12 CFR 1002.12"),
]


def test_section_numbers_survive_tokenisation():
    # A query naming a section has to be able to match it. Splitting on the dot
    # would turn 1002.9 into two useless numbers.
    assert "1002.9" in tokenize("What does § 1002.9 require?")
    assert "1026.37" in tokenize("see 1026.37(a)")
    assert tokenize("Adverse ACTION") == ["adverse", "action"]


def test_bm25_matches_its_formula_by_hand():
    # Two documents, one term. N=2, the term appears in one of them.
    chunks = [_chunk("x", "adverse adverse"), _chunk("y", "unrelated words here")]
    k1, b = 1.5, 0.75
    retriever = Bm25Retriever(chunks, k1=k1, b=b)

    lengths = [2, 3]
    average = sum(lengths) / 2
    idf = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    frequency = 2
    expected = (
        idf
        * frequency
        * (k1 + 1)
        / (frequency + k1 * (1 - b + b * lengths[0] / average))
    )

    scores = retriever.scores("adverse")
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == pytest.approx(0.0)


def test_a_term_in_every_document_is_worth_almost_nothing():
    # This is why no stop-word list is needed: IDF already does that job. Note
    # "almost" - the Lucene form ln(1 + (N-n+0.5)/(n+0.5)) drives a ubiquitous
    # term towards zero without reaching it, where the classical form would go
    # negative and start subtracting relevance.
    # Twenty documents: "the" is in all of them, "creditor" in one. A two
    # document corpus cannot show this, because with N=2 no term can be rare.
    chunks = [_chunk(str(i), f"the applicant word{i}") for i in range(19)]
    chunks.append(_chunk("target", "the creditor"))
    retriever = Bm25Retriever(chunks)

    everywhere = retriever.scores("the")
    distinctive = retriever.scores("creditor")

    assert (everywhere > 0).all()
    assert everywhere.max() < 0.1
    assert distinctive.max() > 20 * everywhere.max()


def test_idf_never_goes_negative():
    # A term in every one of many documents would score negative under the
    # classical IDF, which would mean a match actively pushing a passage down
    # the ranking.
    chunks = [_chunk(str(i), "the creditor shall") for i in range(50)]
    retriever = Bm25Retriever(chunks)

    assert (retriever.scores("the creditor shall") >= 0).all()


def test_an_unknown_word_scores_nothing():
    retriever = Bm25Retriever(CHUNKS)

    assert retriever.scores("supercalifragilistic") == pytest.approx(np.zeros(3))


def test_bm25_finds_the_obvious_passage():
    retriever = Bm25Retriever(CHUNKS)

    hits = retriever.search("adverse action specific reasons", k=2)

    assert hits[0].citation == "12 CFR 1002.9"
    assert hits[0].rank == 1
    assert hits[1].rank == 2
    assert hits[0].score > hits[1].score


def test_a_longer_passage_is_penalised_for_its_length():
    # Same single occurrence, different lengths: b > 0 means the shorter one
    # wins, which is the whole point of length normalisation.
    short = _chunk("short", "adverse action")
    long = _chunk("long", "adverse action " + "filler " * 40)
    retriever = Bm25Retriever([short, long])

    scores = retriever.scores("adverse")
    assert scores[0] > scores[1]


def test_search_never_returns_more_than_exists():
    retriever = Bm25Retriever(CHUNKS)

    assert len(retriever.search("creditor", k=99)) == 3


def test_asking_for_nothing_raises():
    retriever = Bm25Retriever(CHUNKS)

    with pytest.raises(ValueError, match="has to return something"):
        retriever.search("creditor", k=0)


def test_an_empty_index_raises():
    with pytest.raises(ValueError, match="No chunks"):
        Bm25Retriever([])


def _dense(vectors) -> DenseRetriever:
    return DenseRetriever(
        CHUNKS, model_name="unused-in-this-test", embeddings=np.array(vectors)
    )


def test_dense_ranks_by_cosine_similarity(monkeypatch):
    retriever = _dense([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    monkeypatch.setattr(
        retriever, "_encode", lambda texts, batch: np.array([[1.0, 0.0]])
    )

    hits = retriever.search("anything", k=3)

    assert hits[0].chunk.chunk_id == "a"  # identical direction
    assert hits[0].score == pytest.approx(1.0)
    assert hits[-1].chunk.chunk_id == "b"  # orthogonal
    assert hits[-1].score == pytest.approx(0.0, abs=1e-6)


def test_dense_normalises_so_length_does_not_decide(monkeypatch):
    # A chunk whose vector merely has a larger magnitude must not outrank one
    # pointing the same way. Cosine, not dot product.
    retriever = _dense([[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(
        retriever, "_encode", lambda texts, batch: np.array([[1.0, 0.0]])
    )

    hits = retriever.search("anything", k=2)

    assert hits[0].score == pytest.approx(hits[1].score)


def test_dense_refuses_vectors_that_do_not_line_up():
    with pytest.raises(ValueError, match="row for row"):
        DenseRetriever(CHUNKS, model_name="x", embeddings=np.zeros((2, 4)))


class _Fixed:
    """A retriever with a fixed answer, for testing fusion alone."""

    def __init__(self, name, order):
        self.name = name
        self.order = order

    def search(self, query, k):
        from src.rag.retrieve import Hit

        return [
            Hit(chunk=c, score=0.0, rank=rank)
            for rank, c in enumerate(self.order[:k], start=1)
        ]


def test_fusion_rewards_agreement_between_retrievers():
    # 'b' is second on both lists and never first; fusion should still lift it
    # over a chunk that one retriever ranked first and the other ignored.
    first = _Fixed("one", [CHUNKS[0], CHUNKS[1]])
    second = _Fixed("two", [CHUNKS[2], CHUNKS[1]])
    hybrid = HybridRetriever([first, second], rrf_k=1, depth=2)

    hits = hybrid.search("anything", k=3)
    ranking = [h.chunk.chunk_id for h in hits]

    assert ranking[0] == "b"


def test_fusion_uses_ranks_not_scores():
    # The scores here are wildly different in scale; fusion must ignore them.
    from src.rag.retrieve import Hit

    class Loud:
        name = "loud"

        def search(self, query, k):
            return [Hit(chunk=CHUNKS[0], score=1e9, rank=1)]

    class Quiet:
        name = "quiet"

        def search(self, query, k):
            return [Hit(chunk=CHUNKS[1], score=1e-9, rank=1)]

    hybrid = HybridRetriever([Loud(), Quiet()], rrf_k=60, depth=5)
    hits = hybrid.search("anything", k=2)

    assert hits[0].score == pytest.approx(hits[1].score)


def test_fusion_needs_something_to_fuse():
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridRetriever([])


def test_citations_are_deduplicated_in_rank_order():
    # Several chunks of one section is the normal case, and a caller counting
    # sections must not count it five times.
    from src.rag.retrieve import Hit

    hits = [
        Hit(chunk=CHUNKS[0], score=3.0, rank=1),
        Hit(chunk=CHUNKS[0], score=2.0, rank=2),
        Hit(chunk=CHUNKS[1], score=1.0, rank=3),
    ]

    assert retrieved_citations(hits) == ["12 CFR 1002.9", "12 CFR 1002.2"]

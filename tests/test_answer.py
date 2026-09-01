"""Tests for the generation step.

None of these call a model. The parts worth protecting are the parts that run
without one: the prompt that tells the model to refuse rather than guess, and
the check that catches a fabricated citation.
"""

from __future__ import annotations

import pytest

from src.config import AnswerConfig
from src.rag.answer import (
    PROMPT_TEMPLATE,
    REFUSAL,
    Answer,
    AnswerReport,
    build_prompt,
    cited_sections,
    generate,
)
from src.rag.chunk import Chunk
from src.rag.retrieve import Hit

CONFIG = AnswerConfig(
    base_url="https://example.invalid/v1",
    model="test-model",
    api_key_env="TEST_LLM_KEY_THAT_IS_NOT_SET",
    temperature=0.0,
    max_tokens=100,
    max_context_words=40,
)


def _hit(citation: str, text: str, rank: int = 1) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=f"{citation}#{rank}",
            part="1002",
            citation=citation,
            heading=f"{citation} heading",
            text=text,
            position=0,
        ),
        score=1.0,
        rank=rank,
    )


HITS = [
    _hit("12 CFR 1002.9", "A creditor shall notify within 30 days.", 1),
    _hit("12 CFR 1002.2", "Adverse action means a refusal to grant credit.", 2),
]


def test_the_prompt_forbids_outside_knowledge_and_allows_refusal():
    # The instructions that matter in a compliance setting. If either is edited
    # away, the assistant starts inventing regulation. Checked on the rendered
    # prompt rather than the template, because the refusal wording the model is
    # given has to be the exact string the code later looks for.
    prompt = build_prompt("When?", HITS, max_context_words=100)

    assert "Use only the passages provided" in prompt
    assert REFUSAL in prompt
    assert "Cite the section" in prompt
    assert "{refusal}" in PROMPT_TEMPLATE


def test_the_prompt_carries_the_passages_and_their_citations():
    prompt = build_prompt("When?", HITS, max_context_words=100)

    assert "[12 CFR 1002.9]" in prompt
    assert "notify within 30 days" in prompt
    assert "Question: When?" in prompt


def test_the_context_is_truncated_to_a_stated_budget():
    long = _hit("12 CFR 1026.4", " ".join(["word"] * 500))

    prompt = build_prompt("What?", [long], max_context_words=10)
    body = prompt.split("Passages:")[1].split("Question:")[0]

    assert body.count("word") == 10


def test_answering_with_no_passages_is_refused():
    # An empty context is an invitation to invent one.
    with pytest.raises(ValueError, match="No passages"):
        build_prompt("What?", [], max_context_words=100)


def test_citations_are_extracted_in_both_forms_a_model_writes():
    text = "Under 12 CFR 1002.9, and see § 1002.2 as well."

    assert cited_sections(text) == ("12 CFR 1002.9", "12 CFR 1002.2")


def test_an_answer_with_no_citation_extracts_nothing():
    assert cited_sections("It depends on the circumstances.") == ()


def test_a_fabricated_citation_is_caught():
    # The failure that matters: a section the model invented and never saw.
    answer = Answer(
        question="q",
        prompt="p",
        context=tuple(HITS),
        text="See 12 CFR 1002.9 and also 12 CFR 1099.1.",
        model="m",
    )

    assert answer.citation_faithfulness == pytest.approx(0.5)


def test_an_answer_citing_only_what_it_was_given_is_fully_grounded():
    answer = Answer(
        question="q",
        prompt="p",
        context=tuple(HITS),
        text="Within 30 days, per 12 CFR 1002.9.",
        model="m",
    )

    assert answer.citation_faithfulness == pytest.approx(1.0)


def test_a_refusal_is_recognised_and_not_punished():
    # Refusing is the behaviour the prompt asks for. Scoring it zero would
    # reward a model that guesses over one that declines.
    answer = Answer(
        question="q", prompt="p", context=tuple(HITS), text=REFUSAL, model="m"
    )

    assert answer.refused
    assert answer.citation_faithfulness is None


def test_nothing_generated_means_no_score_rather_than_a_zero():
    answer = Answer(question="q", prompt="p", context=tuple(HITS))

    assert not answer.generated
    assert answer.citation_faithfulness is None
    assert "not generated" in answer.summary()


def test_a_missing_key_is_reported_not_raised():
    # The whole design rests on this: no key, no crash, and the retrieval
    # figures stand regardless.
    text, error = generate("prompt", CONFIG)

    assert text is None
    assert CONFIG.api_key_env in error


def test_the_report_says_why_generation_did_not_run():
    report = AnswerReport(available=False, reason="LLM_API_KEY is not set")

    summary = report.summary()
    assert "not run" in summary
    assert "stands without it" in summary


def test_the_report_averages_only_the_answers_that_cited_something():
    grounded = Answer(
        question="a", prompt="p", context=tuple(HITS),
        text="See 12 CFR 1002.9.", model="m",
    )
    invented = Answer(
        question="b", prompt="p", context=tuple(HITS),
        text="See 12 CFR 9999.9.", model="m",
    )
    refusal = Answer(
        question="c", prompt="p", context=tuple(HITS), text=REFUSAL, model="m"
    )
    report = AnswerReport(answers=(grounded, invented, refusal))

    assert report.generated == 3
    assert report.refusals == 1
    assert report.grounded_share == pytest.approx(0.5)


def test_retrieved_citations_are_deduplicated():
    answer = Answer(
        question="q",
        prompt="p",
        context=(HITS[0], _hit("12 CFR 1002.9", "more text", 2), HITS[1]),
    )

    assert answer.retrieved_citations == ("12 CFR 1002.9", "12 CFR 1002.2")

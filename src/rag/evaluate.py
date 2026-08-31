"""D8: scoring retrieval against a hand-built set, failures and all.

Two metrics, both computed over *sections* rather than passages:

``recall@k``
    The share of questions where at least one section that answers it appears
    in the top k results. A question with two acceptable citations counts as
    found if either comes back, because a person reading the answer needs one
    of them, not both.
``MRR``
    Mean reciprocal rank of the first correct section. Recall says whether the
    answer was in the list; MRR says how far down.

**Sections, not passages.** A retriever that returns five chunks of the right
section has answered the question once, not five times. Scoring passages would
reward a retriever for clustering on one section and punish one that spread its
guesses, which is backwards.

That clustering is real and worth watching, so every result also carries the
average number of distinct sections the top k actually covered. Five results
that are five chunks of § 1002.9 give the reader one section to check; five
results spanning five sections give five. The number is not a score, it is
context for reading the two that are.

**The official interpretation of a section counts as an answer to a question
about that section.** Asked what a finance charge is, a retriever returning
the Bureau's interpretation of § 1026.4 has served the reader, and marking that
wrong would understate every retriever equally and misrepresent all of them.
The rule is mechanical rather than a list of hand-added citations: the
commentary on § X answers questions about § X, and nothing else is admitted.
Written out by hand it would be fifteen extra citations nobody could check;
written as a rule it is one function with its own tests.

**Every failure is listed by name.** An average hides the questions that
matter, and the questions a retriever misses say more about it than the ones
it gets. `reports/03_retrieval.md` prints each miss with what came back
instead, the same way the D2 audit prints a reason for every excluded column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from src.rag.retrieve import Retriever, retrieved_citations

#: `12 CFR 1002.9` -> part 1002, provision 9.
_RULE = re.compile(r"12 CFR (\d+)\.(.+)")

REQUIRED_FIELDS = ("id", "question", "citations", "kind", "evidence")
KINDS = ("lexical", "paraphrase")


@dataclass(frozen=True)
class Question:
    """One evaluation item: what was asked, and what answers it."""

    id: str
    question: str
    citations: tuple[str, ...]
    kind: str
    evidence: str


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    retrieved: tuple[str, ...]
    rank: int | None

    @property
    def found(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class RetrievalScores:
    """What one retriever achieved on the whole set."""

    retriever: str
    k: int
    results: tuple[QuestionResult, ...] = field(default_factory=tuple)

    @property
    def questions(self) -> int:
        return len(self.results)

    @property
    def recall(self) -> float:
        return self._mean(r.found for r in self.results)

    @property
    def mrr(self) -> float:
        return self._mean(r.reciprocal_rank for r in self.results)

    @property
    def sections_per_query(self) -> float:
        return self._mean(len(r.retrieved) for r in self.results)

    @property
    def failures(self) -> tuple[QuestionResult, ...]:
        return tuple(r for r in self.results if not r.found)

    def by_kind(self, kind: str) -> "RetrievalScores":
        return RetrievalScores(
            retriever=self.retriever,
            k=self.k,
            results=tuple(r for r in self.results if r.question.kind == kind),
        )

    @staticmethod
    def _mean(values) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> str:
        lines = [
            f"retriever               {self.retriever}",
            f"questions               {self.questions}",
            f"recall@{self.k:<17}{self.recall:.3f}",
            f"MRR                     {self.mrr:.3f}",
            f"sections per query      {self.sections_per_query:.1f} of {self.k}",
        ]
        for kind in KINDS:
            subset = self.by_kind(kind)
            if subset.questions:
                lines.append(
                    f"  {kind:<20}recall {subset.recall:.3f}  "
                    f"MRR {subset.mrr:.3f}  ({subset.questions} questions)"
                )
        return "\n".join(lines)

    def failure_markdown(self) -> str:
        if not self.failures:
            return "No misses."
        lines = [
            "| Question | Kind | Expected | Returned instead |",
            "|---|---|---|---|",
        ]
        for result in self.failures:
            expected = ", ".join(result.question.citations)
            returned = ", ".join(result.retrieved[:3]) or "nothing"
            lines.append(
                f"| {result.question.question} | {result.question.kind} | "
                f"{expected} | {returned} |"
            )
        return "\n".join(lines)


def answers(expected: Sequence[str], retrieved: str) -> bool:
    """Does this retrieved section answer a question expecting `expected`?

    True for the section itself, and for the official interpretation of it.
    The commentary on § 1026.4 is an answer to a question about § 1026.4; the
    commentary on § 1026.40 is not, which is why the match is anchored rather
    than a prefix test.
    """
    if retrieved in expected:
        return True
    for citation in expected:
        match = _RULE.fullmatch(citation)
        if not match:
            continue
        part, provision = match.groups()
        # The commentary keyed to the whole section, `1002 Supp. I 1002.9`.
        if retrieved == f"12 CFR {part} Supp. I {part}.{provision}":
            return True
        # The commentary keyed to a paragraph of it, `1002 Supp. I 9(a)`.
        if re.fullmatch(
            rf"12 CFR {part} Supp\. I {re.escape(provision)}(\(.+\))?", retrieved
        ):
            return True
    return False


def load_questions(path: Path | str) -> tuple[Question, ...]:
    """Read the evaluation set, refusing anything that cannot be scored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No evaluation set at {path}. D8 requires questions written by "
            "hand against the corpus; retrieval cannot be scored without them."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("questions") or []
    if not entries:
        raise ValueError(f"{path} contains no questions.")

    seen: set[str] = set()
    questions: list[Question] = []
    for entry in entries:
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise ValueError(
                f"Question {entry.get('id', '?')!r} in {path} is missing "
                f"{missing}. Every entry needs a citation to be scored against "
                "and evidence proving the citation is the right one."
            )
        if entry["kind"] not in KINDS:
            raise ValueError(
                f"Question {entry['id']!r} has kind {entry['kind']!r}; expected "
                f"one of {list(KINDS)}. The split into lexical and paraphrase "
                "is what keeps the retriever comparison from being decided by "
                "how the questions were worded."
            )
        if entry["id"] in seen:
            raise ValueError(f"Duplicate question id {entry['id']!r} in {path}.")
        seen.add(entry["id"])
        questions.append(
            Question(
                id=entry["id"],
                question=entry["question"],
                citations=tuple(entry["citations"]),
                kind=entry["kind"],
                evidence=entry["evidence"],
            )
        )
    return tuple(questions)


def evaluate_retriever(
    retriever: Retriever,
    questions: Sequence[Question],
    *,
    k: int,
    name: str | None = None,
) -> RetrievalScores:
    """Run every question and record where the right section landed."""
    if not questions:
        raise ValueError("Nothing to evaluate: the question set is empty.")

    results: list[QuestionResult] = []
    for question in questions:
        hits = retriever.search(question.question, k)
        citations = retrieved_citations(hits)
        rank = next(
            (
                position
                for position, citation in enumerate(citations, start=1)
                if answers(question.citations, citation)
            ),
            None,
        )
        results.append(
            QuestionResult(
                question=question, retrieved=tuple(citations), rank=rank
            )
        )

    return RetrievalScores(
        retriever=name or getattr(retriever, "name", "retriever"),
        k=k,
        results=tuple(results),
    )


def comparison_markdown(scores: Sequence[RetrievalScores]) -> str:
    """One row per retriever, with the lexical and paraphrase split."""
    lines = [
        "| Retriever | recall@k | MRR | lexical recall | paraphrase recall | "
        "sections/query |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        lexical = score.by_kind("lexical")
        paraphrase = score.by_kind("paraphrase")
        lines.append(
            f"| {score.retriever} | {score.recall:.3f} | {score.mrr:.3f} | "
            f"{lexical.recall:.3f} | {paraphrase.recall:.3f} | "
            f"{score.sections_per_query:.1f} |"
        )
    return "\n".join(lines)

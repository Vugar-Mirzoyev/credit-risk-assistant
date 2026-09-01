"""D11: the generation step, and the part of it that can be checked for free.

Most of what goes wrong in retrieval-augmented generation goes wrong in
retrieval, and retrieval is measurable without a language model. So generation
sits behind a configured key: the prompt is committed and readable, the
evaluation runs either way, and every retrieval figure in the project stands
without a credential. A repository whose headline numbers cannot be reproduced
without paying for them should say so rather than imply otherwise.

**The provider is configuration, not a dependency.** Any OpenAI-compatible
endpoint works, the key is read from the environment and never stored, and
nothing here is tied to one vendor.

**The prompt demands a citation and permits refusal.** In a compliance setting
an assistant that invents an answer is worse than one that declines, because
the reader has no way to check it and every reason to believe it. The prompt is
a module constant rather than a string assembled inside a function, so it can
be reviewed and diffed like any other artefact.

**Citation faithfulness is measured without a second model.** The usual way to
score faithfulness is to have a language model decompose the answer into claims
and check each one, which needs a second credential and produces a number
nobody can reproduce. There is a harder, cheaper check available for the
failure that matters most here: every section the answer cites must be one that
was actually retrieved. A fabricated `§ 1002.9` is caught by string comparison,
costs nothing, and is exactly the error a regulator would care about.

That check is narrower than faithfulness and is not a substitute for it. It
catches invented citations, not misread ones - an answer citing a real
retrieved section and describing it wrongly passes. Saying which is which is
the point.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Sequence

from src.config import AnswerConfig
from src.rag.retrieve import Hit, Retriever

#: The exact sentence the prompt asks for when the context does not answer the
#: question. Interpolated into the template rather than repeated in it, so the
#: constant the code checks for and the wording the model is given cannot drift
#: apart.
REFUSAL = "The provided passages do not answer this question."

#: Committed on purpose. This is the part of a generation system most likely to
#: be edited casually and least likely to be reviewed, so it lives where a diff
#: will show it.
PROMPT_TEMPLATE = """\
You are answering a question about United States consumer lending regulation.

Rules:
1. Use only the passages provided below. Do not use anything you know from
   elsewhere, even if you are confident it is correct.
2. Cite the section you relied on, in the form shown in the passage heading,
   for example "12 CFR 1002.9".
3. If the passages do not contain the answer, reply with exactly this
   sentence and nothing else: "{refusal}"
   Do not guess, and do not offer a partial answer dressed as a complete one.
4. Be brief. A regulator reads the citation, not the prose.

Passages:
{context}

Question: {question}

Answer:"""

#: The forms a model actually writes. Alternatives rather than one loose
#: pattern, because a pattern that stops at the first full stop truncates
#: `12 CFR 1002.9` to `12 CFR 1002` - and then every fabricated citation
#: silently matches a real section, which is the one thing this check exists
#: to catch.
_CITATION = re.compile(
    r"12 CFR \d+ Supp\. I [\w.()]+"
    r"|12 CFR Appendix \w+ to Part \d+"
    r"|12 CFR \d+\.\d+(?:\(\w+\))*"
    r"|§\s*\d+\.\d+(?:\(\w+\))*"
)

TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Answer:
    """What was asked, what was retrieved, and what came back."""

    question: str
    prompt: str
    context: tuple[Hit, ...]
    text: str | None = None
    model: str = ""
    error: str | None = None

    @property
    def generated(self) -> bool:
        return self.text is not None

    @property
    def retrieved_citations(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for hit in self.context:
            seen.setdefault(hit.citation, None)
        return tuple(seen)

    @property
    def cited(self) -> tuple[str, ...]:
        return cited_sections(self.text or "")

    @property
    def refused(self) -> bool:
        return bool(self.text) and REFUSAL.lower() in self.text.lower()

    @property
    def citation_faithfulness(self) -> float | None:
        """Share of the answer's citations that were actually retrieved.

        None when nothing was generated or nothing was cited: a refusal cites
        nothing and scoring it 0 would punish the behaviour the prompt asks for.
        """
        if not self.generated or not self.cited:
            return None
        retrieved = self.retrieved_citations
        grounded = sum(1 for c in self.cited if _matches_any(c, retrieved))
        return grounded / len(self.cited)

    def summary(self) -> str:
        lines = [
            f"question                {self.question}",
            f"passages retrieved      {len(self.context)} "
            f"({len(self.retrieved_citations)} sections)",
        ]
        if not self.generated:
            lines.append(
                f"answer                  not generated"
                + (f" ({self.error})" if self.error else "")
            )
            return "\n".join(lines)
        lines += [
            f"model                   {self.model}",
            f"refused                 {self.refused}",
            f"cited                   {', '.join(self.cited) or 'nothing'}",
        ]
        score = self.citation_faithfulness
        lines.append(
            "citation faithfulness   "
            + ("not applicable" if score is None else f"{score:.2f}")
        )
        return "\n".join(lines)


@dataclass
class AnswerReport:
    """Generation across a set of questions, or the reason there is none."""

    answers: tuple[Answer, ...] = field(default_factory=tuple)
    available: bool = True
    reason: str = ""

    @property
    def generated(self) -> int:
        return sum(1 for a in self.answers if a.generated)

    @property
    def refusals(self) -> int:
        return sum(1 for a in self.answers if a.refused)

    @property
    def grounded_share(self) -> float | None:
        scores = [
            a.citation_faithfulness
            for a in self.answers
            if a.citation_faithfulness is not None
        ]
        return sum(scores) / len(scores) if scores else None

    def summary(self) -> str:
        if not self.available:
            return (
                "generation              not run\n"
                f"reason                  {self.reason}\n"
                "  Every retrieval figure in this project stands without it."
            )
        score = self.grounded_share
        return (
            f"questions answered      {self.generated} of {len(self.answers)}\n"
            f"refusals                {self.refusals}\n"
            f"citation faithfulness   "
            + ("not applicable" if score is None else f"{score:.3f}")
        )


def cited_sections(text: str) -> tuple[str, ...]:
    """Every regulation reference an answer makes, normalised and deduplicated."""
    seen: dict[str, None] = {}
    for raw in _CITATION.findall(text):
        cleaned = " ".join(raw.replace("§", "").split()).strip(" .,;")
        if not cleaned:
            continue
        if not cleaned.startswith("12 CFR"):
            cleaned = f"12 CFR {cleaned}"
        seen.setdefault(cleaned, None)
    return tuple(seen)


def _matches_any(citation: str, retrieved: Sequence[str]) -> bool:
    """A cited section counts as grounded if it was among those retrieved.

    Compared on the leading rule reference so that `12 CFR 1002.9` matches a
    passage retrieved as `12 CFR 1002.9`, and a fabricated `12 CFR 1099.1`
    matches nothing.
    """
    return any(citation == r or r.startswith(citation) for r in retrieved)


def build_prompt(question: str, hits: Sequence[Hit], max_context_words: int) -> str:
    """Assemble the prompt, truncating the context to a stated budget."""
    if not hits:
        raise ValueError(
            "No passages to answer from. Generation runs on retrieval output; "
            "an empty context would only invite the model to invent one."
        )
    blocks: list[str] = []
    budget = max_context_words
    for hit in hits:
        words = hit.chunk.text.split()
        if budget <= 0:
            break
        body = " ".join(words[:budget])
        budget -= len(words[:budget])
        blocks.append(f"[{hit.citation}] {hit.chunk.heading}\n{body}")
    return PROMPT_TEMPLATE.format(
        context="\n\n".join(blocks), question=question, refusal=REFUSAL
    )


def generate(prompt: str, config: AnswerConfig) -> tuple[str | None, str | None]:
    """Call the configured endpoint. Returns `(text, error)`, never raises."""
    key = os.environ.get(config.api_key_env)
    if not key:
        return None, f"{config.api_key_env} is not set"

    payload = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.load(response)
        return body["choices"][0]["message"]["content"].strip(), None
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as error:
        return None, f"{type(error).__name__}: {error}"


def answer_question(
    question: str,
    retriever: Retriever,
    *,
    config: AnswerConfig,
    k: int,
) -> Answer:
    """Retrieve, build the prompt, and generate if a key is configured."""
    hits = retriever.search(question, k)
    prompt = build_prompt(question, hits, config.max_context_words)
    text, error = generate(prompt, config)
    return Answer(
        question=question,
        prompt=prompt,
        context=tuple(hits),
        text=text,
        model=config.model if text else "",
        error=error,
    )

"""D9: chunking, as a parameter to be measured rather than a choice to defend.

Chunk size decides what retrieval can possibly find. Too coarse and a section
about six unrelated things matches every query about any of them; too fine and
the sentence that answers the question arrives without the clause that makes it
mean something. There is no correct answer available from first principles, so
three strategies are implemented and swept against the evaluation set:

``section``
    One chunk per numbered section. Coarse, and the natural baseline: it is
    what you get if you do no chunking at all beyond the structure the
    regulation already has.
``paragraph``
    One chunk per paragraph. Fine, and it loses context: regulatory paragraphs
    lean on their neighbours constantly, and "(2) Exceptions. The following
    provisions do not apply" is not a retrievable idea on its own.
``window``
    Paragraphs accumulated to a target length with overlap between windows.
    The usual compromise, and the one the configuration defaults to.

**No chunk ever crosses a section boundary.** A chunk spanning § 1002.8 and
§ 1002.9 cannot be cited, and citation is the whole reason the corpus was
parsed structurally in the first place. The constraint costs a little packing
efficiency at the end of short sections and buys the property the project
exists to demonstrate.

**A paragraph shorter than the minimum is merged, never dropped.** Dropping it
would make the chunk statistics tidier and leave a hole in the corpus that
nothing downstream could see. The same reasoning as D1: text that is
inconvenient still has to be accounted for.

**The section heading is prepended to every chunk, and that is configurable.**
A chunk that begins "(2) Exceptions" is nearly meaningless in isolation, and
prepending "§ 1002.3 Limited exceptions for certain classes of transactions"
gives it a topic. It also makes every chunk in a section share the heading's
words, which flatters section-level retrieval, so it is a flag rather than a
fact: `reports/03_retrieval.md` reports the scores with and without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.config import ChunkConfig
from src.rag.corpus import Corpus, Section

STRATEGIES = ("section", "paragraph", "window")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage that still knows where it came from."""

    chunk_id: str
    part: str
    citation: str
    heading: str
    text: str
    position: int

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class ChunkReport:
    strategy: str
    chunks: int
    sections: int
    words: int
    shortest: int
    longest: int
    median: int
    prepend_heading: bool

    def summary(self) -> str:
        return (
            f"strategy                {self.strategy}\n"
            f"heading prepended       {self.prepend_heading}\n"
            f"sections                {self.sections}\n"
            f"chunks                  {self.chunks:,}\n"
            f"words                   {self.words:,}\n"
            f"words per chunk         {self.shortest} / {self.median} / "
            f"{self.longest}  (min / median / max)"
        )


def chunk_corpus(
    corpus: Corpus, config: ChunkConfig
) -> tuple[tuple[Chunk, ...], ChunkReport]:
    """Split every section, and report what the split produced."""
    if config.strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy {config.strategy!r}; expected one of "
            f"{list(STRATEGIES)}. The strategy is swept in the retrieval "
            "evaluation, so a typo here would quietly compare nothing."
        )

    chunks: list[Chunk] = []
    for section in corpus.sections:
        chunks.extend(chunk_section(section, config))

    lengths = sorted(c.words for c in chunks) or [0]
    report = ChunkReport(
        strategy=config.strategy,
        chunks=len(chunks),
        sections=len(corpus.sections),
        words=sum(lengths),
        shortest=lengths[0],
        longest=lengths[-1],
        median=lengths[len(lengths) // 2],
        prepend_heading=config.prepend_heading,
    )
    return tuple(chunks), report


def chunk_section(section: Section, config: ChunkConfig) -> list[Chunk]:
    """Split one section. The output never leaves this section."""
    if config.strategy == "section":
        bodies = [section.text] if section.text else []
    elif config.strategy == "paragraph":
        bodies = _merge_short(section.paragraphs, config.min_words)
    else:
        bodies = _windows(
            _merge_short(section.paragraphs, config.min_words),
            target_words=config.target_words,
            overlap_words=config.overlap_words,
        )

    return [
        Chunk(
            chunk_id=f"{section.number or section.part}#{position}",
            part=section.part,
            citation=section.citation,
            heading=section.heading,
            text=_with_heading(body, section.heading, config.prepend_heading),
            position=position,
        )
        for position, body in enumerate(bodies)
        if body.strip()
    ]


def _with_heading(body: str, heading: str, prepend: bool) -> str:
    if not prepend or not heading:
        return body
    return f"{heading}\n\n{body}"


def _merge_short(paragraphs: Sequence[str], min_words: int) -> list[str]:
    """Fold paragraphs below the minimum into the next one.

    Forward rather than backward so a heading-like lead-in stays attached to
    what it introduces. A trailing short paragraph with nothing after it is
    folded backward instead, because the alternative is losing it.
    """
    merged: list[str] = []
    carry = ""
    for paragraph in paragraphs:
        candidate = f"{carry} {paragraph}".strip() if carry else paragraph
        if len(candidate.split()) < min_words:
            carry = candidate
            continue
        merged.append(candidate)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}".strip()
        else:
            merged.append(carry)
    return merged


def _windows(
    paragraphs: Sequence[str], *, target_words: int, overlap_words: int
) -> list[str]:
    """Accumulate paragraphs to roughly `target_words`, overlapping by tail."""
    if target_words <= 0:
        raise ValueError(
            f"target_words is {target_words}; a window has to have a size."
        )
    if overlap_words >= target_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) is not smaller than target_words "
            f"({target_words}). Overlap at or above the window size would make "
            "each window start inside the previous one and never advance."
        )

    windows: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and length + words > target_words:
            windows.append("\n\n".join(current))
            tail = _tail_words("\n\n".join(current), overlap_words)
            current = [tail] if tail else []
            length = len(tail.split()) if tail else 0
        current.append(paragraph)
        length += words
    if current:
        windows.append("\n\n".join(current))
    return windows


def _tail_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    words = text.split()
    return " ".join(words[-count:]) if words else ""


def citations_of(chunks: Iterable[Chunk]) -> tuple[str, ...]:
    """The distinct sections a set of chunks points at, in first-seen order."""
    seen: dict[str, None] = {}
    for chunk in chunks:
        seen.setdefault(chunk.citation, None)
    return tuple(seen)

"""D7: three retrievers, so that any one of them can be judged.

An embedding retriever reported on its own is unfalsifiable. Nobody reading
"recall@5 of 0.82" can tell whether that is good, because there is nothing to
compare it against, and on this corpus there is a strong candidate: regulatory
text is dense with exact terms of art - "adverse action", "empirically
derived", "annual percentage rate" - and keyword search handles those better
than a sentence embedding often does.

So there are three, and the report carries all of them:

``Bm25Retriever``
    Keyword search, the baseline. Implemented here rather than imported, for
    the same reason PSI and WOE are: it is the number everything else is
    measured against, so it should be checkable line by line against the
    formula, and it is about forty lines of arithmetic.
``DenseRetriever``
    Sentence embeddings and exact cosine similarity.
``HybridRetriever``
    Reciprocal rank fusion of the two.

**Fusion combines ranks, not scores.** A BM25 score of 14.2 and a cosine
similarity of 0.61 are not on the same scale and there is no principled way to
put them on one. Normalising them would mean choosing a transformation nobody
could defend, and the choice would move the result. Reciprocal rank fusion
needs only the ordering.

**Search is exact, with no approximate index.** At five thousand chunks a
brute-force cosine over a 384-dimensional matrix takes under a millisecond, so
an approximate index would add a dependency and lose recall to save time that
does not exist. At a million chunks that trade reverses and something like
FAISS earns its place; saying which regime you are in is the useful part.

**No stemming and no stop-word list.** BM25's own IDF term already drives a
word appearing in every document towards zero weight, so a stop-word list
would be a second, arbitrary mechanism doing the same job. Stemming is a real
omission rather than a free one: without it "notification" does not match
"notify", and that gap is part of what the dense retriever is there to cover.
Reporting the comparison is how the omission stays visible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from src.rag.chunk import Chunk

#: Keeps `1002.9` and `1026.37(a)` as single tokens. A query naming a section
#: should match the section, and splitting on the dot would destroy that.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with where it placed and why."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def citation(self) -> str:
        return self.chunk.citation


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


class Bm25Retriever:
    """Okapi BM25 over the chunk texts.

    score(D, Q) = sum over q in Q of
        IDF(q) * f(q,D) * (k1 + 1)
                 / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

    with IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5)), the form that keeps
    the weight of a term appearing in almost every document at zero rather than
    letting it go negative.
    """

    name = "bm25"

    def __init__(
        self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self._documents = [tokenize(c.text) for c in self.chunks]
        self._lengths = np.array([len(d) for d in self._documents], dtype="float64")
        self._average_length = float(self._lengths.mean()) or 1.0

        self._frequencies: list[Counter] = [Counter(d) for d in self._documents]
        document_count = len(self._documents)
        appearances: Counter = Counter()
        for frequency in self._frequencies:
            appearances.update(frequency.keys())
        self._idf = {
            term: math.log(
                1.0 + (document_count - count + 0.5) / (count + 0.5)
            )
            for term, count in appearances.items()
        }
        # term -> array of (document index, frequency), so a query only touches
        # the documents that contain its words.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, frequency in enumerate(self._frequencies):
            for term, count in frequency.items():
                self._postings.setdefault(term, []).append((index, count))

    def scores(self, query: str) -> np.ndarray:
        totals = np.zeros(len(self.chunks), dtype="float64")
        normaliser = self.k1 * (
            1.0 - self.b + self.b * self._lengths / self._average_length
        )
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequency in self._postings[term]:
                totals[index] += (
                    idf
                    * frequency
                    * (self.k1 + 1.0)
                    / (frequency + normaliser[index])
                )
        return totals

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


class DenseRetriever:
    """Sentence embeddings with exact cosine similarity."""

    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str,
        batch_size: int = 256,
        embeddings: np.ndarray | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = list(chunks)
        self.model_name = model_name
        self._model = None
        if embeddings is None:
            embeddings = self._encode([c.text for c in self.chunks], batch_size)
        self.embeddings = _unit(np.asarray(embeddings, dtype="float32"))
        if len(self.embeddings) != len(self.chunks):
            raise ValueError(
                f"{len(self.embeddings)} embeddings for {len(self.chunks)} "
                "chunks. They must line up row for row."
            )

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype="float32",
        )

    def scores(self, query: str) -> np.ndarray:
        vector = _unit(self._encode([query], 1))[0]
        return self.embeddings @ vector

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


class HybridRetriever:
    """Reciprocal rank fusion over the other two.

    ``score(chunk) = sum over retrievers of 1 / (rrf_k + rank)``

    Rank-based on purpose: the two score scales have nothing to do with each
    other, and any function mapping one onto the other would be a choice
    affecting the result that nobody could justify.
    """

    name = "hybrid"

    def __init__(
        self, retrievers: Sequence[Retriever], *, rrf_k: int = 60, depth: int = 50
    ) -> None:
        if not retrievers:
            raise ValueError("Hybrid retrieval needs at least one retriever.")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        seen: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, self.depth):
                key = hit.chunk.chunk_id
                seen[key] = hit.chunk
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + hit.rank)
        ordered = sorted(fused.items(), key=lambda item: -item[1])[:k]
        return [
            Hit(chunk=seen[key], score=score, rank=rank)
            for rank, (key, score) in enumerate(ordered, start=1)
        ]


def _top_k(chunks: Sequence[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    if k <= 0:
        raise ValueError(f"k is {k}; a search has to return something.")
    k = min(k, len(chunks))
    # argpartition then sort only the k that survive: the corpus is small
    # enough that it hardly matters, but a full sort of every chunk per query
    # is wasted work in the sweep, which runs thousands of queries.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [
        Hit(chunk=chunks[index], score=float(scores[index]), rank=rank)
        for rank, index in enumerate(top, start=1)
    ]


def _unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def retrieved_citations(hits: Iterable[Hit]) -> list[str]:
    """The sections a result set points at, best rank first, no duplicates."""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.citation, None)
    return list(seen)

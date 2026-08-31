"""Embedding cache, so the chunking sweep is affordable.

Encoding the corpus takes about five minutes on a CPU. D9 sweeps three
chunking strategies against the evaluation set, and D7 runs three retrievers
over each, so without a cache the report costs a quarter of an hour of
recomputing vectors that have not changed.

The cache key is the model name and the exact text of every chunk. If either
moves, the key moves and the vectors are rebuilt: a cache that can serve stale
embeddings for edited text would be worse than no cache, because the numbers
would still look reasonable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from src.rag.chunk import Chunk


def cache_key(model_name: str, chunks: Sequence[Chunk]) -> str:
    """Model plus corpus content. Any edit to either invalidates the cache."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()[:16]


def load_cached(directory: Path | str, key: str) -> np.ndarray | None:
    path = Path(directory) / f"embeddings_{key}.npy"
    return np.load(path) if path.is_file() else None


def save_cached(directory: Path | str, key: str, embeddings: np.ndarray) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"embeddings_{key}.npy"
    np.save(path, embeddings)
    return path

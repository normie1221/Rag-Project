"""FAISS-backed vector store for audio transcript chunks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Chunk:
    text: str
    source_id: str
    source_name: str
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    vector: Optional[np.ndarray] = field(default=None, repr=False)


class FAISSStore:
    """
    IndexFlatIP with L2-normalised vectors = exact cosine similarity.
    Scores are in [-1, 1]; score >= 0.32 indicates meaningful relevance.
    """

    def __init__(self, dim: int):
        import faiss                          # lazy — only needed after install
        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = np.stack([c.vector for c in chunks]).astype(np.float32)
        self._index.add(vecs)
        self._chunks.extend(chunks)

    def remove_source(self, source_id: str) -> None:
        remaining = [c for c in self._chunks if c.source_id != source_id]
        self._rebuild(remaining)

    def clear(self) -> None:
        self._rebuild([])

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        actual_k = min(k, len(self._chunks))
        scores, indices = self._index.search(q, actual_k)
        return [
            (self._chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    @property
    def count(self) -> int:
        return len(self._chunks)

    def _rebuild(self, chunks: list[Chunk]) -> None:
        self._index = self._faiss.IndexFlatIP(self._dim)
        self._chunks = []
        if chunks:
            vecs = np.stack([c.vector for c in chunks]).astype(np.float32)
            self._index.add(vecs)
            self._chunks = chunks

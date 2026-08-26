"""BGE-small-en embeddings — offline, retrieval-optimised."""
from __future__ import annotations
import os
import numpy as np

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-small-en-v1.5")

# BGE instruction prefixes improve retrieval quality
_DOC_PREFIX = "Represent this sentence: "
_QUERY_PREFIX = "Represent this question for searching relevant passages: "


class BGEEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        path = _LOCAL_PATH if os.path.isdir(_LOCAL_PATH) else _MODEL_NAME
        self._model = SentenceTransformer(path)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        print(f"[embedder] Loaded BGE model, dim={self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode document texts → (N, dim) float32 normalised."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [_DOC_PREFIX + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query → (dim,) float32 normalised."""
        prefixed = _QUERY_PREFIX + query
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

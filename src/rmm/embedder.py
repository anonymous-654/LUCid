from __future__ import annotations

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class MemoryEmbedder:
    def __init__(self, model_name: str, batch_size: int = 128):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def top_k_from_precomputed(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        k: int,
    ) -> List[int]:
        if candidate_embeddings.size == 0:
            return []

        scores = candidate_embeddings @ query_embedding
        k = min(k, len(scores))
        order = np.argsort(scores)[::-1][:k]
        return [int(i) for i in order]

    def top_k(self, query: str, candidates, k: int):
        if not candidates:
            return []

        q = self.encode_one(query)
        c = self.encode(list(candidates))
        return self.top_k_from_precomputed(q, c, k)
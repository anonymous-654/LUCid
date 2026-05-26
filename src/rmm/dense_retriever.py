from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class DenseRetrievalMaster:
    def __init__(self, retriever: str, batch_size: int = 128):
        self.retriever = retriever
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_model = None
        self.prepare()

    def prepare(self):
        if self.retriever == "flat-contriever":
            model = AutoModel.from_pretrained("facebook/contriever").to(self.device)
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            model.eval()
            self.retriever_model = (tokenizer, model)
        else:
            raise NotImplementedError(f"Unsupported retriever: {self.retriever}")

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        pooled = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return pooled

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self.retriever != "flat-contriever":
            raise NotImplementedError

        if not texts:
            return torch.empty((0, 768), dtype=torch.float32)

        tokenizer, model = self.retriever_model
        all_vecs = []

        with torch.no_grad():
            dataloader = DataLoader(texts, batch_size=self.batch_size, shuffle=False)

            for batch in dataloader:
                inputs = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                vecs = self.mean_pooling(outputs[0], inputs["attention_mask"])
                vecs = F.normalize(vecs, p=2, dim=1)
                all_vecs.append(vecs.cpu())

        return torch.cat(all_vecs, dim=0)

    def encode_query(self, query: str) -> torch.Tensor:
        return self.encode_texts([query])[0]

    def rank_with_cached_docs(
        self,
        query: str,
        doc_vectors: torch.Tensor,
    ) -> List[int]:
        if doc_vectors is None or len(doc_vectors) == 0:
            return []

        q_vec = self.encode_query(query).unsqueeze(0)
        scores = (q_vec @ doc_vectors.T).squeeze(0)
        return torch.argsort(scores, descending=True).tolist()

    def run_flat_retrieval(self, query: str, corpus):
        doc_vectors = self.encode_texts(list(corpus))
        return self.rank_with_cached_docs(query, doc_vectors)
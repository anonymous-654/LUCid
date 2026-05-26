import numpy as np

from src.reranker.hf_rerankers import BGESequenceReranker, BGEGemmaReranker, Qwen3CausalReranker
from src.reranker.llm_rerankers import LLMReranker


class RerankerMaster:
    def __init__(self, reranker_name: str, device: str = None, promptreason: bool = False):
        self.reranker_name = reranker_name
        self.device = device
        self.promptreason = promptreason
        self.reranker = self._build()

    def _build(self):
        name = self.reranker_name.lower()

        if name == "baai/bge-reranker-v2-m3":
            return BGESequenceReranker(
                model_name=self.reranker_name,
                device=self.device,
                max_length=512,
            )

        if name == "baai/bge-reranker-v2-gemma":
            return BGEGemmaReranker(
                model_name=self.reranker_name,
                device=self.device,
                max_length=1024,
            )
        
        if name in {
            "qwen/qwen3-reranker-0.6b",
            "qwen/qwen3-reranker-4b",
            "qwen/qwen3-reranker-8b",
        }:
            return Qwen3CausalReranker(
                model_name=self.reranker_name,
                device=self.device,
                max_length=8192,   # safe default from model card example
            )

        return LLMReranker(
            model_path=self.reranker_name,
            temperature=0.0,
            max_tokens=1024,
            promptreason=self.promptreason
        )

    def rerank(self, query: str, docs, doc_ids=None, top_k=None):
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(docs))]

        if isinstance(self.reranker, LLMReranker):
            rankings = self.reranker.rerank(
                query=query,
                docs=docs,
                doc_ids=doc_ids,
            )
            if top_k is not None:
                rankings = rankings[:top_k]

            ranked_items = [
                {
                    "corpus_id": doc_ids[idx],
                    "text": docs[idx],
                    "score": float(len(rankings) - pos),
                }
                for pos, idx in enumerate(rankings)
            ]
            return ranked_items, rankings

        scores = self.reranker.score_pairs(query, docs)
        rankings = np.argsort(scores)[::-1].tolist()

        if top_k is not None:
            rankings = rankings[:top_k]

        ranked_items = [
            {
                "corpus_id": doc_ids[idx],
                "text": docs[idx],
                "score": float(scores[idx]),
            }
            for idx in rankings
        ]
        return ranked_items, scores
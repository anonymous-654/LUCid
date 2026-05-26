import json
import re
from typing import List, Dict

from src.llm_client import query_llm
from src.reranker.prompts import REASONING_PROMPT, ZERO_SHOT_PROMPT


SYSTEM_PROMPT = """You are a reranker for lifelong personalization.

You will receive:
- a user query
- a list of candidate memory/session items
- each item has a temporary ID and text

Your task:
Rank the candidate IDs from most important to least important for giving a personalized response to the user query.

Important:
- Return all candidate IDs exactly once.
- Do not invent IDs.
- Return only valid JSON.

Return exactly this format:
{
  "ranked_ids": ["id_1", "id_2", "id_3"]
}
"""


def _extract_json(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError(f"Could not parse JSON from:\n{text}")


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _repair_ranked_ids(predicted_ids: List[str], valid_ids: List[str]) -> List[str]:
    valid_set = set(valid_ids)
    cleaned = [x for x in _dedupe_preserve_order(predicted_ids) if x in valid_set]
    missing = [x for x in valid_ids if x not in cleaned]
    return cleaned + missing


def _make_prompt_ids(n: int) -> List[str]:
    return [f"id_{i+1}" for i in range(n)]


def build_prompt(query: str, prompt_ids: List[str], docs: List[str]) -> str:
    assert len(prompt_ids) == len(docs)

    blocks = []
    for pid, doc in zip(prompt_ids, docs):
        blocks.append(f"ID: {pid}\nTEXT:\n{doc}")

    candidate_block = "\n\n--------------------\n\n".join(blocks)

    return f"""User query:
{query}

Candidate sessions:
{candidate_block}

Rank all candidate IDs from most important to least important for giving a personalized response to the user query.

Return exactly:
{{
  "ranked_ids": ["id_1", "id_2", "id_3"]
}}
"""


class LLMReranker:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        promptreason: bool = False
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.promptreason = promptreason

    def rerank(self, query: str, docs: List[str], doc_ids: List[str]) -> List[int]:
        """
        Returns rankings as indices into the provided docs/doc_ids list.
        doc_ids are NOT shown to the model to avoid leakage.
        """
        if len(docs) != len(doc_ids):
            raise ValueError("docs and doc_ids must have the same length")

        if len(docs) == 0:
            return []

        prompt_ids = _make_prompt_ids(len(docs))
        prompt_id_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(prompt_ids)}

        prompt = build_prompt(query=query, prompt_ids=prompt_ids, docs=docs)

        if self.promptreason:
            sys_prompt = REASONING_PROMPT
        else:
            sys_prompt = ZERO_SHOT_PROMPT

        response, _, _ = query_llm(
            model_path=self.model_path,
            sys_prompt=sys_prompt,
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            print_tokens=False,
        )

        # print(response)

        try:
            parsed = _extract_json(response)
            ranked_prompt_ids = parsed["ranked_ids"]
            if not isinstance(ranked_prompt_ids, list):
                raise ValueError("ranked_ids is not a list")
            ranked_prompt_ids = [str(x) for x in ranked_prompt_ids]
        except Exception:
            ranked_prompt_ids = []

        repaired_prompt_ids = _repair_ranked_ids(ranked_prompt_ids, prompt_ids)
        rankings = [prompt_id_to_idx[pid] for pid in repaired_prompt_ids]
        # print(rankings)
        return rankings

    def score_pairs(self, query: str, docs: List[str], doc_ids: List[str] = None) -> List[float]:
        """
        Compatibility wrapper that converts ranked order into descending scores.
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(docs))]

        rankings = self.rerank(query=query, docs=docs, doc_ids=doc_ids)

        scores = [0.0] * len(docs)
        n = len(rankings)
        for rank_pos, idx in enumerate(rankings):
            scores[idx] = float(n - rank_pos)
        return scores
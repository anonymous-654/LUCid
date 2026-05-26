import json
from typing import List, Tuple, Union


def load_data(path: str):
    """
    Supports .json (list of dicts) or .jsonl (one dict per line).
    """
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_item_flat_index(data, granularity: str, sess_id: str):
    corpus = []

    if granularity == "session":
        text = " ".join([turn["content"] for turn in data if turn["role"] == "user"])
        corpus.append(text)
        ids = [sess_id]

        if "answer" in sess_id and all(
            [not turn.get("has_answer", False) for turn in data if turn["role"] == "user"]
        ):
            ids = [sess_id.replace("answer", "noans")]

    elif granularity == "turn":
        ids = []
        for i_turn, turn in enumerate(data):
            if turn["role"] == "user":
                corpus.append(turn["content"])
                turn_id = f"{sess_id}_{i_turn + 1}"

                if "answer" not in sess_id:
                    ids.append(turn_id)
                else:
                    assert "has_answer" in turn
                    assert turn["has_answer"] in [True, False]
                    if turn["has_answer"]:
                        ids.append(turn_id)
                    else:
                        ids.append(turn_id.replace("answer", "noans"))
                        assert "answer" not in ids[-1]
    else:
        raise NotImplementedError(f"Unsupported granularity: {granularity}")

    return corpus, ids


def build_corpus_from_entry(entry: dict, granularity: str):
    corpus, corpus_ids = [], []

    for cur_sess_id, sess_entry in zip(
        entry["haystack_session_ids"], entry["haystack_sessions"]
    ):
        cur_items, cur_ids = process_item_flat_index(
            sess_entry,
            granularity,
            cur_sess_id,
        )
        corpus += cur_items
        corpus_ids += cur_ids

    return corpus, corpus_ids


def get_correct_docs(corpus_ids: List[str]) -> List[str]:
    return list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))


def get_candidates(
    corpus: List[str],
    corpus_ids: List[str],
    candidate_indices: Union[List[int], None] = None,
    candidate_ids: Union[List[str], None] = None,
):
    """
    Restrict reranking to a candidate set.

    Priority:
      1. candidate_indices
      2. candidate_ids
      3. full corpus
    """
    if candidate_indices is not None:
        cand_docs = [corpus[i] for i in candidate_indices]
        cand_ids = [corpus_ids[i] for i in candidate_indices]
        return cand_docs, cand_ids, candidate_indices

    if candidate_ids is not None:
        id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
        valid_ids = [cid for cid in candidate_ids if cid in id_to_idx]
        candidate_indices = [id_to_idx[cid] for cid in valid_ids]
        cand_docs = [corpus[i] for i in candidate_indices]
        cand_ids = [corpus_ids[i] for i in candidate_indices]
        return cand_docs, cand_ids, candidate_indices

    return corpus, corpus_ids, list(range(len(corpus)))


def maybe_extract_candidates_from_entry(entry: dict):
    """
    Optional helper if you later attach first-stage retrieval outputs to each entry.

    Supported patterns:
      entry["candidate_indices"]
      entry["candidate_ids"]

    Otherwise returns (None, None).
    """
    candidate_indices = entry.get("candidate_indices", None)
    candidate_ids = entry.get("candidate_ids", None)
    return candidate_indices, candidate_ids
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from src.retrieval.eval_utils import evaluate_retrieval
from src.reranker.hf_rerankers import (
    BGESequenceReranker,
    BGEGemmaReranker,
    Qwen3CausalReranker,
)


def build_gold_turn_index(entry) -> Set[Tuple[str, int]]:
    gold = set()
    for sess_id, session_turns in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
    ):
        for tid, turn in enumerate(session_turns):
            if turn.get("role") == "user" and turn.get("has_answer", False):
                gold.add((sess_id, tid))
    return gold


def build_gold_session_ids(entry) -> List[str]:
    gold_sessions = []
    for sess_id, session_turns in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
    ):
        has_gold = any(
            turn.get("role") == "user" and turn.get("has_answer", False)
            for turn in session_turns
        )
        if has_gold:
            gold_sessions.append(sess_id)
    return gold_sessions


def memory_is_correct_by_turn(memory: Dict[str, Any], gold_turns: Set[Tuple[str, int]]) -> bool:
    for ref in memory.get("references", []):
        sess_id = ref["session_id"]
        for tid in ref.get("turn_ids", []):
            if (sess_id, tid) in gold_turns:
                return True
    return False


def collapse_ranked_memories_to_sessions(ranked_items: List[Dict[str, Any]]) -> List[str]:
    ranked_sessions = []
    seen = set()

    for item in ranked_items:
        for ref in item.get("references", []):
            sess_id = ref["session_id"]
            if sess_id not in seen:
                seen.add(sess_id)
                ranked_sessions.append(sess_id)

    return ranked_sessions


def evaluate_session_retrieval(ranked_sessions: List[str], gold_sessions: List[str], k: int):
    top_k = ranked_sessions[:k]
    gold_set = set(gold_sessions)

    if not gold_set:
        return None, None, None, None

    hits = [1 if sid in gold_set else 0 for sid in top_k]
    num_hits = sum(hits)

    recall_any = 1.0 if num_hits > 0 else 0.0
    recall_all = num_hits / len(gold_set)
    recall_ge_n = float(num_hits)

    dcg = 0.0
    for rank, rel in enumerate(hits, start=1):
        if rel:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(gold_set), k)
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / np.log2(rank + 1)

    ndcg_any = dcg / idcg if idcg > 0 else 0.0
    return recall_any, recall_all, ndcg_any, recall_ge_n


def entry_has_target(eval_entry) -> bool:
    return any(
        turn.get("role") == "user" and turn.get("has_answer", False)
        for session in eval_entry["haystack_sessions"]
        for turn in session
    )


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_reranker(name: str, model_name: str, device: str):
    if name == "bge-seq":
        return BGESequenceReranker(model_name=model_name, device=device)
    if name == "bge-gemma":
        return BGEGemmaReranker(model_name=model_name, device=device)
    if name == "qwen3":
        return Qwen3CausalReranker(model_name=model_name, device=device)
    raise ValueError(f"Unknown reranker type: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--reranker_type", type=str, required=True,
                        choices=["bge-seq", "bge-gemma", "qwen3"])
    parser.add_argument("--reranker_model", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rerank_top_n", type=int, default=30)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 30])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    data = load_jsonl(Path(args.in_file))
    reranker = make_reranker(args.reranker_type, args.reranker_model, args.device)

    # data = data[:4]

    results = []

    for entry in tqdm(data, desc="Reranking"):
        query = entry["retrieval_results"]["query"]
        ranked_items = entry["retrieval_results"]["ranked_items"]

        head = ranked_items[:args.rerank_top_n]
        tail = ranked_items[args.rerank_top_n:]

        docs = [x["text"] for x in head]
        scores = reranker.score_pairs(query, docs, batch_size=args.batch_size)

        scored = list(zip(head, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked_head = []
        for item, score in scored:
            item = dict(item)
            item["reranker_score"] = score
            reranked_head.append(item)

        reranked_items = reranked_head + tail
        ranked_sessions = collapse_ranked_memories_to_sessions(reranked_items)

        gold_turns = build_gold_turn_index(entry)
        gold_sessions = build_gold_session_ids(entry)

        corpus_ids = [item["corpus_id"] for item in reranked_items]
        correct_docs = [
            item["corpus_id"]
            for item in reranked_items
            if memory_is_correct_by_turn(item, gold_turns)
        ]

        rankings = list(range(len(corpus_ids)))

        entry["retrieval_results"]["ranked_items"] = reranked_items
        entry["retrieval_results"]["ranked_sessions"] = ranked_sessions
        entry["retrieval_results"]["metrics"] = {"memory": {}, "session": {}}
        entry["retrieval_results"]["reranker"] = {
            "type": args.reranker_type,
            "model": args.reranker_model,
            "rerank_top_n": args.rerank_top_n,
        }

        for k in args.ks:
            recall_any, recall_all, ndcg_any, recall_ge_n = evaluate_retrieval(
            rankings=rankings,
            correct_docs=correct_docs,
            corpus_ids=corpus_ids,
            k=k,
            )

            entry["retrieval_results"]["metrics"]["memory"].update({
                f"recall_any@{k}": recall_any,
                f"recall_all@{k}": recall_all,
                f"recall_ge_n@{k}": recall_ge_n,
                f"ndcg_any@{k}": ndcg_any,
            })

            s_recall_any, s_recall_all, s_ndcg_any, s_recall_ge_n = evaluate_session_retrieval(
                ranked_sessions=ranked_sessions,
                gold_sessions=gold_sessions,
                k=k,
            )
            entry["retrieval_results"]["metrics"]["session"].update({
                f"recall_any@{k}": s_recall_any,
                f"recall_all@{k}": s_recall_all,
                f"recall_ge_n@{k}": s_recall_ge_n,
                f"ndcg_any@{k}": s_ndcg_any,
            })

        results.append(entry)

    averaged_results = {"memory": {}, "session": {}}
    if results:
        for level in ["memory", "session"]:
            for metric_name in results[0]["retrieval_results"]["metrics"][level]:
                vals = []
                for ex in results:
                    if not entry_has_target(ex):
                        continue
                    val = ex["retrieval_results"]["metrics"][level][metric_name]
                    if val is not None:
                        vals.append(val)
                averaged_results[level][metric_name] = float(np.mean(vals)) if vals else None

    print(json.dumps(averaged_results, indent=2))
    save_jsonl(Path(args.out_file), results)


if __name__ == "__main__":
    main()
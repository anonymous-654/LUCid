from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from src.retrieval.eval_utils import evaluate_retrieval

from src.rmm.dense_retriever import DenseRetrievalMaster
from src.rmm.embedder import MemoryEmbedder
from src.rmm.llm import LLMClient
from src.rmm.memory import (
    build_dialogue_turns,
    reflect_over_sessions,
    retrieve_over_memory_bank,
)
from src.rmm.utils import get_outfile_prefix, make_session_content_hash, make_session_key


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outfile_prefix", type=str, default=None)

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["true", "false", "none"],
        help="Pass Qwen/OpenAI-compatible thinking flag when supported by the server.",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="flat-contriever",
        choices=["oracle", "flat-contriever"],
    )
    parser.add_argument(
        "--memory_top_k",
        type=int,
        default=5,
        help="Top-K existing memories to consider for merge/update",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 30, 50])

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--precomputed_extractions_file",
        type=str,
        required=True,
        help="Required JSON file containing session-level precomputed extraction results keyed by sess_id",
    )

    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def _parse_enable_thinking(arg: str):
    val = str(arg).strip().lower()
    if val == "true":
        return True
    if val == "false":
        return False
    return None


def build_gold_turn_index(entry) -> Set[Tuple[str, int]]:
    """
    Build gold labels in the same grouped dialogue-turn index space used by memory.py.
    """
    gold = set()

    for sess_id, session_turns in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
    ):
        dialogue_turns = build_dialogue_turns(session_turns)

        for turn in dialogue_turns:
            has_gold = any(
                item.get("role") == "user" and item.get("has_answer", False)
                for item in turn.get("raw_items", [])
            )
            if has_gold:
                gold.add((sess_id, turn["turn_id"]))

    return gold


def build_gold_session_ids(entry) -> List[str]:
    gold_sessions = []

    for sess_id, session_turns in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
    ):
        dialogue_turns = build_dialogue_turns(session_turns)

        has_gold = any(
            any(
                item.get("role") == "user" and item.get("has_answer", False)
                for item in turn.get("raw_items", [])
            )
            for turn in dialogue_turns
        )

        if has_gold:
            gold_sessions.append(sess_id)

    return gold_sessions


def memory_is_correct_by_turn(
    memory: Dict[str, Any],
    gold_turns: Set[Tuple[str, int]],
) -> bool:
    for ref in memory.get("references", []):
        sess_id = ref["session_id"]
        for tid in ref.get("turn_ids", []):
            if (sess_id, tid) in gold_turns:
                return True
    return False


def collapse_ranked_memories_to_sessions(
    ranked_memory_indices: List[int],
    memory_bank: List[Dict[str, Any]],
) -> List[str]:
    ranked_sessions = []
    seen = set()

    for mem_idx in ranked_memory_indices:
        memory = memory_bank[mem_idx]
        for ref in memory.get("references", []):
            sess_id = ref["session_id"]
            if sess_id not in seen:
                seen.add(sess_id)
                ranked_sessions.append(sess_id)

    return ranked_sessions


def evaluate_session_retrieval(
    ranked_sessions: List[str],
    gold_sessions: List[str],
    k: int,
):
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
        msg.get("role") == "user" and msg.get("has_answer", False)
        for session in eval_entry["haystack_sessions"]
        for msg in session
    )


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.in_file, "r") as f:
        in_data = json.load(f)

    if args.limit is not None:
        in_data = in_data[:args.limit]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    total_before_shard = len(in_data)
    in_data = in_data[args.shard_id::args.num_shards]

    print(
        f"Shard {args.shard_id + 1}/{args.num_shards}: "
        f"processing {len(in_data)} of {total_before_shard} examples"
    )

    if not os.path.isfile(args.precomputed_extractions_file):
        raise FileNotFoundError(
            f"Missing precomputed extraction file: {args.precomputed_extractions_file}"
        )

    with open(args.precomputed_extractions_file, "r") as f:
        precomputed_extractions = json.load(f)

    print(
        f"Loaded precomputed extractions for {len(precomputed_extractions)} sessions "
        f"from {args.precomputed_extractions_file}"
    )

    missing_session_keys = []

    for entry in in_data:
        for sess_id, session_turns in zip(
            entry["haystack_session_ids"],
            entry["haystack_sessions"],
        ):
            session_key = make_session_key(sess_id, session_turns)
            if session_key not in precomputed_extractions:
                missing_session_keys.append((sess_id, session_key))

    if missing_session_keys:
        preview = missing_session_keys[:5]
        raise KeyError(
            f"Missing {len(missing_session_keys)} session keys in "
            f"{args.precomputed_extractions_file}. Examples: {preview}"
        )

    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    embedder = MemoryEmbedder(args.embedding_model)

    retriever_master = None
    if args.retriever == "flat-contriever":
        retriever_master = DenseRetrievalMaster(args.retriever)

    outfile_prefix = get_outfile_prefix(args)
    out_file = os.path.join(
        args.out_dir,
        f"{outfile_prefix}_rmm_retrievallog_{args.retriever}_{total_before_shard}"
        f"_shard{args.shard_id:03d}of{args.num_shards:03d}.jsonl",
    )
    # out_file = os.path.join(
    #     args.out_dir,
    #     f"{outfile_prefix}_rmm_retrievallog_{args.retriever}_{len(in_data)}",
    # )
    print(out_file)

    results = []

    for entry in tqdm(in_data, desc="Processing examples", total=len(in_data)):
        memory_bank, memory_bank_embeddings, extraction_log = reflect_over_sessions(
            entry=entry,
            llm=llm,
            embedder=embedder, 
            precomputed_extractions=precomputed_extractions,
            memory_top_k=args.memory_top_k,
        )

        query = entry["query"]

        cached_doc_vectors = None
        if args.retriever == "flat-contriever" and memory_bank:
            cached_doc_vectors = retriever_master.encode_texts(
                [m["summary"] for m in memory_bank]
            )

        rankings = retrieve_over_memory_bank(
            query=query,
            memory_bank=memory_bank,
            retriever_master=retriever_master,
            retriever=args.retriever,
            cached_doc_vectors=cached_doc_vectors,
        )

        gold_turns = build_gold_turn_index(entry)
        gold_sessions = build_gold_session_ids(entry)

        corpus_ids = [m["memory_id"] for m in memory_bank]
        correct_docs = [
            m["memory_id"]
            for m in memory_bank
            if memory_is_correct_by_turn(m, gold_turns)
        ]

        ranked_sessions = collapse_ranked_memories_to_sessions(rankings, memory_bank)

        cur_results = dict(entry)
        cur_results["prospective_reflection"] = {
            "memory_bank": memory_bank,
            "extraction_log": extraction_log,
        }
        cur_results["retrieval_results"] = {
            "query": query,
            "ranked_items": [
                {
                    "corpus_id": memory_bank[rid]["memory_id"],
                    "text": memory_bank[rid]["summary"],
                    "source_session_ids": memory_bank[rid]["source_session_ids"],
                    "references": memory_bank[rid]["references"],
                }
                for rid in rankings
            ],
            "ranked_sessions": ranked_sessions,
            "gold_sessions": gold_sessions,
            "metrics": {
                "memory": {},
                "session": {},
            },
        }

        for k in args.ks:
            recall_any, recall_all, ndcg_any, recall_ge_n = evaluate_retrieval(
                rankings, correct_docs, corpus_ids, k=k
            )
            cur_results["retrieval_results"]["metrics"]["memory"].update({
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
            cur_results["retrieval_results"]["metrics"]["session"].update({
                f"recall_any@{k}": s_recall_any,
                f"recall_all@{k}": s_recall_all,
                f"recall_ge_n@{k}": s_recall_ge_n,
                f"ndcg_any@{k}": s_ndcg_any,
            })

        results.append(cur_results)

    averaged_results = {
        "memory": {},
        "session": {},
    }
    ignored_qs_no_target = set()

    if results:
        for level in ["memory", "session"]:
            metric_dict = results[0]["retrieval_results"]["metrics"].get(level, {})
            for metric_name in metric_dict:
                metric_vals = []
                for eval_entry in results:
                    if not entry_has_target(eval_entry):
                        ignored_qs_no_target.add(eval_entry.get("query", "unknown"))
                        continue

                    val = eval_entry["retrieval_results"]["metrics"][level][metric_name]
                    if val is not None:
                        metric_vals.append(val)

                averaged_results[level][metric_name] = (
                    float(np.mean(metric_vals)) if metric_vals else None
                )

    print(
        f"Additionally ignored {len(ignored_qs_no_target)} instances due to no target turns "
        f"from the user side: {ignored_qs_no_target}"
    )
    print(json.dumps(averaged_results, indent=2))

    with open(out_file, "w") as out_f:
        for entry in results:
            print(json.dumps(entry), file=out_f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
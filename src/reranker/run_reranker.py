import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from src.reranker.utils import (
    load_data,
    save_jsonl,
    build_corpus_from_entry,
    get_correct_docs,
    get_candidates,
    maybe_extract_candidates_from_entry,
)
from src.reranker.eval_utils import (
    evaluate_retrieval,
    evaluate_retrieval_turn2session,
)
from src.reranker.rerank_master import RerankerMaster


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--outfile_prefix", type=str, default=None)
    parser.add_argument("--promptreason", action="store_true")

    parser.add_argument(
        "--granularity",
        type=str,
        required=True,
        choices=["session", "turn"],
    )

    parser.add_argument(
        "--reranker",
        type=str,
        required=True,
        help=(
            "Examples: "
            "BAAI/bge-reranker-v2-m3, "
            "BAAI/bge-reranker-v2-gemma, "
            "Qwen/Qwen2.5-7B-Instruct, "
            "qwen/qwen3-reranker-0.6b",
            "qwen/qwen3-reranker-4b",
            "qwen/qwen3-reranker-8b",
            "gpt-4.1-mini, "
            "gemini-1.5-pro"
        ),
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Trim final reranked output to top_k.",
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 30, 50],
    )

    parser.add_argument(
        "--min_relevant",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--clamp_threshold",
        action="store_true",
        default=False,
        help="Clamp Recall>=N threshold to min(N, number of relevant docs).",
    )

    parser.add_argument(
        "--turn2session_eval",
        action="store_true",
        default=False,
        help="For turn granularity, convert to session-level during evaluation.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    return parser.parse_args()


def get_outfile_prefix(args):
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != "none":
        return args.outfile_prefix
    return os.path.splitext(os.path.basename(args.in_file))[0]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    in_data = load_data(args.in_file)
    reranker = RerankerMaster(
        reranker_name=args.reranker,
        device=args.device,
        promptreason=args.promptreason
    )

    results = []

    in_data = in_data[:3]

    for entry in tqdm(in_data, desc="Reranking", total=len(in_data)):
        query = entry["query"]
        corpus, corpus_ids = build_corpus_from_entry(entry, args.granularity)
        correct_docs = get_correct_docs(corpus_ids)

        candidate_indices, candidate_ids = maybe_extract_candidates_from_entry(entry)
        cand_docs, cand_ids, cand_orig_indices = get_candidates(
            corpus=corpus,
            corpus_ids=corpus_ids,
            candidate_indices=candidate_indices,
            candidate_ids=candidate_ids,
        )

        ranked_items, scores = reranker.rerank(
            query=query,
            docs=cand_docs,
            doc_ids=cand_ids,
            top_k=args.top_k,
        )

        # Convert reranked candidate-local ordering back to full-corpus indices
        cand_id_to_global_idx = {cid: corpus_ids.index(cid) for cid in cand_ids}
        rankings_global = [cand_id_to_global_idx[item["corpus_id"]] for item in ranked_items]

        metric_fn = evaluate_retrieval
        if args.granularity == "turn" and args.turn2session_eval:
            metric_fn = evaluate_retrieval_turn2session

        metrics = {}
        for k in args.ks:
            if args.top_k is not None and k > args.top_k:
                continue

            if metric_fn is evaluate_retrieval_turn2session:
                recall_any, recall_all, ndcg_score, recall_ge_n = metric_fn(
                    rankings_global,
                    correct_docs,
                    corpus_ids,
                    k=k,
                )
            else:
                recall_any, recall_all, ndcg_score, recall_ge_n = metric_fn(
                    rankings_global,
                    correct_docs,
                    corpus_ids,
                    k=k,
                    min_relevant=args.min_relevant,
                    clamp_threshold=args.clamp_threshold,
                )

            metrics[f"recall_any@{k}"] = recall_any
            metrics[f"recall_all@{k}"] = recall_all
            metrics[f"ndcg@{k}"] = ndcg_score
            metrics[f"recall_ge_n@{k}"] = recall_ge_n

        out_entry = dict(entry)
        out_entry["reranking_results"] = {
            "query": query,
            "granularity": args.granularity,
            "reranker": args.reranker,
            "num_corpus_items": len(corpus),
            "num_candidate_items": len(cand_docs),
            "candidate_doc_ids": cand_ids,
            "ranked_items": ranked_items,
            "metrics": metrics,
        }
        results.append(out_entry)

    avg_metrics = {}
    if results:
        metric_names = list(results[0]["reranking_results"]["metrics"].keys())
        for metric_name in metric_names:
            vals = [
                row["reranking_results"]["metrics"][metric_name]
                for row in results
            ]
            avg_metrics[metric_name] = float(np.mean(vals))

    print(json.dumps({"average_metrics": avg_metrics}, indent=2))

    safe_reranker_name = args.reranker.replace("/", "_")
    outfile_prefix = get_outfile_prefix(args)
    out_file = os.path.join(
        args.out_dir,
        f"{outfile_prefix}_rerank_{args.granularity}_{safe_reranker_name}_ispromptreason{args.promptreason}.jsonl",
    )

    save_jsonl(results, out_file)
    print(f"Saved reranking results to: {out_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
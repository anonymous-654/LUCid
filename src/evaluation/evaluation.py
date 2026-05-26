import random
from typing import List, Dict, Any
import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from src.evaluation.runner import EVALUATOR_MODEL, evaluate_entry
from src.evaluation.utils import safe_load_json_or_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def stratified_sample_by_dimension(
    data: List[Dict[str, Any]],
    n_per_group: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Samples up to n_per_group entries per query_dimension.
    """
    random.seed(seed)
    groups = defaultdict(list)

    for row in data:
        dim = str(row.get("query_dimension", "")).strip().lower()
        groups[dim].append(row)

    sampled = []
    for dim, rows in groups.items():
        k = min(n_per_group, len(rows))
        sampled_rows = random.sample(rows, k)
        sampled.extend(sampled_rows)

        print(f"[DEBUG] Sampling {k} rows for dimension='{dim}' (total={len(rows)})")

    return sampled


def print_summary(rows):
    overall = []
    by_dim = defaultdict(list)

    for row in rows:
        match = row.get("evaluator_match", None)
        dim = row.get("query_dimension", row.get("evaluator_dimension", "unknown"))
        if match is None:
            continue
        overall.append(match)
        by_dim[dim].append(match)

    print("\n===== LLM-as-Judge Summary =====", flush=True)

    if overall:
        print(f"Overall match rate: {sum(overall) / len(overall):.4f} ({len(overall)} examples)", flush=True)
    else:
        print("Overall match rate: no valid entries", flush=True)

    for dim, vals in sorted(by_dim.items()):
        print(f"{dim}: {sum(vals) / len(vals):.4f} ({len(vals)})", flush=True)


def main(args):
    in_data = safe_load_json_or_jsonl(args.in_file)

    # # ===== DEBUG MODE =====
    # DEBUG = True
    # if DEBUG:
    #     in_data = stratified_sample_by_dimension(in_data, n_per_group=3)
    #     print(f"[DEBUG] Total sampled rows: {len(in_data)}", flush=True)
    # # =====================

    os.makedirs(args.out_dir, exist_ok=True)

    input_base = os.path.basename(args.in_file)
    if input_base.endswith(".jsonl"):
        input_base = input_base[:-6]
    elif input_base.endswith(".json"):
        input_base = input_base[:-5]

    out_file = os.path.join(args.out_dir, f"{input_base}_judge_eval.jsonl")
    print(out_file, flush=True)
    print(f"Evaluator model: {EVALUATOR_MODEL}", flush=True)

    evaluated_rows = []

    with open(out_file, "w") as out_f:
        for entry in tqdm(in_data, desc="running llm-as-judge evaluation", total=len(in_data)):
            try:
                evaluated = evaluate_entry(entry)
                evaluated_rows.append(evaluated)
                print(json.dumps(evaluated, ensure_ascii=False), file=out_f, flush=True)
            except Exception as e:
                print("One exception captured", repr(e), flush=True)
                continue

    print_summary(evaluated_rows)


if __name__ == "__main__":
    args = parse_args()
    main(args)
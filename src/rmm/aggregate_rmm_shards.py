# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import glob
# import json
# import os
# from typing import Any, Dict, List, Optional

# import numpy as np


# def entry_has_target(eval_entry: Dict[str, Any]) -> bool:
#     return any(
#         msg.get("role") == "user" and msg.get("has_answer", False)
#         for session in eval_entry["haystack_sessions"]
#         for msg in session
#     )


# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     rows = []
#     with open(path, "r") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rows.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"Bad JSON in {path}:{line_no}: {e}") from e
#     return rows


# def metric_sort_key(name: str):
#     # Sort like recall_any@1, recall_any@3, ..., ndcg_any@1, ...
#     if "@" in name:
#         prefix, k = name.rsplit("@", 1)
#         try:
#             return (prefix, int(k))
#         except ValueError:
#             return (prefix, k)
#     return (name, -1)


# def recompute_averages(results: List[Dict[str, Any]]) -> Dict[str, Any]:
#     averaged_results = {
#         "memory": {},
#         "session": {},
#     }
#     ignored_qs_no_target = set()

#     if not results:
#         return {
#             "num_examples": 0,
#             "num_scored_examples": 0,
#             "num_ignored_no_target": 0,
#             "ignored_queries_no_target": [],
#             "averaged_results": averaged_results,
#         }

#     for level in ["memory", "session"]:
#         # collect union of all metric names just in case
#         metric_names = set()
#         for entry in results:
#             metric_names.update(
#                 entry.get("retrieval_results", {})
#                 .get("metrics", {})
#                 .get(level, {})
#                 .keys()
#             )

#         for metric_name in sorted(metric_names, key=metric_sort_key):
#             metric_vals = []
#             for eval_entry in results:
#                 if not entry_has_target(eval_entry):
#                     ignored_qs_no_target.add(eval_entry.get("query", "unknown"))
#                     continue

#                 val = (
#                     eval_entry.get("retrieval_results", {})
#                     .get("metrics", {})
#                     .get(level, {})
#                     .get(metric_name)
#                 )
#                 if val is not None:
#                     metric_vals.append(val)

#             averaged_results[level][metric_name] = (
#                 float(np.mean(metric_vals)) if metric_vals else None
#             )

#     num_scored_examples = sum(1 for x in results if entry_has_target(x))

#     return {
#         "num_examples": len(results),
#         "num_scored_examples": num_scored_examples,
#         "num_ignored_no_target": len(ignored_qs_no_target),
#         "ignored_queries_no_target": sorted(ignored_qs_no_target),
#         "averaged_results": averaged_results,
#     }


# def maybe_dedup_results(
#     results: List[Dict[str, Any]],
#     dedup_key: Optional[str],
# ) -> List[Dict[str, Any]]:
#     if dedup_key is None:
#         return results

#     seen = set()
#     deduped = []

#     for row in results:
#         if dedup_key == "query":
#             key = row.get("query")
#         elif dedup_key == "id":
#             key = row.get("id")
#         elif dedup_key == "query+answer":
#             key = (
#                 row.get("query"),
#                 json.dumps(row.get("answer"), sort_keys=True, ensure_ascii=False),
#             )
#         else:
#             raise ValueError(f"Unsupported dedup_key: {dedup_key}")

#         if key in seen:
#             continue
#         seen.add(key)
#         deduped.append(row)

#     return deduped


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input_glob",
#         type=str,
#         required=True,
#         help="Glob for shard JSONL files, e.g. 'src/rmm/logs/shards/*_shard*of*.jsonl'",
#     )
#     parser.add_argument(
#         "--merged_jsonl",
#         type=str,
#         required=True,
#         help="Path to merged JSONL output",
#     )
#     parser.add_argument(
#         "--summary_json",
#         type=str,
#         default=None,
#         help="Optional path to write aggregate summary JSON",
#     )
#     parser.add_argument(
#         "--dedup_key",
#         type=str,
#         default=None,
#         choices=[None, "query", "id", "query+answer"],
#         help="Optional dedup if you accidentally merged overlapping shards",
#     )
#     args = parser.parse_args()

#     shard_files = sorted(glob.glob(args.input_glob))
#     if not shard_files:
#         raise FileNotFoundError(f"No files matched: {args.input_glob}")

#     print("Found shard files:")
#     for p in shard_files:
#         print(f"  {p}")

#     merged_results: List[Dict[str, Any]] = []
#     counts_by_file = {}

#     for path in shard_files:
#         rows = load_jsonl(path)
#         counts_by_file[path] = len(rows)
#         merged_results.extend(rows)

#     print("\nRows per shard:")
#     for path, n in counts_by_file.items():
#         print(f"  {path}: {n}")

#     before_dedup = len(merged_results)
#     merged_results = maybe_dedup_results(merged_results, args.dedup_key)
#     after_dedup = len(merged_results)

#     os.makedirs(os.path.dirname(args.merged_jsonl) or ".", exist_ok=True)
#     with open(args.merged_jsonl, "w") as f:
#         for row in merged_results:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     summary = recompute_averages(merged_results)

#     print("\nMerged output written to:")
#     print(f"  {args.merged_jsonl}")

#     print("\nMerge stats:")
#     print(json.dumps(
#         {
#             "num_input_files": len(shard_files),
#             "num_rows_before_dedup": before_dedup,
#             "num_rows_after_dedup": after_dedup,
#             "dedup_key": args.dedup_key,
#         },
#         indent=2,
#     ))

#     print("\nFull metrics:")
#     print(json.dumps(summary, indent=2))

#     if args.summary_json:
#         os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
#         with open(args.summary_json, "w") as f:
#             json.dump(summary, f, indent=2)
#         print("\nSummary JSON written to:")
#         print(f"  {args.summary_json}")


# if __name__ == "__main__":
#     main()

import json

HARD_DIMENSIONS = {
    "age_group",
    "location/country",
    "religion",
}

K_VALUES = [1, 3, 5, 10, 20, 30, 50]
METRIC_PREFIXES = ["recall_any", "recall_all", "recall_ge_n", "ndcg_any"]


def init_metric_store():
    store = {}
    for prefix in METRIC_PREFIXES:
        for k in K_VALUES:
            metric_name = f"{prefix}@{k}"
            store[metric_name] = {"sum": 0.0, "count": 0}
    return store


def init_group():
    return {
        "num_examples": 0,
        "memory": init_metric_store(),
        "session": init_metric_store(),
    }


def update_metric_store(store, metrics_dict):
    for metric_name, stats in store.items():
        val = metrics_dict.get(metric_name)
        if isinstance(val, (int, float)):
            stats["sum"] += float(val)
            stats["count"] += 1


def finalize_metric_store(store):
    out = {}
    for metric_name, stats in store.items():
        if stats["count"] > 0:
            out[metric_name] = stats["sum"] / stats["count"]
    return out


def aggregate_single_file_memory_session_fast(filepath, outpath=None):
    groups = {
        "all": init_group(),
        "hard": init_group(),
    }

    seen = set()
    num_duplicates_removed = 0
    num_lines = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            num_lines += 1
            e = json.loads(line)

            # Drop big fields immediately after parsing to reduce memory pressure.
            e.pop("haystack_sessions", None)
            e.pop("haystack_session_ids", None)
            e.pop("haystack_sessions_ids", None)

            key = (
                e.get("query"),
                e.get("expected_category"),
                tuple(sorted(e.get("answer_session_ids", []))),
            )

            if key in seen:
                num_duplicates_removed += 1
                continue
            seen.add(key)

            target_groups = ["all"]
            if e.get("query_dimension") in HARD_DIMENSIONS:
                target_groups.append("hard")

            metrics = e.get("retrieval_results", {}).get("metrics", {})

            for group_name in target_groups:
                groups[group_name]["num_examples"] += 1
                update_metric_store(groups[group_name]["memory"], metrics.get("memory", {}))
                update_metric_store(groups[group_name]["session"], metrics.get("session", {}))

            if num_lines % 100000 == 0:
                print(f"Processed {num_lines:,} lines...")

    output = {
        "num_lines_read": num_lines,
        "num_duplicates_removed": num_duplicates_removed,
        "all": {
            "num_examples": groups["all"]["num_examples"],
            "memory": finalize_metric_store(groups["all"]["memory"]),
            "session": finalize_metric_store(groups["all"]["session"]),
        },
        "hard": {
            "num_examples": groups["hard"]["num_examples"],
            "memory": finalize_metric_store(groups["hard"]["memory"]),
            "session": finalize_metric_store(groups["hard"]["session"]),
        },
    }

    if outpath is not None:
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return output


if __name__ == "__main__":
    res = aggregate_single_file_memory_session_fast(
        "src/rmm/logs/all_shards_merged_rerank.jsonl",
        outpath="src/rmm/logs/aggregated_result_with_hard_reranker.json",
    )
# aggregate_retrieval_logs.py

import os
import json
import argparse
from collections import defaultdict
import pandas as pd

# MCQ_DIMENSIONS = {"age_group", "domain", "location/country", "style_pref"}
# SUITABILITY_DIMENSIONS = {"religion", "medical_health_condition"}

HARD_DIMENSIONS = {
    "age_group",
    "location/country",
    "religion"
}

K_VALUES = [1, 3, 5, 10, 20, 30, 50]
METRIC_PREFIXES = ["recall_any", "recall_all", "recall_ge_n", "ndcg_any"]


def deduplicate_entries(entries):
    # Convert to DataFrame
    df = pd.DataFrame(entries)

    # Normalize answer_session_ids (so lists compare correctly)
    def normalize_ids(x):
        if isinstance(x, list):
            return tuple(sorted(x))
        return tuple()

    df["answer_session_ids_norm"] = df["answer_session_ids"].apply(normalize_ids)

    # Drop duplicates
    df_dedup = df.drop_duplicates(
        subset=["query", "expected_category", "answer_session_ids_norm"]
    )

    # Drop helper column
    df_dedup = df_dedup.drop(columns=["answer_session_ids_norm"])

    # Convert back to list of dicts

    # print(f"Before - {len(df)}")
    # print(f"after - {len()}")
    return df_dedup.to_dict(orient="records")


def safe_mean(values):
    return sum(values) / len(values) if values else None


def is_hard_instance(entry):
    qdim = entry.get("query_dimension")
    return qdim in HARD_DIMENSIONS


def read_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return rows


def collect_metric_names(entries):
    metric_names = set()
    for entry in entries:
        retrieval_results = entry.get("retrieval_results", {})
        metrics = retrieval_results.get("metrics", {})
        for granularity in ["session", "turn"]:
            gran_metrics = metrics.get(granularity, {})
            for metric_name in gran_metrics:
                metric_names.add((granularity, metric_name))
    return metric_names


def default_metric_store():
    return defaultdict(list)


def aggregate_entries(entries):
    grouped = {
        "all": {
            "count": 0,
            "metrics": {
                "session": default_metric_store(),
                "turn": default_metric_store(),
            },
        },
        "hard": {
            "count": 0,
            "metrics": {
                "session": default_metric_store(),
                "turn": default_metric_store(),
            },
        },
    }

    for entry in entries:
        retrieval_results = entry.get("retrieval_results", {})
        metrics = retrieval_results.get("metrics", {})

        target_groups = ["all"]
        if is_hard_instance(entry):
            target_groups.append("hard")

        for group_name in target_groups:
            grouped[group_name]["count"] += 1

            for granularity in ["session", "turn"]:
                gran_metrics = metrics.get(granularity, {})
                for metric_name, value in gran_metrics.items():
                    if isinstance(value, (int, float)):
                        grouped[group_name]["metrics"][granularity][metric_name].append(value)

    # average metrics
    output = {}
    for group_name, group_data in grouped.items():
        output[group_name] = {
            "count": group_data["count"],
            "session": {},
            "turn": {},
        }

        for granularity in ["session", "turn"]:
            metric_store = group_data["metrics"][granularity]
            for metric_name, values in sorted(metric_store.items()):
                output[group_name][granularity][metric_name] = safe_mean(values)

    return output


def filter_to_requested_metrics(agg_result):
    """
    Keep only metrics for k in [1,3,5,10,20,30,50]
    and prefixes in [recall_any, recall_all, recall_ge_n, ndcg_any].
    """
    filtered = {}

    for group_name, group_data in agg_result.items():
        filtered[group_name] = {
            "count": group_data["count"],
            "session": {},
            "turn": {},
        }

        for granularity in ["session", "turn"]:
            for prefix in METRIC_PREFIXES:
                for k in K_VALUES:
                    metric_name = f"{prefix}@{k}"
                    if metric_name in group_data.get(granularity, {}):
                        filtered[group_name][granularity][metric_name] = group_data[granularity][metric_name]

    return filtered


def aggregate_file(filepath):
    entries = read_jsonl(filepath)
    entries = deduplicate_entries(entries)
    agg = aggregate_entries(entries)
    agg = filter_to_requested_metrics(agg)
    return agg


def find_log_files(root_dir):
    """
    Walk through every subfolder under src/retrieval/retrieval_logs
    and collect files. By default, this script assumes each file is a JSONL retrieval log.
    """
    log_files = []
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            fpath = os.path.join(cur_root, fname)
            # skip obvious non-log files
            if fname.startswith("."):
                continue
            log_files.append(fpath)
    return sorted(log_files)


def build_output_structure(root_dir):
    """
    Output format:
    {
      "file1": {
        "all": {...},
        "hard": {...}
      },
      "file2": {
        "all": {...},
        "hard": {...}
      }
    }
    """
    results = {}
    log_files = find_log_files(root_dir)

    for filepath in log_files:
        rel_path = os.path.relpath(filepath, root_dir)
        try:
            file_agg = aggregate_file(filepath)
            results[rel_path] = file_agg
        except Exception as e:
            results[rel_path] = {
                "error": str(e)
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_root",
        type=str,
        default="src/retrieval/retrieval_logs",
        help="Root directory containing retrieval log folders",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="src/retrieval/retrieval_logs/aggregated_results.json",
        help="Path to save aggregated JSON",
    )
    args = parser.parse_args()

    output = build_output_structure(args.logs_root)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated results to: {args.out_file}")


if __name__ == "__main__":
    main()
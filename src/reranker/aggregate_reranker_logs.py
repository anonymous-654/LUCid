import os
import json

HARD_DIMENSIONS = {
    "age_group",
    "location/country",
    "religion",
}

K_VALUES = [1, 3, 5, 10, 20, 30, 50]
METRIC_PREFIXES = ["recall_any", "recall_all", "recall_ge_n", "ndcg"]


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
        "metrics": init_metric_store(),
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


def aggregate_reranker_file(filepath):
    groups = {
        "all": init_group(),
        "hard": init_group(),
    }

    seen = set()
    num_lines_read = 0
    num_duplicates_removed = 0

    granularity = None
    reranker_name = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            num_lines_read += 1
            e = json.loads(line)

            # Drop big keys if present to reduce memory pressure
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

            rerank_block = e.get("reranking_results", {})
            metrics = rerank_block.get("metrics", {})

            if granularity is None:
                granularity = rerank_block.get("granularity")
            if reranker_name is None:
                reranker_name = rerank_block.get("reranker")

            target_groups = ["all"]
            if e.get("query_dimension") in HARD_DIMENSIONS:
                target_groups.append("hard")

            for group_name in target_groups:
                groups[group_name]["num_examples"] += 1
                update_metric_store(groups[group_name]["metrics"], metrics)

            if num_lines_read % 100000 == 0:
                print(f"[{os.path.basename(filepath)}] Processed {num_lines_read:,} lines...")

    output = {
        "file": filepath,
        "granularity": granularity,
        "reranker": reranker_name,
        "num_lines_read": num_lines_read,
        "num_duplicates_removed": num_duplicates_removed,
        "all": {
            "num_examples": groups["all"]["num_examples"],
            "metrics": finalize_metric_store(groups["all"]["metrics"]),
        },
        "hard": {
            "num_examples": groups["hard"]["num_examples"],
            "metrics": finalize_metric_store(groups["hard"]["metrics"]),
        },
    }

    return output


def find_jsonl_files(root_dir):
    files = []
    for cur_root, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith("."):
                continue
            if fname.endswith(".jsonl"):
                files.append(os.path.join(cur_root, fname))
    return sorted(files)


def aggregate_reranker_logs(log_root, outpath):
    all_results = {}

    jsonl_files = find_jsonl_files(log_root)
    print(f"Found {len(jsonl_files)} jsonl files under {log_root}")

    for filepath in jsonl_files:
        rel_path = os.path.relpath(filepath, log_root)
        print(f"Aggregating: {rel_path}")
        try:
            all_results[rel_path] = aggregate_reranker_file(filepath)
        except Exception as e:
            all_results[rel_path] = {
                "file": filepath,
                "error": str(e),
            }

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated reranker results to: {outpath}")
    return all_results


if __name__ == "__main__":
    log_root = "src/reranker/reranker_logs"   # change this
    outpath = os.path.join(log_root, "aggregated_reranker_results.json")

    aggregate_reranker_logs(log_root, outpath)
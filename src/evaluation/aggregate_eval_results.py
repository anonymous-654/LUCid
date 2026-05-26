import os
import json
from collections import defaultdict
from typing import Any, Dict, Tuple


# =========================
# Configure here
# =========================

LOG_ROOT = "src/evaluation/evaluation_logs"
OUTPUT_JSON = os.path.join(LOG_ROOT, "aggregated_generation_results.json")

HARD_DIMENSIONS = {
    "age_group",
    "location/country",
    "religion",
}

KNOWN_DIMENSIONS = [
    "age_group",
    "domain",
    "location/country",
    "style_pref",
    "religion",
    "medical_health_condition",
]


# =========================
# Helpers
# =========================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_for_compare(x: Any) -> str:
    return normalize_text(x).casefold()


def get_dimension(row: Dict[str, Any]) -> str:
    return normalize_text(
        row.get("query_dimension", row.get("evaluator_dimension", "unknown"))
    )


def safe_score(matched: int, total: int):
    return matched / total if total > 0 else None


def apply_manual_match_fix(row: Dict[str, Any], raw_match: Any) -> int:
    try:
        raw_match_int = int(raw_match)
    except (TypeError, ValueError):
        raw_match_int = 0

    if raw_match_int == 1:
        return 1

    dim = get_dimension(row)
    if dim != "location/country":
        return raw_match_int

    expected = normalize_text(row.get("evaluator_expected"))
    prediction = normalize_text(row.get("evaluator_prediction"))

    if expected == "United States" and prediction == "United states":
        return 1

    if (
        normalize_for_compare(expected) == "united states"
        and normalize_for_compare(prediction) == "united states"
    ):
        return 1

    return raw_match_int


def init_group() -> Dict[str, Any]:
    return {
        "num_examples": 0,
        "num_matched": 0,
        "num_skipped": 0,
        "by_dimension": defaultdict(lambda: {"matched": 0, "total": 0}),
    }


def finalize_group(group: Dict[str, Any]) -> Dict[str, Any]:
    by_dimension_out = {}

    dims_present = set(group["by_dimension"].keys())
    ordered_dims = [d for d in KNOWN_DIMENSIONS if d in dims_present]
    ordered_dims += sorted(d for d in dims_present if d not in KNOWN_DIMENSIONS)

    for dim in ordered_dims:
        counts = group["by_dimension"][dim]
        by_dimension_out[dim] = {
            "response_accuracy": safe_score(counts["matched"], counts["total"]),
            "num_examples": counts["total"],
            "num_matched": counts["matched"],
        }

    return {
        "response_accuracy": safe_score(group["num_matched"], group["num_examples"]),
        "num_examples": group["num_examples"],
        "num_matched": group["num_matched"],
        "num_skipped": group["num_skipped"],
        "by_dimension": by_dimension_out,
    }


def make_dedup_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    answer_session_ids = row.get("answer_session_ids", [])
    if isinstance(answer_session_ids, list):
        answer_session_ids = tuple(sorted(str(x) for x in answer_session_ids))
    else:
        answer_session_ids = tuple()

    return (
        normalize_text(row.get("query")),
        normalize_text(row.get("query_dimension", row.get("evaluator_dimension"))),
        normalize_text(row.get("expected_category")),
        answer_session_ids,
        normalize_text(row.get("evaluator_expected")),
    )


def update_group(group: Dict[str, Any], row: Dict[str, Any], matched: int) -> None:
    dim = get_dimension(row)

    group["num_examples"] += 1
    group["num_matched"] += matched
    group["by_dimension"][dim]["total"] += 1
    group["by_dimension"][dim]["matched"] += matched


# =========================
# Core aggregation
# =========================

def aggregate_generation_file(filepath: str) -> Dict[str, Any]:
    groups = {
        "all": init_group(),
        "hard": init_group(),
    }

    num_lines_read = 0
    num_duplicates_removed = 0
    seen = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            num_lines_read += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                groups["all"]["num_skipped"] += 1
                groups["hard"]["num_skipped"] += 1
                continue

            row.pop("haystack_sessions", None)
            row.pop("haystack_session_ids", None)
            row.pop("haystack_sessions_ids", None)

            dedup_key = make_dedup_key(row)
            if dedup_key in seen:
                num_duplicates_removed += 1
                continue
            seen.add(dedup_key)

            raw_match = row.get("evaluator_match")
            if raw_match is None:
                groups["all"]["num_skipped"] += 1
                if get_dimension(row) in HARD_DIMENSIONS:
                    groups["hard"]["num_skipped"] += 1
                continue

            matched = apply_manual_match_fix(row, raw_match)

            target_groups = ["all"]
            if get_dimension(row) in HARD_DIMENSIONS:
                target_groups.append("hard")

            for group_name in target_groups:
                update_group(groups[group_name], row, matched)

            if num_lines_read % 100000 == 0:
                print(f"[{os.path.basename(filepath)}] Processed {num_lines_read:,} lines...")

    return {
        "file": filepath,
        "num_lines_read": num_lines_read,
        "num_duplicates_removed": num_duplicates_removed,
        "all": finalize_group(groups["all"]),
        "hard": finalize_group(groups["hard"]),
    }


def find_jsonl_files(root_dir: str):
    files = []
    for cur_root, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith("."):
                continue
            if fname.endswith("_judge_eval.jsonl"):
                files.append(os.path.join(cur_root, fname))
    return sorted(files)


def aggregate_generation_logs(log_root: str, outpath: str):
    all_results = {}

    jsonl_files = find_jsonl_files(log_root)
    print(f"Found {len(jsonl_files)} judge_eval files under {log_root}")

    for filepath in jsonl_files:
        rel_path = os.path.relpath(filepath, log_root)
        print(f"Aggregating: {rel_path}")
        try:
            all_results[rel_path] = aggregate_generation_file(filepath)
        except Exception as e:
            all_results[rel_path] = {
                "file": filepath,
                "error": str(e),
            }

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated generation results to: {outpath}")
    return all_results


if __name__ == "__main__":
    aggregate_generation_logs(LOG_ROOT, OUTPUT_JSON)
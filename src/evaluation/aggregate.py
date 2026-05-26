import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Any, Optional, List


MCQ_DIMENSIONS = {"age_group", "domain", "location/country", "style_pref"}
SUITABILITY_DIMENSIONS = {"religion", "medical_health_condition"}

ALL_DIMENSIONS = MCQ_DIMENSIONS | SUITABILITY_DIMENSIONS
MULTI_SESSION_DIMS = {"domain", "style_pref"}
SINGLE_SESSION_DIMS = ALL_DIMENSIONS - MULTI_SESSION_DIMS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing *_judge_eval.jsonl files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save aggregated results JSON",
    )
    return parser.parse_args()


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Example:
    gpt-5.4-mini_gold_testlog_top5context_jsonformat_useronlyfalse_judge_eval.jsonl

    Extract:
      model   = gpt-5.4-mini
      setting = gold
    """
    suffix = "_judge_eval.jsonl"
    if not filename.endswith(suffix):
        return None

    stem = filename[:-len(suffix)]

    settings = ["gold", "no-retrieval", "oracle-session", "orig-session"]
    for setting in settings:
        marker = f"_{setting}_testlog"
        if marker in stem:
            model = stem.split(marker)[0]
            return {"model": model, "setting": setting}

    return None


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


def apply_manual_match_fix(row: Dict[str, Any], raw_match: Any) -> int:
    """
    Manual patch for the known location/country issue:
    evaluator options may contain both:
      - "United States"
      - "United states"
    and the evaluator sometimes picks the lowercase-s version even though
    the expected answer is effectively the same.

    We treat this as a match only for location/country.
    """
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

    # Specific manual fix requested
    if expected == "United States" and prediction == "United states":
        return 1

    # Slightly more robust version for the same bug pattern:
    # only for location/country, treat capitalization-only mismatch on
    # "United States" as equivalent.
    if normalize_for_compare(expected) == "united states" and normalize_for_compare(prediction) == "united states":
        return 1

    return raw_match_int


def empty_counter_dict(dimensions: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        dim: {"matched": 0, "total": 0}
        for dim in dimensions
    }


def safe_score(matched: int, total: int):
    return matched / total if total > 0 else None


def compute_file_stats(file_path: str) -> Dict[str, Any]:
    overall_matched = 0
    overall_total = 0
    skipped = 0

    by_dim_counts = empty_counter_dict(sorted(ALL_DIMENSIONS))
    unknown_dim_counts = defaultdict(lambda: {"matched": 0, "total": 0})

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            raw_match = row.get("evaluator_match", None)
            if raw_match is None:
                skipped += 1
                continue

            fixed_match = apply_manual_match_fix(row, raw_match)
            dim = get_dimension(row)

            overall_total += 1
            overall_matched += fixed_match

            if dim in by_dim_counts:
                by_dim_counts[dim]["total"] += 1
                by_dim_counts[dim]["matched"] += fixed_match
            else:
                unknown_dim_counts[dim]["total"] += 1
                unknown_dim_counts[dim]["matched"] += fixed_match

    by_dimension = {}
    for dim, counts in by_dim_counts.items():
        by_dimension[dim] = {
            "match_score": safe_score(counts["matched"], counts["total"]),
            "num_examples": counts["total"],
            "num_matched": counts["matched"],
        }

    if unknown_dim_counts:
        by_dimension["__unknown__"] = {}
        for dim, counts in sorted(unknown_dim_counts.items()):
            by_dimension["__unknown__"][dim] = {
                "match_score": safe_score(counts["matched"], counts["total"]),
                "num_examples": counts["total"],
                "num_matched": counts["matched"],
            }

    multi_matched = sum(by_dim_counts[d]["matched"] for d in MULTI_SESSION_DIMS)
    multi_total = sum(by_dim_counts[d]["total"] for d in MULTI_SESSION_DIMS)

    single_matched = sum(by_dim_counts[d]["matched"] for d in SINGLE_SESSION_DIMS)
    single_total = sum(by_dim_counts[d]["total"] for d in SINGLE_SESSION_DIMS)

    return {
        "match_score": safe_score(overall_matched, overall_total),
        "num_examples": overall_total,
        "num_matched": overall_matched,
        "num_skipped": skipped,
        "by_dimension": by_dimension,
        "multi_session_score": safe_score(multi_matched, multi_total),
        "multi_session_num_examples": multi_total,
        "multi_session_num_matched": multi_matched,
        "single_session_score": safe_score(single_matched, single_total),
        "single_session_num_examples": single_total,
        "single_session_num_matched": single_matched,
    }


def aggregate_results(out_dir: str) -> Dict[str, Any]:
    """
    Output format:
    {
      "gpt-5.4-mini": {
        "gold": { ... },
        "orig-session": { ... }
      },
      "Qwen_Qwen3.5-27B-FP8": {
        "gold": { ... }
      }
    }
    """
    aggregated = defaultdict(dict)

    for filename in sorted(os.listdir(out_dir)):
        if not filename.endswith("_judge_eval.jsonl"):
            continue

        parsed = parse_filename(filename)
        if parsed is None:
            print(f"[WARN] Could not parse filename: {filename}")
            continue

        model = parsed["model"]
        setting = parsed["setting"]
        file_path = os.path.join(out_dir, filename)

        stats = compute_file_stats(file_path)
        stats["file"] = filename

        aggregated[model][setting] = stats

        print(
            f"[INFO] model={model}, setting={setting}, "
            f"overall={stats['match_score']}, "
            f"multi_session={stats['multi_session_score']}, "
            f"single_session={stats['single_session_score']}, "
            f"n={stats['num_examples']}"
        )

    return dict(aggregated)


def main():
    args = parse_args()
    results = aggregate_results(args.out_dir)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved aggregated results to: {args.output_json}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd

# Hardcoded paths
ROOT = Path("src/retrieval/retrieval_logs")
OUTFILE = Path("src/retrieval/retrieval_logs/retrieval_metrics_summary.csv")

# Only keep files whose basename ends with this suffix.
# Set to None if you want all files.
ENDS_WITH = "1953"


def extract_haystack_size(filename: str):
    """
    Examples:
      lucid_b.json_retrievallog_session_flat-contriever_1953 -> 200
      older full-size retrieval log names are also supported.
    """
    lucid_sizes = {"c": 30, "s": 50, "b": 200, "l": 500}
    m = re.search(r"lucid_([csbl])\.json", filename)
    if m:
        return lucid_sizes[m.group(1)]

    m = re.search(r"full_(\d+)", filename)
    return int(m.group(1)) if m else None


def flatten_metrics(metrics_dict, prefix=""):
    """
    Recursively flatten nested metric dicts into a single-level dict.
    Keeps only numeric values.
    """
    flat = {}
    for k, v in metrics_dict.items():
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def read_jsonl_metrics(path: Path):
    """
    Reads a retrieval log file line-by-line as JSONL and extracts:
      row["retrieval_results"]["metrics"]
    Averages all numeric metrics across rows.
    """
    metric_rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON in {path} line {line_num}")
                continue

            retrieval_results = row.get("retrieval_results", {})
            metrics = retrieval_results.get("metrics", {})

            if not isinstance(metrics, dict) or not metrics:
                continue

            flat = flatten_metrics(metrics)
            if flat:
                metric_rows.append(flat)

    if not metric_rows:
        return {}

    df = pd.DataFrame(metric_rows)
    return df.mean(numeric_only=True).to_dict() | {"num_examples": len(df)}


def main():
    rows = []

    if not ROOT.exists():
        raise FileNotFoundError(f"Root folder not found: {ROOT}")

    for file_path in ROOT.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip obvious non-log files
        if file_path.suffix in {".csv", ".tsv", ".parquet", ".pkl", ".py"}:
            continue

        if ENDS_WITH is not None and not file_path.name.endswith(ENDS_WITH):
            continue

        # Expect structure like:
        # src/retrieval/retrieval_logs/{retriever}/{granularity}/filename
        parts = file_path.relative_to(ROOT).parts
        if len(parts) < 3:
            # not in expected retriever/granularity/file layout
            continue

        retriever = parts[0]
        granularity = parts[1]
        filename = file_path.name
        haystack_size = extract_haystack_size(filename)

        metrics = read_jsonl_metrics(file_path)
        if not metrics:
            print(f"[WARN] No metrics found in {file_path}")
            continue

        row = {
            "haystack_size": haystack_size,
            "retriever": retriever,
            "granularity": granularity,
            "file": str(file_path),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        print("[WARN] No matching files found.")
        return

    df = pd.DataFrame(rows)

    # Sort columns: metadata first, then metric columns alphabetically
    meta_cols = ["haystack_size", "retriever", "granularity", "file", "num_examples"]
    metric_cols = sorted([c for c in df.columns if c not in meta_cols])
    ordered_cols = [c for c in meta_cols if c in df.columns] + metric_cols
    df = df[ordered_cols]

    # Nice sorting
    sort_cols = [c for c in ["haystack_size", "retriever", "granularity"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTFILE, index=False)

    print(f"[OK] Wrote {len(df)} rows to {OUTFILE}")


if __name__ == "__main__":
    main()

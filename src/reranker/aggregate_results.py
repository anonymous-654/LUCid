#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def iter_records(path: Path):
    """
    Load records from:
      - .jsonl (one JSON object per line)
      - .json (list or dict)
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path} line {line_num}: invalid JSON ({e})")

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for row in data:
                yield row
        elif isinstance(data, dict):
            if isinstance(data.get("results"), list):
                for row in data["results"]:
                    yield row
            else:
                yield data
        else:
            raise ValueError(f"{path}: unsupported JSON structure")

    else:
        raise ValueError(f"{path}: unsupported file type")


def extract_metrics(record: dict):
    """
    Adjust this if your structure changes.
    Expected:
        record["reranking_results"]["metrics"]
    """
    rr = record.get("reranking_results", {})
    metrics = rr.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def extract_model_name(path: Path):
    """
    Extracts model name from filename like:
    session_<MODEL_NAME>.jsonl
    """
    name = path.name

    if name.startswith("session_") and name.endswith(".jsonl"):
        return name[len("session_"):-len(".jsonl")]

    # fallback (in case pattern differs slightly)
    return name.replace("session_", "").replace(".jsonl", "")


def summarize_file(path: Path):
    metric_values = {}
    num_records = 0

    for record in iter_records(path):
        metrics = extract_metrics(record)
        if not metrics:
            continue

        num_records += 1

        for name, value in metrics.items():
            try:
                metric_values.setdefault(name, []).append(float(value))
            except (TypeError, ValueError):
                continue

    summary = {
        "file_name": path.name,
        "model_name": extract_model_name(path),
        "file_path": str(path),
        "num_entries_with_metrics": num_records,
    }

    for name, values in sorted(metric_values.items()):
        summary[name] = mean(values) if values else None

    return summary


def save_csv(rows, out_path: Path):
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    base_cols = ["file_name", "file_path", "num_entries_with_metrics"]
    metric_cols = sorted(k for k in all_keys if k not in base_cols)
    fieldnames = base_cols + metric_cols

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(rows, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Compute average metrics per file in a folder."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.jsonl")
    parser.add_argument("--output_file", type=Path, required=True)

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Folder not found: {args.input_dir}")

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found with pattern: {args.pattern}")

    summaries = []

    for path in files:
        try:
            summaries.append(summarize_file(path))
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not summaries:
        raise SystemExit("No valid data found.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.output_file.suffix.lower() == ".csv":
        save_csv(summaries, args.output_file)
    elif args.output_file.suffix.lower() == ".json":
        save_json(summaries, args.output_file)
    else:
        raise SystemExit("Output must be .csv or .json")

    print(f"Saved results → {args.output_file}")


if __name__ == "__main__":
    main()